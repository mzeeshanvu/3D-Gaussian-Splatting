"""
The rasteriser: turning 3D Gaussians into an image.

This is the core of the method, and the whole of it is differentiable, so the
gradient of a photometric loss reaches every Gaussian parameter.

The six stages
--------------
1. Transform each Gaussian's centre into camera space and drop anything behind
   the near plane or outside the frustum.
2. Project the centres to pixel coordinates.
3. Push the 3D covariance through the same projection to get a 2D covariance.
   Perspective projection is not linear, so this uses its Jacobian -- the
   local linear approximation at each Gaussian's own depth.
4. Work out the pixel footprint of each Gaussian and which image tiles it
   touches.
5. Within each tile, sort the Gaussians by depth.
6. Alpha-composite front to back until the tile is opaque.

Stages 4-6 are what the paper implements as a CUDA kernel. Here they are a
loop over tiles with vectorised tensor maths inside, which is the same
algorithm at a speed a CPU can manage.
"""

import math

import torch

# A Gaussian thinner than about one pixel would alias badly or vanish between
# sample points, so the 2D covariance is given this much isotropic blur. The
# paper uses the same dilation.
SCREEN_BLUR = 0.3

# Beyond three standard deviations a Gaussian contributes nothing visible, so
# that is where its footprint is cut off.
RADIUS_SIGMAS = 3.0


def project_gaussians(means, cov3d, R, t, K, width, height, near=0.01):
    """
    Stages 1-3: camera transform, projection, and the 2D covariance.

    :param means: (n, 3) world-space centres
    :param cov3d: (n, 3, 3) world-space covariances
    :param R: (3, 3) world-to-camera rotation
    :param t: (3,) world-to-camera translation
    :returns: dict with the surviving Gaussians' 2D centres, 2D covariances,
              depths, camera-space directions and the mask that selected them
    """
    cam = means @ R.T + t                              # (n, 3) camera space
    z = cam[:, 2]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Project. Guard the divide so culled Gaussians cannot produce NaNs that
    # would poison the backward pass even though they are masked out later.
    z_safe = torch.where(z > near, z, torch.full_like(z, near))
    u = fx * cam[:, 0] / z_safe + cx
    v = fy * cam[:, 1] / z_safe + cy

    # The Jacobian of (x, y, z) -> (u, v) at this Gaussian's depth. Beyond the
    # frustum edge the linearisation degrades badly, so the ratios are clamped
    # a little outside the visible field rather than left to blow up.
    lim_x = 1.3 * (0.5 * width) / fx
    lim_y = 1.3 * (0.5 * height) / fy
    tx = torch.clamp(cam[:, 0] / z_safe, -lim_x, lim_x) * z_safe
    ty = torch.clamp(cam[:, 1] / z_safe, -lim_y, lim_y) * z_safe

    zero = torch.zeros_like(z_safe)
    J = torch.stack([
        torch.stack([fx / z_safe, zero, -fx * tx / (z_safe * z_safe)], dim=-1),
        torch.stack([zero, fy / z_safe, -fy * ty / (z_safe * z_safe)], dim=-1),
    ], dim=1)                                          # (n, 2, 3)

    # Sigma_2D = J R Sigma R^T J^T, with R taking world covariance to camera.
    W = J @ R                                          # (n, 2, 3)
    cov2d = W @ cov3d @ W.transpose(1, 2)              # (n, 2, 2)

    # Dilate so nothing is thinner than a pixel.
    cov2d = cov2d + SCREEN_BLUR * torch.eye(2, device=cov2d.device).expand_as(cov2d)

    # Keep what is in front of the camera and roughly on screen. The margin
    # lets a Gaussian whose centre is just outside still paint its edge.
    radius = _footprint_radius(cov2d)
    visible = ((z > near)
               & (u + radius > 0) & (u - radius < width)
               & (v + radius > 0) & (v - radius < height))

    return {
        "uv": torch.stack([u, v], dim=1)[visible],
        "cov2d": cov2d[visible],
        "depth": z[visible],
        "radius": radius[visible],
        "cam": cam[visible],
        "visible": visible,
    }


def _footprint_radius(cov2d):
    """
    Three sigma along the major axis of each 2D Gaussian, in pixels.

    The larger eigenvalue of a symmetric 2x2 matrix has a closed form, so no
    eigendecomposition is needed:  lambda = m + sqrt(m^2 - det), m = trace/2.
    """
    a, b, c = cov2d[:, 0, 0], cov2d[:, 0, 1], cov2d[:, 1, 1]
    mid = 0.5 * (a + c)
    det = a * c - b * b
    disc = torch.clamp(mid * mid - det, min=1e-9)
    lambda_max = mid + torch.sqrt(disc)
    return RADIUS_SIGMAS * torch.sqrt(torch.clamp(lambda_max, min=1e-9))


def _inverse_cov2d(cov2d):
    """
    The 'conic' form: the inverse of each 2x2 covariance, as (a, b, c).

    Inverting analytically rather than with `torch.inverse` keeps this cheap
    and keeps the gradient simple.
    """
    a, b, c = cov2d[:, 0, 0], cov2d[:, 0, 1], cov2d[:, 1, 1]
    det = (a * c - b * b).clamp_min(1e-9)
    return c / det, -b / det, a / det


def render(model, R, t, K, width, height, background=(0.0, 0.0, 0.0),
           tile=16, return_aux=False):
    """
    Render the model from one camera.

    :param model: a :class:`gsplat.model.GaussianModel`
    :param R, t, K: world-to-camera pose and intrinsics for this view
    :param background: colour showing through where the scene is transparent
    :param tile: tile edge in pixels; smaller means fewer Gaussians considered
                 per pixel but more Python iterations
    :param return_aux: also return per-Gaussian screen-space data, which the
                       densification step needs
    :returns: (H, W, 3) image, or (image, aux) when `return_aux`
    """
    device = model.means.device
    R = torch.as_tensor(R, dtype=torch.float32, device=device)
    t = torch.as_tensor(t, dtype=torch.float32, device=device)
    K = torch.as_tensor(K, dtype=torch.float32, device=device)
    bg = torch.as_tensor(background, dtype=torch.float32, device=device)

    p = project_gaussians(model.means, model.covariance(), R, t, K,
                          width, height)
    vis = p["visible"]

    image = bg.view(1, 1, 3).repeat(height, width, 1)
    if vis.sum() == 0:
        return (image, None) if return_aux else image

    # Colour each surviving Gaussian for this viewpoint.
    centre = -R.T @ t
    dirs = model.means[vis] - centre
    colors = model.colors(dirs)[vis] if model.sh_degree > 0 else model.colors()[vis]
    opacity = model.opacities[vis].squeeze(1)

    uv, depth, radius = p["uv"], p["depth"], p["radius"]
    # Densification is driven by d(loss)/d(screen position), and uv is an
    # intermediate value whose gradient autograd would otherwise discard.
    if return_aux and uv.requires_grad:
        uv.retain_grad()
    conic_a, conic_b, conic_c = _inverse_cov2d(p["cov2d"])

    # -- stage 4: which tiles does each Gaussian touch? --------------------
    n_tx = (width + tile - 1) // tile
    n_ty = (height + tile - 1) // tile
    tx0 = torch.clamp(((uv[:, 0] - radius) / tile).floor().long(), 0, n_tx - 1)
    tx1 = torch.clamp(((uv[:, 0] + radius) / tile).ceil().long(), 0, n_tx)
    ty0 = torch.clamp(((uv[:, 1] - radius) / tile).floor().long(), 0, n_ty - 1)
    ty1 = torch.clamp(((uv[:, 1] + radius) / tile).ceil().long(), 0, n_ty)

    ys = torch.arange(height, device=device, dtype=torch.float32)
    xs = torch.arange(width, device=device, dtype=torch.float32)

    rows = []
    for ty in range(n_ty):
        y0, y1 = ty * tile, min((ty + 1) * tile, height)
        cols = []
        for tx in range(n_tx):
            x0, x1 = tx * tile, min((tx + 1) * tile, width)
            hit = (tx0 <= tx) & (tx1 > tx) & (ty0 <= ty) & (ty1 > ty)
            idx = hit.nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                cols.append(bg.view(1, 1, 3).repeat(y1 - y0, x1 - x0, 1))
                continue

            # -- stage 5: depth order within the tile ----------------------
            order = torch.argsort(depth[idx])
            g = idx[order]

            # Offset from each Gaussian centre to every pixel in the tile.
            px = xs[x0:x1].view(1, 1, -1)              # (1, 1, w)
            py = ys[y0:y1].view(1, -1, 1)              # (1, h, 1)
            dx = px - uv[g, 0].view(-1, 1, 1)
            dy = py - uv[g, 1].view(-1, 1, 1)

            power = -0.5 * (conic_a[g].view(-1, 1, 1) * dx * dx
                            + 2.0 * conic_b[g].view(-1, 1, 1) * dx * dy
                            + conic_c[g].view(-1, 1, 1) * dy * dy)
            alpha = (opacity[g].view(-1, 1, 1) * torch.exp(power.clamp(max=0.0))
                     ).clamp(0.0, 0.99)

            # -- stage 6: front-to-back alpha compositing ------------------
            # Transmittance reaching Gaussian i is the product of (1 - alpha)
            # over everything nearer. An exclusive cumulative product gives all
            # of them at once, which keeps the whole tile one tensor op.
            one_minus = 1.0 - alpha
            trans = torch.cumprod(one_minus, dim=0) / one_minus.clamp_min(1e-10)
            weight = alpha * trans                      # (g, h, w)

            rgb = (weight.unsqueeze(-1) * colors[g].view(-1, 1, 1, 3)).sum(0)
            remaining = trans[-1] * one_minus[-1]
            cols.append(rgb + remaining.unsqueeze(-1) * bg.view(1, 1, 3))
        rows.append(torch.cat(cols, dim=1))
    image = torch.cat(rows, dim=0)

    if return_aux:
        return image, {"uv": uv, "radius": radius, "visible": vis,
                       "depth": depth, "n_visible": int(vis.sum())}
    return image
