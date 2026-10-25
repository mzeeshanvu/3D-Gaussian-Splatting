"""
Saving the trained Gaussians.

The PLY written here uses the field names the 3D Gaussian Splatting community
settled on, so the result drops straight into the usual viewers -- SuperSplat,
antimatter15's splat, the SIBR viewer -- without conversion:

    x y z                       centre
    nx ny nz                    unused, written as zeros for compatibility
    f_dc_0..2                   the DC spherical-harmonic coefficients
    f_rest_0..N                 higher-order coefficients, if any
    opacity                     the raw logit, not the sigmoid
    scale_0..2                  log scales, not the exponentiated values
    rot_0..3                    quaternion, w first

Note that opacity and scale are stored *before* their activation functions.
That looks odd but is what the format specifies, and writing the activated
values instead produces a file that loads and renders wrongly.

That layout is what a splat renderer wants, but it carries no ``red``/
``green``/``blue`` fields, so an ordinary point-cloud viewer shows the scene
in a flat default grey -- the colour is there, encoded in ``f_dc_*``, but
nothing outside a splat viewer knows to decode it. :func:`save_point_cloud`
therefore writes a second, plain PLY beside it with the colours already
decoded, so the trained model can be opened in colour anywhere.
"""

import numpy as np
import torch


def save_ply(model, path):
    """
    Write the model in the standard 3DGS PLY layout.

    :returns: (path, number of Gaussians)
    """
    with torch.no_grad():
        xyz = model.means.detach().cpu().numpy()
        n = len(xyz)
        sh = model.sh.detach().cpu().numpy()            # (n, coeffs, 3)
        dc = sh[:, 0, :]
        rest = sh[:, 1:, :].reshape(n, -1) if sh.shape[1] > 1 else np.zeros((n, 0))
        opacity = model.opacities_raw.detach().cpu().numpy().reshape(n, 1)
        scale = model.log_scales.detach().cpu().numpy()
        rot = model.quats.detach().cpu().numpy()
        rot = rot / np.linalg.norm(rot, axis=1, keepdims=True).clip(1e-12)

    names = ["x", "y", "z", "nx", "ny", "nz"]
    names += [f"f_dc_{i}" for i in range(3)]
    names += [f"f_rest_{i}" for i in range(rest.shape[1])]
    names += ["opacity"] + [f"scale_{i}" for i in range(3)]
    names += [f"rot_{i}" for i in range(4)]

    rows = np.concatenate(
        [xyz, np.zeros_like(xyz), dc, rest, opacity, scale, rot],
        axis=1).astype(np.float32)

    with open(path, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(f"element vertex {n}\n".encode())
        for nm in names:
            f.write(f"property float {nm}\n".encode())
        f.write(b"end_header\n")
        f.write(rows.tobytes())
    return path, n


def gaussian_colors(model, dirs=None):
    """
    Decode the per-Gaussian colour that is stored in the SH coefficients.

    The PLY keeps the DC coefficient rather than the colour itself, which is
    why a trained scene looks grey anywhere that does not know the convention.
    Undoing it is just the constructor's encoding run backwards.

    :param dirs: optional (n, 3) viewing directions; at ``sh_degree > 0`` the
                 colour depends on them. Left out, only the DC term is used,
                 which is the view-independent base colour.
    :returns: (n, 3) RGB in 0..1
    """
    with torch.no_grad():
        rgb = model.colors(dirs).clamp(0.0, 1.0)
    return rgb.detach().cpu().numpy()


def gaussian_point_cloud(model, min_opacity=0.05):
    """
    The trained scene as a plain coloured point cloud.

    Every Gaussian becomes one point at its centre, carrying its decoded
    colour. The near-transparent ones are dropped: training leaves a haze of
    very low-opacity Gaussians that a splat renderer correctly makes almost
    invisible, but a point viewer would draw at full strength and bury the
    object in fog.

    :param min_opacity: discard Gaussians fainter than this (0 keeps all)
    :returns: (xyz (m, 3), rgb (m, 3) in 0..1)
    """
    with torch.no_grad():
        xyz = model.means.detach().cpu().numpy()
        alpha = model.opacities.detach().cpu().numpy().reshape(-1)
    rgb = gaussian_colors(model)

    keep = alpha >= min_opacity
    # A threshold that rejected everything would be worse than none at all.
    if not keep.any():
        keep = np.ones_like(alpha, dtype=bool)
    return xyz[keep], rgb[keep]


def save_point_cloud(path, xyz, rgb):
    """
    Write ``x y z red green blue`` in binary little-endian PLY.

    The companion to :func:`save_ply`: the same scene in the one layout every
    viewer understands, so the result can be inspected in colour without a
    splat renderer.

    :param rgb: (n, 3) in 0..1, written as 8-bit
    :returns: (path, number of points)
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = (np.clip(np.asarray(rgb), 0.0, 1.0) * 255.0).astype(np.uint8)
    n = len(xyz)

    with open(path, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(f"element vertex {n}\n".encode())
        for nm in ("x", "y", "z"):
            f.write(f"property float {nm}\n".encode())
        for nm in ("red", "green", "blue"):
            f.write(f"property uchar {nm}\n".encode())
        f.write(b"end_header\n")
        rows = np.empty(n, dtype=np.dtype([("x", "<f4"), ("y", "<f4"),
                                           ("z", "<f4"), ("red", "u1"),
                                           ("green", "u1"), ("blue", "u1")]))
        rows["x"], rows["y"], rows["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        rows["red"], rows["green"], rows["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        f.write(rows.tobytes())
    return path, n


def save_colored_ply(model, path, min_opacity=0.05):
    """
    Write the trained Gaussians as a coloured point cloud.

    Convenience wrapper over :func:`gaussian_point_cloud` and
    :func:`save_point_cloud`.

    :returns: (path, number of points written)
    """
    xyz, rgb = gaussian_point_cloud(model, min_opacity=min_opacity)
    return save_point_cloud(path, xyz, rgb)


def save_image(img, path):
    """Write an (H, W, 3) tensor or array in 0..1 as an 8-bit PNG."""
    import cv2
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    rgb = np.clip(img, 0.0, 1.0)
    cv2.imwrite(str(path), cv2.cvtColor((rgb * 255).astype(np.uint8),
                                        cv2.COLOR_RGB2BGR))
    return path


def load_ply(path, sh_degree=None):
    """
    Rebuild a :class:`gsplat.model.GaussianModel` from a saved PLY.

    The inverse of :func:`save_ply`. Needed to render a trained scene without
    retraining it -- to produce novel views, or to carry on optimising from a
    checkpoint.

    :param sh_degree: inferred from the number of f_rest fields when omitted
    :returns: the reconstructed model
    """
    import math

    from .model import GaussianModel, SH_C0

    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file")
        names, n = [], 0
        while True:
            parts = f.readline().split()
            if parts[0] == b"element":
                n = int(parts[2])
            elif parts[0] == b"property":
                names.append(parts[2].decode())
            elif parts[0] == b"end_header":
                break
        rows = np.frombuffer(f.read(4 * len(names) * n),
                             dtype="<f4").reshape(n, len(names))

    col = {nm: i for i, nm in enumerate(names)}
    xyz = rows[:, [col["x"], col["y"], col["z"]]]
    dc = rows[:, [col["f_dc_0"], col["f_dc_1"], col["f_dc_2"]]]

    n_rest = sum(1 for nm in names if nm.startswith("f_rest_"))
    if sh_degree is None:
        sh_degree = int(round(math.sqrt(n_rest / 3 + 1))) - 1 if n_rest else 0

    # save_ply stores the DC coefficient, not the colour; invert that here so
    # the constructor's own encoding round-trips exactly.
    model = GaussianModel(xyz, SH_C0 * dc + 0.5, sh_degree=sh_degree)
    with torch.no_grad():
        if n_rest:
            rest = rows[:, [col[f"f_rest_{i}"] for i in range(n_rest)]]
            model.sh[:, 1:, :] = torch.from_numpy(
                rest.reshape(n, -1, 3).copy())
        model.opacities_raw.copy_(torch.from_numpy(
            rows[:, [col["opacity"]]].copy()))
        model.log_scales.copy_(torch.from_numpy(
            rows[:, [col[f"scale_{i}"] for i in range(3)]].copy()))
        model.quats.copy_(torch.from_numpy(
            rows[:, [col[f"rot_{i}"] for i in range(4)]].copy()))
    return model
