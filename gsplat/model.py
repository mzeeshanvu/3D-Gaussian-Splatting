"""
The scene representation: a set of anisotropic 3D Gaussians.

Each Gaussian carries a position, a shape, an opacity and a colour, and every
one of those is a free parameter optimised against the training photographs.

Parameterisation
----------------
The shape is a 3x3 covariance, but a covariance cannot be optimised directly --
gradient descent would quickly push it somewhere that is not positive
semi-definite, which has no geometric meaning. It is therefore stored
factorised as a scale and a rotation,

    Sigma = R S S^T R^T

with ``S = diag(s)`` and ``R`` from a quaternion. Any s and q give a valid
covariance, so the optimiser cannot leave the valid set.

Two more quantities are stored through a squashing function for the same
reason: scales as ``log s`` (keeping s > 0) and opacity as a logit (keeping
alpha in 0..1).
"""

import numpy as np
import torch
import torch.nn as nn

# Zeroth-order spherical-harmonic basis value. The DC coefficient is divided by
# this so that a freshly initialised Gaussian renders at exactly its seed colour.
SH_C0 = 0.28209479177387814


def inverse_sigmoid(x, eps=1e-6):
    """Logit, so that an opacity can be stored unconstrained."""
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x / (1.0 - x))


def quaternion_to_rotation(q):
    """
    Rotation matrices from (possibly unnormalised) quaternions.

    Normalising inside means the optimiser never has to keep ``q`` on the unit
    sphere; any non-zero quaternion maps to a valid rotation.

    :param q: (n, 4) as (w, x, y, z)
    :returns: (n, 3, 3)
    """
    q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], dim=1)


def nearest_neighbour_spacing(points, k=3, chunk=2048):
    """
    Mean distance to the k nearest other points, per point.

    Used to give each seed Gaussian a sensible starting radius: dense regions
    begin small, sparse regions begin large. Chunked so the pairwise distance
    matrix is never held in full.
    """
    pts = torch.as_tensor(points, dtype=torch.float32)
    out = torch.empty(len(pts))
    for i in range(0, len(pts), chunk):
        block = pts[i:i + chunk]
        d = torch.cdist(block, pts)
        # Column 0 of the sorted row is the point itself, at distance zero.
        out[i:i + len(block)] = d.topk(k + 1, largest=False).values[:, 1:].mean(1)
    return out.clamp_min(1e-7)


class GaussianModel(nn.Module):
    """
    The optimisable scene.

    :param points: (n, 3) seed positions, from the SfM cloud
    :param colors: (n, 3) seed RGB in 0..1
    :param sh_degree: 0 keeps colour constant with viewing angle; higher
                      degrees add view-dependent appearance (gloss, specular
                      highlights) at 3*(d+1)^2 coefficients per Gaussian
    """

    def __init__(self, points, colors, sh_degree=0, init_opacity=0.1):
        super().__init__()
        n = len(points)
        self.sh_degree = sh_degree

        spacing = nearest_neighbour_spacing(points)
        scales = torch.log(spacing).unsqueeze(1).repeat(1, 3)

        quats = torch.zeros(n, 4)
        quats[:, 0] = 1.0                                  # identity rotation

        # The DC term holds the seed colour; higher orders start at zero so the
        # initial render is exactly the input colours.
        n_coeffs = (sh_degree + 1) ** 2
        sh = torch.zeros(n, n_coeffs, 3)
        sh[:, 0, :] = (torch.as_tensor(colors, dtype=torch.float32) - 0.5) / SH_C0

        self.means = nn.Parameter(torch.as_tensor(points, dtype=torch.float32))
        self.log_scales = nn.Parameter(scales)
        self.quats = nn.Parameter(quats)
        self.sh = nn.Parameter(sh)
        self.opacities_raw = nn.Parameter(
            torch.full((n, 1), float(inverse_sigmoid(init_opacity))))

    # -- derived quantities ------------------------------------------------

    @property
    def n(self):
        return self.means.shape[0]

    @property
    def scales(self):
        return torch.exp(self.log_scales)

    @property
    def opacities(self):
        return torch.sigmoid(self.opacities_raw)

    def covariance(self):
        """
        The 3D covariance of every Gaussian, ``Sigma = R S S^T R^T``.

        :returns: (n, 3, 3)
        """
        R = quaternion_to_rotation(self.quats)
        S = torch.diag_embed(self.scales)
        M = R @ S
        return M @ M.transpose(1, 2)

    def colors(self, dirs=None):
        """
        Colour per Gaussian, evaluated for a viewing direction.

        At degree 0 this is just the DC term and `dirs` is ignored. Higher
        degrees add the direction-dependent bands, which is what lets a
        surface look different from different angles.

        :param dirs: (n, 3) unit vectors from camera to Gaussian
        :returns: (n, 3) RGB, clamped to be non-negative
        """
        c = SH_C0 * self.sh[:, 0, :] + 0.5
        if self.sh_degree > 0 and dirs is not None:
            c = c + _sh_higher_order(self.sh, dirs, self.sh_degree)
        return c.clamp_min(0.0)


def _sh_higher_order(sh, dirs, degree):
    """
    Bands 1..degree of the real spherical-harmonic expansion.

    Only the terms the paper uses; written out rather than looped so the
    constants stay visible.
    """
    d = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-12)
    x, y, z = d[:, 0:1], d[:, 1:2], d[:, 2:3]

    out = torch.zeros_like(sh[:, 0, :])
    if degree >= 1:
        c1 = 0.4886025119029199
        out = out + c1 * (-y * sh[:, 1, :] + z * sh[:, 2, :] - x * sh[:, 3, :])
    if degree >= 2:
        xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
        out = (out
               + 1.0925484305920792 * xy * sh[:, 4, :]
               + -1.0925484305920792 * yz * sh[:, 5, :]
               + 0.31539156525252005 * (2.0 * zz - xx - yy) * sh[:, 6, :]
               + -1.0925484305920792 * xz * sh[:, 7, :]
               + 0.5462742152960396 * (xx - yy) * sh[:, 8, :])
    return out
