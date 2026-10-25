"""
3D Gaussian splatting, implemented from scratch.

Every stage of the method is written out here -- the Gaussian parameterisation,
the 3D-to-2D covariance projection, the tile-based rasteriser, the alpha
compositing, the loss and the adaptive density control. PyTorch supplies the
tensor maths and differentiates the code in :mod:`gsplat.render`; nothing is
imported from an existing splatting library.

The scene is initialised from the Structure-from-Motion result in ``output/``:
the recovered camera poses say where each training photograph was taken, and
the sparse point cloud seeds the Gaussians. That dependency is the point --
splatting refines geometry, it does not discover it.

Reference
---------
Kerbl, Kopanas, Leimkuehler and Drettakis, "3D Gaussian Splatting for
Real-Time Radiance Field Rendering", SIGGRAPH 2023.
"""

from . import data, densify, io, loss, model, paths, render, report

__all__ = ["data", "densify", "io", "loss", "model", "paths", "render",
           "report"]
