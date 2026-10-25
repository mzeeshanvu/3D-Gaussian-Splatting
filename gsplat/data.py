"""
Training views and the initial point cloud, read from the SfM output.

Splatting needs two things the reconstruction already produced: a photograph
with a known camera pose for every view, and a sparse set of 3D points to seed
the Gaussians. Both come out of the Structure-from-Motion stage, which lives
in its own repository and is included here as the ``sfm/`` submodule;
:mod:`gsplat.paths` knows where to look for its output. Reading the poses and
the cloud from the same run is what keeps the two halves in one coordinate
frame.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import json
import numpy as np

from . import paths

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class View:
    """One training photograph and the pose it was taken from."""

    image: np.ndarray        # (H, W, 3) RGB in 0..1
    R: np.ndarray            # (3, 3) world-to-camera rotation
    t: np.ndarray            # (3,)   world-to-camera translation
    K: np.ndarray            # (3, 3) intrinsics, scaled to this image size
    name: str

    @property
    def size(self):
        """(width, height) of the stored image."""
        h, w = self.image.shape[:2]
        return w, h

    @property
    def centre(self):
        """World-space camera centre, ``C = -R^T t``."""
        return -self.R.T @ self.t


def _scale_intrinsics(K, from_size, to_size):
    """
    Rescale a camera matrix after the image has been resized.

    Training at reduced resolution is the only way this runs on a CPU, and the
    intrinsics have to follow the pixels or every projection lands in the wrong
    place.
    """
    (w0, h0), (w1, h1) = from_size, to_size
    S = np.diag([w1 / w0, h1 / h0, 1.0])
    return S @ K


def load_scene(dataset="templeRing", max_width=None, n_views=None,
               sfm_output=None):
    """
    Load every registered view plus the sparse cloud that seeds the Gaussians.

    :param max_width: downscale images so their width is at most this, and
                      scale the intrinsics to match. CPU training is only
                      practical at small resolutions.
    :param n_views: use only the first N views (for quick experiments)
    :param sfm_output: an SfM output folder to read instead of searching; see
                       :func:`gsplat.paths.find_sfm_output`
    :returns: (views, points (n, 3), colors (n, 3) in 0..1)
    """
    out = paths.find_sfm_output(dataset, explicit=sfm_output)
    traj_path = out / paths.TRAJECTORY
    ply_path = out / paths.POINT_CLOUD

    traj = json.loads(traj_path.read_text())
    params = traj["parameters"]
    if n_views is not None:
        params = params[:n_views]

    # The trajectory stores poses by registration order, which is the sorted
    # image order the pipeline used. The registry that says where those images
    # live belongs to the SfM project, so it is read from the submodule rather
    # than duplicated here.
    paths.add_sfm_to_path()
    from src.config import get_dataset
    spec = get_dataset(dataset)
    files = sorted(spec.path.glob(spec.pattern))

    views = []
    for i, p in enumerate(params):
        # Open3D writes these column-major.
        E = np.array(p["extrinsic"], float).reshape(4, 4).T
        K = np.array(p["intrinsic"]["intrinsic_matrix"], float).reshape(3, 3).T
        w0, h0 = p["intrinsic"]["width"], p["intrinsic"]["height"]

        bgr = cv2.imread(str(files[i]))
        if bgr is None:
            raise FileNotFoundError(files[i])
        if max_width is not None and bgr.shape[1] > max_width:
            scale = max_width / bgr.shape[1]
            bgr = cv2.resize(bgr, (max_width, int(round(bgr.shape[0] * scale))),
                             interpolation=cv2.INTER_AREA)
        h, w = bgr.shape[:2]

        views.append(View(
            image=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0,
            R=E[:3, :3], t=E[:3, 3],
            K=_scale_intrinsics(K, (w0, h0), (w, h)),
            name=files[i].name))

    points, colors = _read_ply(ply_path)
    return views, points, colors


def _read_ply(path):
    """
    Read vertices and colours from a PLY, ascii or binary little-endian.

    Written here rather than taken from Open3D so the splatting half of the
    project has no hard dependency on it.
    """
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file")

        fmt, n, props = None, 0, []
        in_vertex = False
        while True:
            line = f.readline()
            if not line:
                raise ValueError("unexpected end of PLY header")
            parts = line.split()
            if parts[0] == b"format":
                fmt = parts[1].decode()
            elif parts[0] == b"element":
                in_vertex = parts[1] == b"vertex"
                if in_vertex:
                    n = int(parts[2])
            elif parts[0] == b"property" and in_vertex:
                props.append((parts[1].decode(), parts[2].decode()))
            elif parts[0] == b"end_header":
                break

        names = [p[1] for p in props]
        if fmt == "ascii":
            rows = np.array([f.readline().split() for _ in range(n)], float)
        else:
            np_fmt = {"float": "f4", "float32": "f4", "double": "f8",
                      "uchar": "u1", "uint8": "u1", "int": "i4"}
            order = "<" if "little" in fmt else ">"
            dtype = np.dtype([(nm, order + np_fmt[ty]) for ty, nm in props])
            rec = np.frombuffer(f.read(dtype.itemsize * n), dtype=dtype, count=n)
            rows = np.stack([rec[nm].astype(float) for nm in names], axis=1)

    idx = {nm: k for k, nm in enumerate(names)}
    xyz = rows[:, [idx["x"], idx["y"], idx["z"]]]
    if "red" in idx:
        rgb = rows[:, [idx["red"], idx["green"], idx["blue"]]]
        # uchar colours arrive as 0..255, floats already as 0..1.
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
    else:
        rgb = np.full_like(xyz, 0.5)
    return xyz.astype(np.float32), rgb.astype(np.float32)
