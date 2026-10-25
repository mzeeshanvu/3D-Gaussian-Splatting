#!/usr/bin/env python
"""
Open the trained Gaussians in the Open3D viewer, in colour.

Each Gaussian is drawn as one coloured point at its centre. This is *not* a
splat render -- the per-Gaussian scale, rotation and opacity need SuperSplat
or another splat renderer to mean anything, which is why the surface reads as
dots rather than as continuous geometry. What it does show is the refined
geometry and its colour, which `gaussians.ply` alone will not in any ordinary
viewer: the 3DGS layout keeps colour in the spherical-harmonic coefficients
and carries no red/green/blue fields, so everything that is not a splat
renderer draws the scene in a flat default grey.

Usage
-----
    python view_splat.py                     # the trained scene
    python view_splat.py --min-opacity 0.2   # only the solid Gaussians
    python view_splat.py --no-cameras        # splats only
    python view_splat.py --dataset templeRing_rot90ccw

The recovered cameras are drawn alongside, each a wireframe frustum with a dot
at its exact optical centre, C = -R^T t. The frustum apex is already at that
point, but a line set renders only its edges, so without the dot there is
nothing marking the position itself.

Controls: drag to orbit, scroll to zoom, middle-drag to pan, `h` for the full
key list in the terminal, `q` to quit.
"""

import argparse
import json
import sys
from pathlib import Path

from gsplat import paths

REPO_ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="View the trained Gaussians in colour, in Open3D.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", default="templeRing",
                   help="which output/<dataset> to open")
    p.add_argument("--min-opacity", type=float, default=0.05,
                   help="hide Gaussians fainter than this; training leaves a "
                        "haze of near-transparent ones that a point viewer "
                        "would otherwise draw at full strength")
    p.add_argument("--sfm-output", default=None,
                   help="the Structure-from-Motion output holding the camera "
                        "poses; searched automatically when omitted")
    p.add_argument("--no-cameras", action="store_true",
                   help="hide the camera frusta and their centre dots")
    p.add_argument("--no-camera-dots", action="store_true",
                   help="hide the dot drawn at each camera's optical centre")
    p.add_argument("--dot-size", type=float, default=0.0,
                   help="radius of the camera dots in scene units; 0 sizes "
                        "them from the spacing between neighbouring cameras")
    p.add_argument("--dot-color", default="220,30,30",
                   help="R,G,B of the camera dots in 0..255")
    p.add_argument("--point-size", type=float, default=3.0,
                   help="rendered size of each Gaussian")
    p.add_argument("--background", default="0.1,0.1,0.1",
                   help="R,G,B of the background in 0..1; the splats are lit "
                        "by their own colour only, so white washes them out")
    return p.parse_args(argv)


def parse_triplet(text, scale, fallback):
    """
    Read an `a,b,c` option, falling back with a warning rather than crashing.

    A malformed colour is a typo, not a reason to lose the window.
    """
    try:
        parts = tuple(float(v) / scale for v in text.split(","))
        if len(parts) != 3:
            raise ValueError
        return parts
    except ValueError:
        print(f"  (ignoring {text!r}; expected three comma-separated numbers)",
              file=sys.stderr)
        return fallback


def camera_centres(sfm_out):
    """
    Every recovered camera's optical centre, ``C = -R^T t``.

    Read from `camera_trajectory.json`, which stores the extrinsics
    themselves, so the returned coordinate is the pose as the reconstruction
    solved it rather than anything derived from a drawn shape. Falls back to
    `camera_centres.ply` if the trajectory is missing.

    :returns: (n, 3), empty when neither file is there
    """
    import numpy as np

    traj = sfm_out / paths.TRAJECTORY
    if traj.exists():
        params = json.loads(traj.read_text())["parameters"]
        centres = []
        for p in params:
            # Open3D writes the 4x4 world-to-camera matrix column-major.
            E = np.array(p["extrinsic"], float).reshape(4, 4).T
            centres.append(-E[:3, :3].T @ E[:3, 3])
        return np.array(centres, float)

    ply = sfm_out / "05_camera_poses" / "camera_centres.ply"
    if ply.exists():
        rows = []
        with open(ply) as f:
            for line in f:
                if line.strip() == "end_header":
                    break
            for line in f:
                parts = line.split()
                if len(parts) >= 3:
                    rows.append([float(v) for v in parts[:3]])
        return np.array(rows, float)

    return np.zeros((0, 3))


def camera_dot_markers(centres, o3d, radius=0.0, color=(220, 30, 30)):
    """
    A small sphere at each camera centre.

    Spheres rather than a point cloud: Open3D's point size is a single
    render-wide setting shared with the splats, so a point marker could only
    be made visible by enlarging every Gaussian too. A sphere carries its own
    size and sits at exactly the coordinate given.

    :param radius: 0 sizes the spheres from the median gap between
                   neighbouring cameras, so they stay separate at any count
    :returns: list of geometries
    """
    import numpy as np

    if len(centres) == 0:
        return []

    if radius <= 0:
        if len(centres) > 1:
            d = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=-1)
            np.fill_diagonal(d, np.inf)
            radius = float(np.median(d.min(axis=1))) * 0.12
        else:
            radius = 0.01
    rgb = np.array(color, float) / 255.0

    markers = []
    for C in centres:
        # resolution 8 keeps the triangle count low; these are dots, not
        # shapes anyone will inspect, and there can be dozens of them.
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius,
                                                         resolution=8)
        sphere.translate(np.asarray(C, float))
        sphere.paint_uniform_color(rgb)
        sphere.compute_vertex_normals()
        markers.append(sphere)
    return markers


def main(argv=None):
    args = parse_args(argv)

    try:
        import open3d as o3d
    except ImportError:
        print("Open3D is not installed. Install it with:\n"
              "    .venv\\Scripts\\python.exe -m pip install open3d",
              file=sys.stderr)
        return 1

    splat_dir = paths.splat_output(args.dataset)
    geometries = []

    # Prefer the coloured cloud written beside the splat file; fall back to
    # decoding the 3DGS PLY itself, so outputs from before that file existed
    # still open.
    rgb_path = splat_dir / "gaussians_rgb.ply"
    splat_path = splat_dir / "gaussians.ply"
    if rgb_path.exists():
        pcd = o3d.io.read_point_cloud(str(rgb_path))
        geometries.append(pcd)
        print(f"  splats  {len(pcd.points):,} (coloured)")
    elif splat_path.exists():
        import numpy as np

        from gsplat import io as gio
        xyz, rgb = gio.gaussian_point_cloud(gio.load_ply(splat_path),
                                            min_opacity=args.min_opacity)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(xyz, float))
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(rgb, float))
        geometries.append(pcd)
        print(f"  splats  {len(xyz):,} (coloured, decoded from "
              f"{splat_path.name})")
    else:
        print(f"\nNo trained scene in:\n    {splat_dir}\n\n"
              f"Train one first:\n\n"
              f"    python run_pipeline.py --dataset {args.dataset}\n",
              file=sys.stderr)
        return 1

    if not args.no_cameras:
        # The cameras belong to the SfM result, not to this repository, so a
        # missing one is a reason to carry on without them rather than to
        # fail -- the splats are what this viewer is for.
        try:
            sfm_out = paths.find_sfm_output(args.dataset,
                                            explicit=args.sfm_output)
        except SystemExit:
            sfm_out = None
            print("  (no SfM output found; showing the splats alone)",
                  file=sys.stderr)

        if sfm_out is not None:
            frusta = sfm_out / "05_camera_poses" / "cameras.ply"
            if frusta.exists():
                lines = o3d.io.read_line_set(str(frusta))
                geometries.append(lines)
                # Eight edges per camera.
                print(f"  cameras {len(lines.lines) // 8}")

            if not args.no_camera_dots:
                centres = camera_centres(sfm_out)
                if len(centres):
                    color = parse_triplet(args.dot_color, 1.0, (220.0, 30.0, 30.0))
                    dots = camera_dot_markers(centres, o3d,
                                              radius=args.dot_size,
                                              color=color)
                    geometries.extend(dots)
                    print(f"  dots    {len(dots)} at the camera centres")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"{args.dataset} -- gaussian splats")
    for g in geometries:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.point_size = args.point_size
    opt.background_color = list(
        parse_triplet(args.background, 1.0, (0.1, 0.1, 0.1)))
    vis.run()
    vis.destroy_window()
    return 0


if __name__ == "__main__":
    sys.exit(main())
