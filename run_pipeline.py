#!/usr/bin/env python
"""
The whole thing, end to end: 2D photographs in, Gaussian splats out.

This is the only script that spans both halves of the project. It runs
Structure from Motion over a folder of images to recover the camera poses and
a sparse 3D point cloud, then trains 3D Gaussians on that result. Neither
half contains any method logic of its own -- `sfm/main.py` owns stage 1 and
`train_splat.py` owns stage 2; this file only sequences them and checks that
the handoff between them actually happened.

    images  ->  [ sfm/ submodule ]  ->  camera poses + 3D points
                                              |
                                              v
                              [ train_splat.py / gsplat/ ]  ->  gaussians.ply

The ordering is not a convenience. Splatting does not discover geometry: it
refines it. The Gaussians are seeded at the SfM points and optimised against
photographs whose poses SfM recovered, so stage 2 cannot run until stage 1
has produced both.

Stage 1 lives in its own repository, included here as the `sfm/` submodule.
If it is empty, fetch it:

    git submodule update --init --recursive

Usage
-----
    python run_pipeline.py                    # templeRing, full quality
    python run_pipeline.py --quick            # a few minutes, to prove it runs
    python run_pipeline.py --dataset templeRing_rot90ccw
    python run_pipeline.py --iterations 3000 --width 200
    python run_pipeline.py --skip-sfm         # reuse an existing reconstruction
    python run_pipeline.py --sfm-extra "--no-mesh --lowes-ratio 0.75"

Run `python sfm/main.py --help` or `python train_splat.py --help` for the
options each stage accepts; `--sfm-extra` and `--splat-extra` forward anything
not exposed here.
"""

import argparse
import os
import shlex
import sys
import time
from pathlib import Path

from gsplat import paths

REPO_ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run the full pipeline: 2D images -> SfM -> 3D Gaussian "
                    "splatting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", default="templeRing",
                   help="registered dataset name (see sfm/src/config.py)")

    p.add_argument("--quick", action="store_true",
                   help="a small, fast run that exercises every stage; the "
                        "result is not representative, it only proves the "
                        "pipeline works end to end")

    stage1 = p.add_argument_group("stage 1 -- structure from motion")
    stage1.add_argument("--n-images", type=int, default=None,
                        help="number of views to reconstruct from; omit for "
                             "the dataset default")
    stage1.add_argument("--no-mesh", action="store_true",
                        help="skip Poisson surface reconstruction")
    stage1.add_argument("--sfm-extra", default="",
                        help="further arguments forwarded to sfm/main.py, "
                             "quoted")

    stage2 = p.add_argument_group("stage 2 -- gaussian splatting")
    stage2.add_argument("--iterations", type=int, default=None,
                        help="training iterations; omit for train_splat.py's "
                             "own default")
    stage2.add_argument("--width", type=int, default=None,
                        help="training resolution; cost grows with its square")
    stage2.add_argument("--sh-degree", type=int, default=None, choices=[0, 1, 2],
                        help="0 is constant colour; higher adds view-dependence")
    stage2.add_argument("--splat-extra", default="",
                        help="further arguments forwarded to train_splat.py, "
                             "quoted")

    skip = p.add_argument_group("running only part of it")
    skip.add_argument("--skip-sfm", action="store_true",
                      help="reuse an existing reconstruction and go straight "
                           "to splatting")
    skip.add_argument("--skip-splat", action="store_true",
                      help="stop after the reconstruction")

    return p.parse_args(argv)


def banner(text):
    print()
    print("#" * 72)
    print(f"# {text}")
    print("#" * 72)
    print()


def check_dataset(dataset):
    """
    Fail early and usefully if the images are not there.

    The alternative is a stack trace several minutes in, or -- worse -- a
    reconstruction from zero images that looks like a subtler bug.

    The dataset registry belongs to the SfM project, so it is read from the
    submodule rather than duplicated here.

    :returns: (number of images found, the folder they are in)
    """
    paths.add_sfm_to_path()
    from src.config import get_dataset

    spec = get_dataset(dataset)
    if not spec.path.is_dir():
        raise SystemExit(
            f"\nThe dataset folder does not exist:\n    {spec.path}\n\n"
            f"{spec.source}\n"
            f"Download it and extract it there, then run this again.")

    n = len(sorted(spec.path.glob(spec.pattern)))
    if n == 0:
        raise SystemExit(
            f"\nNo images matching '{spec.pattern}' in:\n    {spec.path}\n\n"
            f"The folder exists but is empty or the files are named "
            f"differently.")
    return n, spec.path


def sfm_args(args):
    """The argv for stage 1, assembled from this script's own options."""
    argv = ["--dataset", args.dataset]
    if args.quick:
        # Few enough views to finish in a minute, and no match overlays --
        # those are a per-pair figure, so they dominate a short run.
        argv += ["--n-images", "12", "--no-match-images", "--no-mesh"]
    if args.n_images is not None:
        argv += ["--n-images", str(args.n_images)]
    if args.no_mesh and "--no-mesh" not in argv:
        argv += ["--no-mesh"]
    return argv + shlex.split(args.sfm_extra)


def splat_args(args, sfm_out):
    """The argv for stage 2, assembled from this script's own options."""
    argv = ["--dataset", args.dataset, "--sfm-output", str(sfm_out)]
    if args.quick:
        argv += ["--n-views", "12", "--iterations", "300", "--width", "100"]
    if args.iterations is not None:
        argv += ["--iterations", str(args.iterations)]
    if args.width is not None:
        argv += ["--width", str(args.width)]
    if args.sh_degree is not None:
        argv += ["--sh-degree", str(args.sh_degree)]
    return argv + shlex.split(args.splat_extra)


def run_sfm(args):
    """
    Run stage 1 inside the submodule.

    `sfm/main.py` is run with the submodule as the working directory, and the
    original restored afterwards, so its outputs and its run log land in the
    SfM project's own tree rather than scattered across this one.

    :returns: the folder stage 1 wrote to
    """
    sfm_root = paths.add_sfm_to_path()
    import main as sfm_main

    cwd = os.getcwd()
    os.chdir(sfm_root)
    try:
        rc = sfm_main.main(sfm_args(args))
    finally:
        os.chdir(cwd)
    if rc != 0:
        raise SystemExit(f"Structure from Motion failed (exit {rc}).")
    return sfm_root / "output" / args.dataset


def main(argv=None):
    args = parse_args(argv)
    if args.skip_sfm and args.skip_splat:
        raise SystemExit("--skip-sfm and --skip-splat together leave nothing "
                         "to run.")

    started = time.time()
    n_images, image_dir = check_dataset(args.dataset)

    banner(f"FULL PIPELINE -- {args.dataset}"
           + ("  (quick run)" if args.quick else ""))
    print(f"  input      {n_images} images in {image_dir}")
    print(f"  stage 1    structure from motion  "
          f"{'(skipped)' if args.skip_sfm else '-> 3D points + camera poses'}")
    print(f"  stage 2    gaussian splatting     "
          f"{'(skipped)' if args.skip_splat else '-> gaussians.ply'}")

    if args.skip_sfm:
        # Not run here, so its location is whatever the usual search finds.
        sfm_out = paths.find_sfm_output(args.dataset)
        print(f"  reusing    {sfm_out}")
    else:
        banner("STAGE 1 of 2 -- STRUCTURE FROM MOTION  (2D images -> 3D points)")
        sfm_out = run_sfm(args)
        # Checked rather than assumed: a stage that exits 0 without writing
        # the handoff would otherwise fail later and look like a stage 2 bug.
        sfm_out = paths.find_sfm_output(args.dataset, explicit=sfm_out)

    if args.skip_splat:
        banner("PIPELINE DONE  (stage 1 only)")
        print(f"  3D model   {sfm_out / paths.POINT_CLOUD}")
        print(f"  runtime    {time.time() - started:.1f}s")
        return 0

    banner("STAGE 2 of 2 -- GAUSSIAN SPLATTING  (3D points -> gaussians)")
    import train_splat

    rc = train_splat.main(splat_args(args, sfm_out))
    if rc != 0:
        raise SystemExit(f"Gaussian splatting failed (exit {rc}).")

    splat_dir = paths.splat_output(args.dataset)
    banner("PIPELINE DONE")
    print(f"  stage 1   {sfm_out / paths.POINT_CLOUD}")
    print(f"  stage 2   {splat_dir / 'gaussians.ply'}")
    print(f"            {splat_dir / 'gaussians_rgb.ply'}  (coloured points)")
    print(f"  runtime   {(time.time() - started) / 60:.1f} min total")
    print()
    print("  view the splats:")
    print(f"      python view_splat.py --dataset {args.dataset}")
    print("      or drag gaussians.ply onto https://superspl.at/editor")
    return 0


if __name__ == "__main__":
    sys.exit(main())
