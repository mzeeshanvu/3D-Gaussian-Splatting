"""
Where the Structure-from-Motion result lives, and where ours goes.

Splatting is seeded from another project's output, and since that project is
a separate repository -- included here as the `sfm/` submodule -- the two no
longer share one `output/` tree by construction. This module is the single
place that knows how to find the handoff, so nothing else has to guess.

The search order, first hit wins:

1. an explicit path, passed on the command line
2. ``$SFM_OUTPUT``, for a reconstruction kept outside either repository
3. ``<this repo>/output/<dataset>``, if an SfM result has been copied in
4. ``<this repo>/sfm/output/<dataset>``, where the submodule writes by default

Our own results always go to ``<this repo>/output/<dataset>/07_splatting``.
Reading from the submodule but never writing to it keeps `git status` clean
in there, so the submodule only ever moves when its code actually changes.
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The SfM project, vendored as a git submodule.
SFM_ROOT = REPO_ROOT / "sfm"

#: What stage 1 must have left behind before stage 2 can start: the poses the
#: training views are rendered from, and the cloud the Gaussians are seeded at.
TRAJECTORY = Path("05_camera_poses") / "camera_trajectory.json"
POINT_CLOUD = Path("06_model") / "reconstruction.ply"


def add_sfm_to_path():
    """
    Make the submodule's ``src`` package importable.

    The dataset registry -- image locations, filename patterns, intrinsics --
    belongs to the SfM project and is read from there rather than copied, so
    that the two halves cannot disagree about what `templeRing` means.

    :raises SystemExit: if the submodule has not been checked out
    """
    if not (SFM_ROOT / "src" / "config.py").exists():
        raise SystemExit(
            f"\nThe sfm/ submodule is empty:\n    {SFM_ROOT}\n\n"
            f"It holds the Structure-from-Motion stage this one is built on.\n"
            f"Fetch it with:\n\n    git submodule update --init --recursive\n")
    if str(SFM_ROOT) not in sys.path:
        sys.path.insert(0, str(SFM_ROOT))
    return SFM_ROOT


def sfm_candidates(dataset):
    """Every place an SfM result for `dataset` might be, in search order."""
    out = []
    env = os.environ.get("SFM_OUTPUT")
    if env:
        env = Path(env)
        # Accept either the output root or the dataset folder inside it.
        out.append(env / dataset if (env / dataset).is_dir() else env)
    out.append(REPO_ROOT / "output" / dataset)
    out.append(SFM_ROOT / "output" / dataset)
    return out


def find_sfm_output(dataset, explicit=None):
    """
    Locate a completed reconstruction for `dataset`.

    A folder only counts if it holds *both* handoff files. A half-finished
    run is skipped rather than returned, so a crashed reconstruction in one
    location does not mask a good one in another.

    :param explicit: a path given on the command line, which overrides the
                     search but is still checked for completeness
    :returns: the dataset's output folder
    :raises SystemExit: with every path tried, when none qualifies
    """
    if explicit is not None:
        explicit = Path(explicit)
        candidates = [explicit / dataset
                      if (explicit / dataset).is_dir() else explicit]
    else:
        candidates = sfm_candidates(dataset)

    for path in candidates:
        if all((path / rel).exists() for rel in (TRAJECTORY, POINT_CLOUD)):
            return path

    tried = "\n".join(f"    {p}" for p in candidates)
    raise SystemExit(
        f"\nNo completed Structure-from-Motion result for '{dataset}'.\n"
        f"Looked in:\n{tried}\n\n"
        f"Each of those must contain both:\n"
        f"    {TRAJECTORY}\n    {POINT_CLOUD}\n\n"
        f"Run the first stage, which writes them:\n\n"
        f"    python run_pipeline.py --dataset {dataset}\n\n"
        f"or point at an existing reconstruction with --sfm-output.")


def splat_output(dataset):
    """Where this repository writes its own results."""
    return REPO_ROOT / "output" / dataset / "07_splatting"
