#!/usr/bin/env python
"""
Train 3D Gaussians on the Structure-from-Motion reconstruction.

Everything the method does is implemented in `gsplat/`; this file only runs
the loop. The Structure-from-Motion stage must have run first, since the
splats are initialised from its camera poses and sparse point cloud --
`python run_pipeline.py` does both in order.

Usage
-----
    python train_splat.py                      # defaults, CPU-sized
    python train_splat.py --iterations 3000
    python train_splat.py --width 200 --sh-degree 1
    python train_splat.py --n-views 12 --iterations 500    # a quick smoke run

On a CPU this is slow -- the reference implementation is a CUDA kernel and
this is a loop over image tiles in Python. Train at a reduced resolution and
expect minutes per few hundred iterations rather than the paper's 30,000.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from gsplat import (data, densify, io, loss as losses, model as gmodel,
                    paths as gpaths, render, report)

REPO_ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train 3D Gaussian splatting on the SfM reconstruction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", default="templeRing")
    p.add_argument("--sfm-output", default=None,
                   help="the Structure-from-Motion output folder to train "
                        "from; searched automatically when omitted")
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--width", type=int, default=160,
                   help="training resolution; the cost grows with its square")
    p.add_argument("--n-views", type=int, default=None,
                   help="use only the first N views")
    p.add_argument("--sh-degree", type=int, default=0, choices=[0, 1, 2],
                   help="0 is constant colour; higher adds view-dependence")
    p.add_argument("--densify-from", type=int, default=300)
    p.add_argument("--densify-until", type=int, default=1500)
    p.add_argument("--densify-every", type=int, default=100)
    p.add_argument("--densify-percentile", type=float, default=92.0,
                   help="densify the Gaussians above this percentile of "
                        "screen-space gradient; use --densify-threshold for "
                        "the paper's fixed value instead")
    p.add_argument("--densify-threshold", type=float, default=None,
                   help="absolute gradient threshold (paper uses 0.0002)")
    p.add_argument("--opacity-reset-every", type=int, default=900,
                   help="0 disables the periodic opacity reset")
    p.add_argument("--tile", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--preview-every", type=int, default=250)
    p.add_argument("--rgb-min-opacity", type=float, default=0.05,
                   help="when writing the coloured point cloud, drop "
                        "Gaussians fainter than this; 0 keeps every one")
    p.add_argument("--out", default=None, help="defaults to output/<dataset>/07_splatting")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))

    out_dir = Path(args.out) if args.out else gpaths.splat_output(args.dataset)
    (out_dir / "previews").mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("3D GAUSSIAN SPLATTING")
    print("=" * 72)

    views, points, colors = data.load_scene(
        args.dataset, max_width=args.width, n_views=args.n_views,
        sfm_output=args.sfm_output)
    w, h = views[0].size
    centres = np.array([v.centre for v in views])
    extent = float(np.linalg.norm(centres - centres.mean(0), axis=1).max())
    print(f"  {len(views)} views at {w}x{h}")
    print(f"  {len(points):,} seed points from the SfM cloud")
    print(f"  scene extent {extent:.3f}")

    model = gmodel.GaussianModel(points, colors, sh_degree=args.sh_degree)
    optimizer = densify.build_optimizer(model, scene_extent=extent)
    controller = densify.DensityController(
        grad_threshold=args.densify_threshold,
        grad_percentile=args.densify_percentile,
        scene_extent=extent)
    controller.reset(model.n)
    print(f"  {model.n:,} Gaussians, sh degree {args.sh_degree}")
    print(f"  writing to {out_dir}")
    print()

    history = []
    started = time.time()
    for it in range(1, args.iterations + 1):
        view = views[random.randrange(len(views))]
        target = torch.from_numpy(view.image)

        img, aux = render.render(model, view.R, view.t, view.K, w, h,
                                 tile=args.tile, return_aux=True)
        total, l1, ssim_v = losses.photometric_loss(img, target)

        optimizer.zero_grad(set_to_none=True)
        total.backward()

        if aux is not None and args.densify_from <= it <= args.densify_until:
            controller.record(aux["uv"].grad, aux["visible"])

        optimizer.step()

        if it % args.log_every == 0 or it == 1:
            p = losses.psnr(img.detach(), target)
            el = time.time() - started
            print(f"  [{it:5d}/{args.iterations}] loss {total.item():.4f}  "
                  f"L1 {l1.item():.4f}  SSIM {ssim_v.item():.4f}  "
                  f"PSNR {p:.2f} dB  |  {model.n:,} gaussians  "
                  f"{el / it:.2f}s/it")
            history.append({"iteration": it, "loss": total.item(),
                            "l1": l1.item(), "ssim": ssim_v.item(), "psnr": p,
                            "n_gaussians": model.n})

        if (args.densify_from <= it <= args.densify_until
                and it % args.densify_every == 0):
            stats = controller.step(model, optimizer)
            optimizer = densify.build_optimizer(model, scene_extent=extent)
            print(f"      densify: +{stats['cloned']} cloned, "
                  f"+{stats['split']} split, -{stats['pruned']} pruned  "
                  f"({stats['before']:,} -> {stats['after']:,})")

        if args.opacity_reset_every and it % args.opacity_reset_every == 0:
            controller.reset_opacities(model)
            print("      opacity reset")

        if args.preview_every and it % args.preview_every == 0:
            with torch.no_grad():
                prev = render.render(model, views[0].R, views[0].t, views[0].K,
                                     w, h, tile=args.tile)
            io.save_image(prev, out_dir / "previews" / f"iter_{it:05d}.png")

    # -- final artefacts ---------------------------------------------------
    with torch.no_grad():
        final = render.render(model, views[0].R, views[0].t, views[0].K, w, h,
                              tile=args.tile)
    io.save_image(final, out_dir / "render_view000.png")
    io.save_image(views[0].image, out_dir / "target_view000.png")

    ply, n = io.save_ply(model, out_dir / "gaussians.ply")
    # The 3DGS layout hides the colour in the SH coefficients, so nothing but a
    # splat renderer shows the scene in colour. Write it again as a plain
    # coloured cloud that any viewer understands.
    rgb_ply, n_rgb = io.save_colored_ply(
        model, out_dir / "gaussians_rgb.ply",
        min_opacity=args.rgb_min_opacity)
    (out_dir / "training_log.json").write_text(json.dumps({
        "dataset": args.dataset, "iterations": args.iterations,
        "resolution": [w, h], "views": len(views),
        "sh_degree": args.sh_degree, "final_gaussians": model.n,
        "runtime_seconds": round(time.time() - started, 1),
        "history": history,
    }, indent=2))

    _, mean_diff = report.training_summary(
        views[0].image, final.detach().cpu().numpy(), history,
        out_dir / "training_summary.png",
        title=(f"3D Gaussian splatting -- {model.n:,} Gaussians, "
               f"{args.iterations} iterations at {w}x{h}, {len(views)} views"))
    print(f"  wrote training_summary.png (mean abs difference "
          f"{mean_diff:.2f}/255)")

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)
    print(f"  {n:,} Gaussians")
    print(f"  runtime {time.time() - started:.1f}s")
    print(f"  splat file:  {ply}")
    print(f"  drag it onto https://superspl.at/editor to view")
    print(f"  colour file: {rgb_ply}  ({n_rgb:,} points)")
    print(f"  or: python view_splat.py --dataset {args.dataset}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
