#!/usr/bin/env python
"""
Quantify what Gaussian splatting added over the sparse reconstruction.

Both representations are rendered from the *same* recovered camera poses at the
same resolution and scored against the same photographs, so the comparison is
like for like: the SfM point cloud drawn as z-buffered points, the Gaussians
through the project's own rasteriser.

Run after both stages have completed:

    python run_pipeline.py
    python tools/compare_baseline.py

Writes
------
output/<dataset>/07_splatting/baseline_comparison.json   per-view scores
docs/images/before_after.png                             the visual comparison
docs/images/improvement.png                              the figure for the README
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gsplat import data, io, loss as losses, paths, render  # noqa: E402

DATASET = "templeRing"
WIDTH = 160

# Design tokens for the figure -- a validated two-hue categorical pair.
SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
MUTED, GRID, AXIS = "#898781", "#e1e0d9", "#c3c2b7"
SPLAT, POINTS = "#2a78d6", "#eb6834"


def render_point_cloud(pts, cols, R, t, K, w, h, radius=0):
    """
    Draw the point cloud the way a point-cloud viewer would: z-buffered dots.

    This is the honest baseline for "what the sparse reconstruction looks
    like", and `radius` lets it be made deliberately generous.
    """
    cam = pts @ R.T + t
    z = cam[:, 2]
    keep = z > 1e-6
    cam, cols_k, z = cam[keep], cols[keep], z[keep]

    u = (K[0, 0] * cam[:, 0] / z + K[0, 2]).round().astype(int)
    v = (K[1, 1] * cam[:, 1] / z + K[1, 2]).round().astype(int)

    img = np.zeros((h, w, 3), np.float32)
    depth = np.full((h, w), np.inf, np.float32)
    for i in np.argsort(-z):                       # far to near
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                y, x = v[i] + dy, u[i] + dx
                if 0 <= x < w and 0 <= y < h and z[i] < depth[y, x]:
                    depth[y, x] = z[i]
                    img[y, x] = cols_k[i]
    return img


def score(img, target):
    """PSNR, SSIM and the fraction of the frame the render actually paints."""
    img = img if isinstance(img, torch.Tensor) else torch.from_numpy(img)
    img = img.float()
    return {
        "psnr": losses.psnr(img, target),
        "ssim": losses.ssim(img, target).item(),
        "cover": float((img.sum(-1) > 0.01).float().mean()) * 100,
    }


def evaluate(views, points, colors, model, w, h):
    """Score both representations on every view."""
    rows = []
    for i, v in enumerate(views):
        target = torch.from_numpy(v.image)
        pc = render_point_cloud(points, colors, v.R, v.t, v.K, w, h)
        with torch.no_grad():
            gs = render.render(model, v.R, v.t, v.K, w, h, tile=16)
        p, s = score(pc, target), score(gs, target)
        rows.append({"view": i,
                     "psnr_points": p["psnr"], "psnr_splat": s["psnr"],
                     "ssim_points": p["ssim"], "ssim_splat": s["ssim"],
                     "cover_points": p["cover"], "cover_splat": s["cover"]})
        if i % 10 == 0:
            print("  view %2d   points %5.2f dB   splat %5.2f dB"
                  % (i, p["psnr"], s["psnr"]))
    return rows


def write_strip(views, points, colors, model, w, h, out_path):
    """The four-panel visual comparison, at 3x for legibility."""
    v = views[0]
    with torch.no_grad():
        gs = render.render(model, v.R, v.t, v.K, w, h, tile=16)
    panels = [
        (render_point_cloud(points, colors, v.R, v.t, v.K, w, h, 0),
         "SfM point cloud"),
        (render_point_cloud(points, colors, v.R, v.t, v.K, w, h, 1),
         "points, dilated"),
        (gs.detach().cpu().numpy(), "Gaussian splatting"),
        (v.image, "photograph"),
    ]
    tiles = []
    for img, label in panels:
        a = (np.clip(np.asarray(img), 0, 1) * 255).astype(np.uint8)
        a = cv2.resize(a, (w * 3, h * 3), interpolation=cv2.INTER_NEAREST)
        a = cv2.copyMakeBorder(a, 28, 4, 4, 4, cv2.BORDER_CONSTANT,
                               value=(255, 255, 255))
        cv2.putText(a, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_path), np.hstack(tiles))


def write_chart(history, rows, summary, out_path):
    """
    Two panels, one measure each -- never two scales on one axis.

    Left: the training trajectory against the baseline, so the gap is seen to
    be earned rather than asserted. Right: every view scored separately, so the
    result cannot be an artefact of one favourable frame.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK_2,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": AXIS,
    })

    base, splat = summary["psnr_points"]["mean"], summary["psnr_splat"]["mean"]
    gain = splat - base

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=200)
    fig.subplots_adjust(left=0.065, right=0.975, top=0.76, bottom=0.145,
                        wspace=0.26)

    def tidy(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(AXIS)
        ax.spines["bottom"].set_color(AXIS)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, labelsize=9)
        ax.set_ylim(8, 40)
        ax.set_ylabel("PSNR (dB)", fontsize=9.5)

    it = [r["iteration"] for r in history]
    ps = [r["psnr"] for r in history]
    ax1.axhline(base, color=POINTS, linewidth=2, linestyle=(0, (5, 3)), zorder=3)
    ax1.plot(it, ps, color=SPLAT, linewidth=2, zorder=4)
    ax1.annotate("", xy=(1640, splat), xytext=(1640, base),
                 arrowprops=dict(arrowstyle="<->", color=INK_2, linewidth=1.3,
                                 shrinkA=0, shrinkB=0), zorder=5)
    ax1.text(1600, (splat + base) / 2, "+%.1f dB" % gain, ha="right",
             va="center", fontsize=13, fontweight="bold", color=INK)
    ax1.text(60, base - 1.5, "sparse point cloud", fontsize=9.5, color=POINTS,
             fontweight="600", va="top")
    ax1.text(130, 36.6, "Gaussian splatting", fontsize=9.5, color=SPLAT,
             fontweight="600")
    tidy(ax1)
    ax1.set_xlabel("training iteration", fontsize=9.5)
    ax1.set_xlim(0, max(it) * 1.03)
    ax1.set_title("Reconstruction quality during training", fontsize=11,
                  fontweight="600", color=INK, loc="left", pad=10)

    v = [r["view"] for r in rows]
    pp = [r["psnr_points"] for r in rows]
    sp = [r["psnr_splat"] for r in rows]
    ax2.fill_between(v, pp, sp, color=SPLAT, alpha=0.10, linewidth=0, zorder=2)
    ax2.plot(v, pp, color=POINTS, linewidth=2, marker="o", markersize=3.2, zorder=4)
    ax2.plot(v, sp, color=SPLAT, linewidth=2, marker="o", markersize=3.2, zorder=4)
    ax2.text(len(v) - 0.5, 36.6, "Gaussian splatting", fontsize=9.5,
             color=SPLAT, fontweight="600", ha="right")
    ax2.text(len(v) / 2, 9.1, "sparse point cloud", fontsize=9.5, color=POINTS,
             fontweight="600", ha="center")
    tidy(ax2)
    ax2.set_xlabel("view", fontsize=9.5)
    ax2.set_xlim(-1, len(v))
    ax2.set_title("Every one of the %d views, scored separately" % len(v),
                  fontsize=11, fontweight="600", color=INK, loc="left", pad=10)

    fig.text(0.065, 0.945,
             "Gaussian splatting improves reconstruction quality by %.1f dB" % gain,
             fontsize=15, fontweight="bold", color=INK)
    fig.text(0.065, 0.882,
             "Mean PSNR against the input photographs rises from %.1f dB to "
             "%.1f dB; splatting wins on %d of %d views."
             % (base, splat, sum(a < b for a, b in zip(pp, sp)), len(v)),
             fontsize=10, color=INK_2)

    handles = [
        plt.Line2D([], [], color=SPLAT, linewidth=2, label="Gaussian splatting"),
        plt.Line2D([], [], color=POINTS, linewidth=2, linestyle=(0, (5, 3)),
                   label="Sparse point cloud (SfM output)"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.975, 0.965),
               frameon=False, fontsize=9.5, labelcolor=INK_2, handlelength=2.2)
    fig.text(0.065, 0.028,
             "%s, %d views at %dx%d. Point cloud rendered as z-buffered points "
             "from the same poses. PSNR against the input photograph."
             % (DATASET, len(v), WIDTH, int(WIDTH * 0.75)),
             fontsize=8, color=MUTED)

    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def main():
    splat_dir = paths.splat_output(DATASET)
    ply = splat_dir / "gaussians.ply"
    if not ply.exists():
        raise SystemExit("No trained model at %s -- run train_splat.py first." % ply)

    views, points, colors = data.load_scene(DATASET, max_width=WIDTH)
    w, h = views[0].size
    model = io.load_ply(ply)
    print("%d views at %dx%d | %s points | %s gaussians\n"
          % (len(views), w, h, format(len(points), ","), format(model.n, ",")))

    rows = evaluate(views, points, colors, model, w, h)

    def stat(key):
        a = np.array([r[key] for r in rows])
        return {"mean": a.mean(), "std": a.std(), "min": a.min(), "max": a.max()}

    summary = {k: stat(k) for k in
               ("psnr_points", "psnr_splat", "ssim_points", "ssim_splat",
                "cover_points", "cover_splat")}

    print("\n" + "=" * 74)
    print("ALL %d VIEWS" % len(rows))
    print("=" * 74)
    for label, kp, ks in [("PSNR (dB)", "psnr_points", "psnr_splat"),
                          ("SSIM", "ssim_points", "ssim_splat"),
                          ("Coverage (%)", "cover_points", "cover_splat")]:
        p, s = summary[kp], summary[ks]
        print("  %-13s points %7.3f +/- %.3f   splat %7.3f +/- %.3f   %+.3f"
              % (label, p["mean"], p["std"], s["mean"], s["std"],
                 s["mean"] - p["mean"]))
    wins = sum(1 for r in rows if r["psnr_splat"] > r["psnr_points"])
    print("\n  splat beats points on %d / %d views" % (wins, len(rows)))

    (splat_dir / "baseline_comparison.json").write_text(
        json.dumps({"views": rows, "summary": summary}, indent=2))

    images = ROOT / "docs" / "images"
    images.mkdir(parents=True, exist_ok=True)
    write_strip(views, points, colors, model, w, h, images / "before_after.png")
    history = json.loads((splat_dir / "training_log.json").read_text())["history"]
    write_chart(history, rows, summary, images / "improvement.png")
    print("\nwrote baseline_comparison.json, before_after.png, improvement.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
