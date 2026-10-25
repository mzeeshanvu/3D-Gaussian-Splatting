"""
The figure that shows whether training worked.

Three images across the top -- the input photograph, the same view rendered
from the Gaussians, and the absolute difference between them -- and the
training curves underneath. The difference panel is the honest one: it shows
*where* the model is still wrong, which a single PSNR number hides.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def training_summary(target, rendered, history, path, title=""):
    """
    Write the comparison-and-curves figure.

    :param target: (H, W, 3) input photograph in 0..1
    :param rendered: (H, W, 3) render of the same view in 0..1
    :param history: list of per-log-step dicts from the training loop
    :returns: (path, mean absolute difference in 0..255)
    """
    target = np.clip(np.asarray(target), 0, 1)
    rendered = np.clip(np.asarray(rendered), 0, 1)
    diff = np.abs(target - rendered).mean(axis=2) * 255.0

    fig = plt.figure(figsize=(16, 8))
    panels = [(target, "Input photograph", {}),
              (rendered, "Rendered from Gaussians", {}),
              (diff, f"Absolute difference (mean {diff.mean():.1f}/255)",
               {"cmap": "inferno", "vmin": 0, "vmax": 60})]
    for k, (img, name, kw) in enumerate(panels):
        ax = fig.add_subplot(2, 3, k + 1)
        im = ax.imshow(img, **kw)
        ax.set_title(name, fontsize=12)
        ax.axis("off")
        if kw:
            fig.colorbar(im, ax=ax, fraction=0.046)

    it = [r["iteration"] for r in history]
    curves = [
        (3, [(("psnr", "seagreen", None))], "PSNR (dB)", "Reconstruction quality"),
        (4, [("loss", "darkorange", "total"), ("l1", "steelblue", "L1")],
         "loss", "Training loss"),
        (5, [("n_gaussians", "purple", None)], "Gaussians",
         "Adaptive density control"),
    ]
    for slot, series, ylabel, name in curves:
        ax = fig.add_subplot(2, 3, slot + 1)
        for key, color, label in series:
            ax.plot(it, [r[key] for r in history], color=color, label=label)
        ax.set_xlabel("iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(name)
        ax.grid(alpha=0.3)
        if any(lbl for _, _, lbl in series):
            ax.legend()

    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=(0, 0, 1, 0.96) if title else None)
    plt.savefig(path, dpi=170)
    plt.close()
    return path, float(diff.mean())
