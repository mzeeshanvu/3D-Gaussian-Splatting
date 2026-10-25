"""
Adaptive density control: changing how many Gaussians there are.

Optimising the parameters of a fixed set of Gaussians cannot fix a set that is
simply wrong -- too few to cover a surface, or one blob spanning detail it
cannot represent. So the population itself is edited during training.

The signal is the gradient of the loss with respect to each Gaussian's screen
position, accumulated over recent views. A large value means the renderer
keeps wanting to move that Gaussian, which is what happens when one primitive
is being asked to explain more than it can. Those get:

* **cloned**, if the Gaussian is small -- an under-reconstructed region needs
  more primitives, so a duplicate is added to be pulled into place;
* **split**, if the Gaussian is large -- an over-sized blob is replaced by two
  smaller ones sampled inside it.

Separately, Gaussians that have gone essentially transparent are pruned, since
they cost time and contribute nothing.
"""

import torch

# Dividing a split Gaussian's scale by this is the paper's choice; it keeps the
# two children covering about the span of the parent without overlapping.
SPLIT_SCALE_DECAY = 1.6


class DensityController:
    """
    Accumulates the screen-space gradient statistics and applies the edits.

    :param grad_threshold: average positional gradient above which a Gaussian
                           is considered under-reconstructed
    :param min_opacity: prune anything fainter than this
    :param percent_dense: a Gaussian larger than this fraction of the scene
                          extent is split rather than cloned
    """

    def __init__(self, grad_threshold=None, grad_percentile=92.0,
                 min_opacity=0.005, percent_dense=0.01, scene_extent=1.0):
        # The paper's fixed 0.0002 is calibrated to its own resolution and loss
        # normalisation; at a different image size the gradients simply live on
        # a different scale and a fixed cut either fires on everything or on
        # nothing. Selecting a top percentile instead adapts automatically, and
        # the absolute threshold stays available for faithfulness.
        self.grad_threshold = grad_threshold
        self.grad_percentile = grad_percentile
        self.min_opacity = min_opacity
        self.percent_dense = percent_dense
        self.scene_extent = scene_extent
        self.reset(0)

    def reset(self, n):
        self.grad_accum = torch.zeros(n)
        self.counts = torch.zeros(n)

    def resize(self, n):
        """Grow the statistics after the population has changed."""
        self.reset(n)

    def record(self, uv_grad, visible):
        """
        Add one view's screen-space gradients.

        :param uv_grad: (n_visible, 2) d(loss)/d(screen position)
        :param visible: (n,) bool mask that produced those gradients
        """
        if uv_grad is None:
            return
        idx = visible.nonzero(as_tuple=True)[0]
        if len(self.grad_accum) != len(visible):
            self.resize(len(visible))
        self.grad_accum[idx] += uv_grad.norm(dim=1).detach().cpu()
        self.counts[idx] += 1

    def _average_gradient(self):
        return self.grad_accum / self.counts.clamp_min(1.0)

    @torch.no_grad()
    def step(self, model, optimizer, max_gaussians=400_000):
        """
        Clone, split and prune, then rebuild the optimiser state.

        The optimiser is rebuilt rather than patched because Adam keeps a
        moment estimate per parameter element; changing the number of
        Gaussians invalidates those entries, and stale moments produce wild
        updates on the new primitives.

        :returns: dict describing what changed
        """
        grads = self._average_gradient()
        if self.grad_threshold is not None:
            cut = self.grad_threshold
        else:
            seen = grads[self.counts > 0]
            cut = (float(torch.quantile(seen, self.grad_percentile / 100.0))
                   if seen.numel() else float("inf"))
        scales = model.scales.max(dim=1).values.cpu()
        big = scales > self.percent_dense * self.scene_extent
        wants_more = (grads > cut) & (self.counts > 0)

        clone = wants_more & ~big
        split = wants_more & big
        if model.n + int(clone.sum()) + int(split.sum()) > max_gaussians:
            clone[:] = False
            split[:] = False

        new_means, new_logs, new_quats, new_sh, new_op = [], [], [], [], []

        if clone.any():
            # A copy in place; the optimiser separates the pair over the next
            # few steps because they receive different gradients.
            new_means.append(model.means[clone])
            new_logs.append(model.log_scales[clone])
            new_quats.append(model.quats[clone])
            new_sh.append(model.sh[clone])
            new_op.append(model.opacities_raw[clone])

        if split.any():
            k = int(split.sum())
            stds = model.scales[split]
            # Sample the children inside the parent's own ellipsoid.
            offs = torch.randn(k, 3, device=stds.device) * stds
            from .model import quaternion_to_rotation
            rot = quaternion_to_rotation(model.quats[split])
            world_offs = torch.bmm(rot, offs.unsqueeze(-1)).squeeze(-1)
            for sign in (1.0, -1.0):
                new_means.append(model.means[split] + sign * world_offs)
                new_logs.append(model.log_scales[split] - torch.log(
                    torch.tensor(SPLIT_SCALE_DECAY, device=stds.device)))
                new_quats.append(model.quats[split])
                new_sh.append(model.sh[split])
                new_op.append(model.opacities_raw[split])

        keep = ~split.to(model.means.device)
        faint = (model.opacities.squeeze(1) < self.min_opacity).cpu()
        keep = keep & ~faint.to(keep.device)

        means = [model.means[keep]] + new_means
        logs = [model.log_scales[keep]] + new_logs
        quats = [model.quats[keep]] + new_quats
        sh = [model.sh[keep]] + new_sh
        op = [model.opacities_raw[keep]] + new_op

        stats = {"cloned": int(clone.sum()), "split": int(split.sum()),
                 "pruned": int(faint.sum()), "before": model.n}
        _replace_parameters(model, torch.cat(means), torch.cat(logs),
                            torch.cat(quats), torch.cat(sh), torch.cat(op))
        stats["after"] = model.n
        self.reset(model.n)
        return stats

    @torch.no_grad()
    def reset_opacities(self, model, value=0.01):
        """
        Push every opacity back down near zero.

        Done periodically in the paper: Gaussians that are genuinely needed
        recover their opacity within a few hundred steps, while floaters that
        were only surviving on inertia fall below the prune threshold and are
        removed.
        """
        from .model import inverse_sigmoid
        cap = float(inverse_sigmoid(value))
        model.opacities_raw.data.clamp_(max=cap)


def _replace_parameters(model, means, log_scales, quats, sh, opacities):
    """Swap in a new population, keeping them as leaf parameters."""
    import torch.nn as nn
    model.means = nn.Parameter(means.contiguous())
    model.log_scales = nn.Parameter(log_scales.contiguous())
    model.quats = nn.Parameter(quats.contiguous())
    model.sh = nn.Parameter(sh.contiguous())
    model.opacities_raw = nn.Parameter(opacities.contiguous())


def build_optimizer(model, lr_means=1.6e-4, lr_scales=5e-3, lr_quats=1e-3,
                    lr_sh=2.5e-3, lr_opacity=5e-2, scene_extent=1.0):
    """
    Adam with a separate learning rate per parameter group.

    The rates differ by orders of magnitude because the quantities do. A
    position is in scene units, an opacity logit is dimensionless, and moving
    them at the same speed makes one of them diverge. The position rate is
    scaled by the scene extent so the same settings work whatever units the
    reconstruction came out in.
    """
    return torch.optim.Adam([
        {"params": [model.means], "lr": lr_means * scene_extent, "name": "means"},
        {"params": [model.log_scales], "lr": lr_scales, "name": "scales"},
        {"params": [model.quats], "lr": lr_quats, "name": "quats"},
        {"params": [model.sh], "lr": lr_sh, "name": "sh"},
        {"params": [model.opacities_raw], "lr": lr_opacity, "name": "opacity"},
    ], eps=1e-15)
