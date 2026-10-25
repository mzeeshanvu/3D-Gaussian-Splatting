"""
The training objective.

The paper uses a weighted sum of an absolute-error term and a structural one,

    L = (1 - lambda) * L1 + lambda * (1 - SSIM),    lambda = 0.2

L1 alone drives the mean colour to the right place but is happy with a blurry
result, because blurring barely changes the average error. SSIM compares local
means, variances and covariance over a window, so it penalises lost texture and
pushes the Gaussians to resolve detail.
"""

import torch
import torch.nn.functional as F


def gaussian_kernel_1d(window=11, sigma=1.5):
    """A normalised 1D Gaussian, used separably for the SSIM windows."""
    x = torch.arange(window, dtype=torch.float32) - window // 2
    k = torch.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def _filter2d(img, k1d):
    """
    Separable Gaussian blur over an (H, W, C) image.

    Applied as two 1D passes rather than one 2D kernel: same result, far fewer
    multiplications.
    """
    c = img.shape[-1]
    x = img.permute(2, 0, 1).unsqueeze(0)                   # (1, C, H, W)
    pad = len(k1d) // 2
    kx = k1d.view(1, 1, 1, -1).expand(c, 1, 1, -1).to(img.dtype)
    ky = k1d.view(1, 1, -1, 1).expand(c, 1, -1, 1).to(img.dtype)
    x = F.conv2d(F.pad(x, (pad, pad, 0, 0), mode="reflect"), kx, groups=c)
    x = F.conv2d(F.pad(x, (0, 0, pad, pad), mode="reflect"), ky, groups=c)
    return x.squeeze(0).permute(1, 2, 0)


def ssim(a, b, window=11, sigma=1.5, c1=0.01 ** 2, c2=0.03 ** 2):
    """
    Mean structural similarity between two (H, W, C) images in 0..1.

    Returns 1.0 for identical images, so the loss below uses ``1 - ssim``.
    """
    k = gaussian_kernel_1d(window, sigma).to(a.device)
    mu_a, mu_b = _filter2d(a, k), _filter2d(b, k)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b

    # Var(x) = E[x^2] - E[x]^2, each expectation taken under the same window.
    var_a = _filter2d(a * a, k) - mu_a2
    var_b = _filter2d(b * b, k) - mu_b2
    cov_ab = _filter2d(a * b, k) - mu_ab

    num = (2 * mu_ab + c1) * (2 * cov_ab + c2)
    den = (mu_a2 + mu_b2 + c1) * (var_a + var_b + c2)
    return (num / den).mean()


def photometric_loss(rendered, target, lambda_ssim=0.2):
    """
    The combined objective, plus its parts for logging.

    :returns: (loss, l1, ssim_value)
    """
    l1 = torch.abs(rendered - target).mean()
    s = ssim(rendered, target)
    return (1.0 - lambda_ssim) * l1 + lambda_ssim * (1.0 - s), l1, s


def psnr(rendered, target):
    """Peak signal-to-noise ratio in dB -- the usual number to report."""
    mse = torch.mean((rendered - target) ** 2).clamp_min(1e-12)
    return float(10.0 * torch.log10(1.0 / mse))
