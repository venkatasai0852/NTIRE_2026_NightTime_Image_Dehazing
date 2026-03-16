"""
losses/base_loss.py
Custom loss functions used across training phases.
"""

import torch
import torch.nn as nn
from pytorch_msssim import ssim


class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss — differentiable approximation of L1.
    More robust to outliers than MSE; preferred for image restoration.

    L = mean( sqrt( (x - y)^2 + eps^2 ) )
    """
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + self.eps ** 2))


class SSIMCharbonnierLoss(nn.Module):
    """
    Combined Charbonnier + SSIM loss.
    Used in Phase 4 to improve structural/perceptual quality.

    L = Charbonnier(x, y) + alpha * (1 - SSIM(x, y))
    """
    def __init__(self, eps: float = 1e-3, ssim_weight: float = 0.2):
        super().__init__()
        self.charb = CharbonnierLoss(eps)
        self.ssim_weight = ssim_weight

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        charb_loss = self.charb(x, y)
        ssim_loss  = 1.0 - ssim(x, y, data_range=1.0, size_average=True)
        return charb_loss + self.ssim_weight * ssim_loss
