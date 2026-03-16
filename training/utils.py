"""
training/utils.py
Training utilities: EMA, MixUp/CutMix, nighttime physics augmentation.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


# ─── EMA ──────────────────────────────────────────────────────────────────────

class ModelEMA:
    """
    Exponential Moving Average of model weights.
    Produces a smoother, generalization-friendly model for saving/evaluation.

    Usage:
        ema = ModelEMA(model, decay=0.999)
        # after each optimizer.step():
        ema.update()
        # to save EMA weights:
        ema.apply_shadow()
        torch.save(...)
        ema.restore()
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay  = decay
        self.model  = model
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    (1.0 - self.decay) * param.data
                    + self.decay * self.shadow[name]
                )

    def apply_shadow(self):
        """Swap EMA weights into the model (for saving/eval)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore training weights after apply_shadow."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# ─── MIXUP / CUTMIX ───────────────────────────────────────────────────────────

def apply_mixup_cutmix(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
    mix_prob: float = 0.5,
    cutmix_prob: float = 0.5,
) -> tuple:
    """
    Randomly apply MixUp or CutMix to a batch.
    Applied to both input (hazy) and target (GT) identically.

    Args:
        x: hazy image batch [B, C, H, W]
        y: GT image batch   [B, C, H, W]
        alpha: Beta distribution parameter
        mix_prob: probability of applying any mixing
        cutmix_prob: given mixing is applied, probability of CutMix vs MixUp

    Returns:
        (mixed_x, mixed_y)
    """
    if random.random() > mix_prob:
        return x, y

    lam   = np.random.beta(alpha, alpha)
    B     = x.size(0)
    index = torch.randperm(B, device=x.device)

    if random.random() < cutmix_prob:
        # CutMix: replace a rectangular region
        H, W = x.shape[2], x.shape[3]
        cut_w = int(W * np.sqrt(1.0 - lam))
        cut_h = int(H * np.sqrt(1.0 - lam))
        cx, cy = np.random.randint(W), np.random.randint(H)

        x1 = np.clip(cx - cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y2 = np.clip(cy + cut_h // 2, 0, H)

        x = x.clone()
        y = y.clone()
        x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
        y[:, :, y1:y2, x1:x2] = y[index, :, y1:y2, x1:x2]
    else:
        # MixUp: global pixel blend
        x = lam * x + (1 - lam) * x[index]
        y = lam * y + (1 - lam) * y[index]

    return x, y


# ─── NIGHTTIME PHYSICS AUGMENTATION ──────────────────────────────────────────

def apply_nighttime_physics(
    x: torch.Tensor,
    noise_prob: float = 0.5,
    glow_prob: float = 0.5,
) -> torch.Tensor:
    """
    Synthetic nighttime augmentation applied to input hazy images (GPU tensor).

    1. Low-light sensor noise  — Gaussian noise at random intensity
    2. Streetlight glow effect — blur of bright regions re-added as halo

    Only applied to the hazy INPUT, not the GT.

    Args:
        x: hazy batch [B, C, H, W] in [0, 1]

    Returns:
        augmented batch, clamped to [0, 1]
    """
    x_aug = x.clone()

    # 1. Additive Gaussian noise simulating low-light sensor noise
    if random.random() < noise_prob:
        noise_std = random.uniform(0.01, 0.08)
        x_aug = x_aug + torch.randn_like(x_aug) * noise_std

    # 2. Light glow / halo simulation
    if random.random() < glow_prob:
        threshold     = random.uniform(0.6, 0.8)
        bright_spots  = torch.clamp(x_aug - threshold, min=0.0)
        kernel_size   = random.choice([15, 21, 31])
        sigma         = random.uniform(3.0, 7.0)
        glow          = TF.gaussian_blur(
            bright_spots,
            kernel_size=[kernel_size, kernel_size],
            sigma=[sigma, sigma]
        )
        glow_intensity = random.uniform(0.5, 1.5)
        x_aug = x_aug + glow * glow_intensity

    return torch.clamp(x_aug, 0.0, 1.0)
