"""
evaluation/metrics.py
Metric utilities: batch PSNR (training), full PSNR/SSIM/LPIPS (evaluation).
"""

import torch
import numpy as np


def batch_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    Fast PSNR over a batch of tensors in [0, 1] range.
    Used during training for monitoring.

    Args:
        pred: predicted batch [B, C, H, W]
        gt:   ground-truth batch [B, C, H, W]

    Returns:
        mean PSNR (dB) as float
    """
    mse  = torch.mean((pred - gt) ** 2, dim=[1, 2, 3])
    mse  = torch.clamp(mse, min=1e-10)
    psnr = 20.0 * torch.log10(1.0 / torch.sqrt(mse))
    return torch.mean(psnr).item()


def compute_metrics(pred_np: np.ndarray, gt_np: np.ndarray,
                    device: torch.device = None,
                    lpips_model=None) -> dict:
    """
    Full per-image metric computation.

    Args:
        pred_np: predicted image [H, W, 3] uint8
        gt_np:   ground-truth image [H, W, 3] uint8
        device:  torch device (for LPIPS)
        lpips_model: pre-loaded lpips.LPIPS model (optional)

    Returns:
        dict with keys 'psnr', 'ssim', 'lpips' (lpips is None if model not given)
    """
    from skimage.metrics import peak_signal_noise_ratio as _psnr
    from skimage.metrics import structural_similarity as _ssim

    psnr_val = _psnr(gt_np, pred_np, data_range=255)
    ssim_val = _ssim(gt_np, pred_np, channel_axis=2, data_range=255)

    lpips_val = None
    if lpips_model is not None and device is not None:
        import torchvision.transforms as T
        tf = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        from PIL import Image
        gt_t   = tf(Image.fromarray(gt_np)).unsqueeze(0).to(device)
        pred_t = tf(Image.fromarray(pred_np)).unsqueeze(0).to(device)
        with torch.no_grad():
            lpips_val = lpips_model(gt_t, pred_t).item()

    return {'psnr': psnr_val, 'ssim': ssim_val, 'lpips': lpips_val}
