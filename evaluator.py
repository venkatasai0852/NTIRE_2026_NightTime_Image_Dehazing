"""
evaluator.py
Compute PSNR / SSIM / LPIPS between two folders of images.

Usage:
    python evaluator.py --pred ./results --gt ./data/val/gt
"""

import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import lpips

from evaluation.metrics import compute_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pred', required=True,  help='Folder with predicted images')
    p.add_argument('--gt',   required=True,  help='Folder with ground-truth images')
    p.add_argument('--no_lpips', action='store_true', help='Skip LPIPS (faster)')
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lpips_model = None
    if not args.no_lpips:
        print('[evaluator] Loading LPIPS model (AlexNet)...')
        lpips_model = lpips.LPIPS(net='alex').to(device).eval()

    gt_files = sorted([
        f for f in os.listdir(args.gt)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    results = {'psnr': [], 'ssim': [], 'lpips': []}

    print(f"\n{'IMAGE':<30}  {'PSNR':>8}  {'SSIM':>8}  {'LPIPS':>8}")
    print('-' * 62)

    for fname in tqdm(gt_files, desc='Evaluating'):
        gt_path   = os.path.join(args.gt, fname)
        pred_path = os.path.join(args.pred, fname)

        if not os.path.exists(pred_path):
            print(f'  [skip] {fname} not found in --pred folder')
            continue

        gt_img   = np.array(Image.open(gt_path).convert('RGB'))
        pred_img = np.array(Image.open(pred_path).convert('RGB'))

        if gt_img.shape != pred_img.shape:
            from PIL import Image as PILImage
            pred_img = np.array(
                PILImage.fromarray(pred_img).resize(
                    (gt_img.shape[1], gt_img.shape[0]), PILImage.BICUBIC
                )
            )

        m = compute_metrics(pred_img, gt_img, device, lpips_model)
        results['psnr'].append(m['psnr'])
        results['ssim'].append(m['ssim'])
        if m['lpips'] is not None:
            results['lpips'].append(m['lpips'])

        lpips_str = f"{m['lpips']:.4f}" if m['lpips'] is not None else '  N/A  '
        print(f"  {fname[:28]:<28}  {m['psnr']:>8.2f}  {m['ssim']:>8.4f}  {lpips_str:>8}")

    print('\n' + '=' * 62)
    print(f"  Images evaluated : {len(results['psnr'])}")
    print(f"  Avg PSNR         : {np.mean(results['psnr']):.4f} dB")
    print(f"  Avg SSIM         : {np.mean(results['ssim']):.4f}")
    if results['lpips']:
        print(f"  Avg LPIPS        : {np.mean(results['lpips']):.4f}")
    print('=' * 62)


if __name__ == '__main__':
    main()
