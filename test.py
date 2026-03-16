"""
test.py
Inference script with 8-fold geometric TTA and tiled inference.

Usage:
    # Standard inference (full-image with padding)
    python test.py --input ./data/test/input --output ./results \
                   --weights ./weights/finetuned_phase4_ssim_ema.pth

    # With tiled inference (for very large images)
    python test.py --input ./data/test/input --output ./results \
                   --weights ./weights/finetuned_phase4_ssim_ema.pth \
                   --tiled

    # Skip TTA for faster inference
    python test.py --input ./data/test/input --output ./results \
                   --weights ./weights/finetuned_phase4_ssim_ema.pth \
                   --no_tta
"""

import os
import argparse
import shutil
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from configs.config import ALIGN, TTA_FOLDS, TILE_SIZE, TILE_OVERLAP
from model import get_model


# ─── TTA ──────────────────────────────────────────────────────────────────────

def forward_tta(model, x: torch.Tensor) -> torch.Tensor:
    """
    8-fold geometric TTA: original + 7 flips/rotations.
    All augmentations are invertible so outputs are averaged in original space.

    Augmentations:
        0: original
        1: hflip
        2: vflip
        3: hflip + vflip
        4: rot90
        5: rot90 + hflip
        6: rot90 + vflip
        7: rot90 + hflip + vflip
    """
    def _rot90(t):  return torch.rot90(t, 1, [2, 3])
    def _irot90(t): return torch.rot90(t, 3, [2, 3])

    preds = []

    # Original axes
    for hf in [False, True]:
        for vf in [False, True]:
            aug = x
            if hf: aug = torch.flip(aug, [3])
            if vf: aug = torch.flip(aug, [2])
            out = model(aug)
            if vf: out = torch.flip(out, [2])
            if hf: out = torch.flip(out, [3])
            preds.append(out)

    # Rotated axes
    for hf in [False, True]:
        for vf in [False, True]:
            aug = _rot90(x)
            if hf: aug = torch.flip(aug, [3])
            if vf: aug = torch.flip(aug, [2])
            out = model(aug)
            if vf: out = torch.flip(out, [2])
            if hf: out = torch.flip(out, [3])
            preds.append(_irot90(out))

    return torch.stack(preds).mean(0)


def forward_simple(model, x: torch.Tensor) -> torch.Tensor:
    return model(x)


# ─── PAD HELPER ───────────────────────────────────────────────────────────────

def pad_to_align(x: torch.Tensor, align: int = 16):
    _, _, H, W = x.shape
    pad_h = (align - H % align) % align
    pad_w = (align - W % align) % align
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    return x, H, W


# ─── TILED INFERENCE ──────────────────────────────────────────────────────────

def forward_tiled(model, x: torch.Tensor, tile: int, overlap: int,
                  use_tta: bool) -> torch.Tensor:
    """
    Process a large image by running the model on overlapping tiles.
    Tiles are blended with a Gaussian weight to avoid border artifacts.
    """
    _, C, H, W = x.shape
    stride = tile - overlap

    output = torch.zeros_like(x)
    weight = torch.zeros(1, 1, H, W, device=x.device)

    # Gaussian weight map for one tile
    g = torch.ones(1, 1, tile, tile, device=x.device)
    pad = overlap // 2
    g[:, :, :pad, :] *= 0.5
    g[:, :, -pad:, :] *= 0.5
    g[:, :, :, :pad] *= 0.5
    g[:, :, :, -pad:] *= 0.5

    fwd = forward_tta if use_tta else forward_simple

    for y in range(0, max(H - tile, 0) + 1, stride):
        for x_pos in range(0, max(W - tile, 0) + 1, stride):
            y2 = min(y + tile, H)
            x2 = min(x_pos + tile, W)
            y1 = max(0, y2 - tile)
            x1 = max(0, x2 - tile)

            tile_in = x[:, :, y1:y2, x1:x2]
            tile_in, th, tw = pad_to_align(tile_in)
            with torch.no_grad():
                tile_out = fwd(model, tile_in)
            tile_out = tile_out[:, :, :th, :tw]

            gh = min(g.shape[2], y2 - y1)
            gw = min(g.shape[3], x2 - x1)
            output[:, :, y1:y2, x1:x2] += tile_out * g[:, :, :gh, :gw]
            weight[:, :, y1:y2, x1:x2] += g[:, :, :gh, :gw]

    return output / weight.clamp(min=1e-6)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input',   required=True)
    p.add_argument('--output',  required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--no_tta',  action='store_true', help='Disable TTA')
    p.add_argument('--tiled',   action='store_true', help='Use tiled inference')
    p.add_argument('--tile',    type=int, default=TILE_SIZE)
    p.add_argument('--overlap', type=int, default=TILE_OVERLAP)
    p.add_argument('--zip',     action='store_true', help='Zip output folder for submission')
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[test] Device: {device}')
    print(f'[test] TTA: {not args.no_tta} ({TTA_FOLDS}-fold) | Tiled: {args.tiled}')

    model = get_model(args.weights).to(device)
    os.makedirs(args.output, exist_ok=True)

    EXTS = ('.png', '.jpg', '.jpeg', '.bmp')
    files = sorted([
        f for f in os.listdir(args.input)
        if f.lower().endswith(EXTS)
    ])
    print(f'[test] {len(files)} images found')

    to_tensor = T.ToTensor()
    to_pil    = T.ToPILImage()
    use_tta   = not args.no_tta

    for fname in tqdm(files, desc='Dehazing'):
        img    = Image.open(os.path.join(args.input, fname)).convert('RGB')
        W, H   = img.size
        tensor = to_tensor(img).unsqueeze(0).to(device)

        with torch.no_grad():
            if args.tiled:
                out = forward_tiled(model, tensor, args.tile, args.overlap, use_tta)
            else:
                padded, pH, pW = pad_to_align(tensor, ALIGN)
                fwd = forward_tta if use_tta else forward_simple
                out = fwd(model, padded)
                out = out[:, :, :pH, :pW]

        out_img = to_pil(torch.clamp(out, 0, 1).squeeze(0).cpu())
        out_img.save(os.path.join(args.output, fname))

    print(f'[test] Results saved to {args.output}')

    if args.zip:
        zip_path = args.output.rstrip('/\\')
        shutil.make_archive(zip_path, 'zip', args.output)
        print(f'[test] Zipped → {zip_path}.zip')


if __name__ == '__main__':
    main()
