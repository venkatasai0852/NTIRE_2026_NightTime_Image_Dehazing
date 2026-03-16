"""
augment.py
Generate augmented patch dataset from the 25 official training images.

Produces: orig / hflip / vflip / rot90 patches at 256x256.

Usage:
    python augment.py \
        --input  ./data/train/input \
        --gt     ./data/train/gt \
        --output ./data/augmented \
        --target 9000
"""

import os
import cv2
import argparse
import random
import shutil
import numpy as np
from tqdm import tqdm

from configs.config import AUG_TARGET_TOTAL, AUG_PATCH_SIZE, AUG_STRIDE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input',  default='./data/train/input')
    p.add_argument('--gt',     default='./data/train/gt')
    p.add_argument('--output', default='./data/augmented')
    p.add_argument('--target', type=int, default=AUG_TARGET_TOTAL)
    p.add_argument('--patch',  type=int, default=AUG_PATCH_SIZE)
    p.add_argument('--stride', type=int, default=AUG_STRIDE)
    return p.parse_args()


def save_aug(img, gt, out_in, out_gt, base_name, suffix):
    name = f"{base_name}_{suffix}.png"
    cv2.imwrite(os.path.join(out_in, name), img)
    cv2.imwrite(os.path.join(out_gt, name), gt)


def process_one(filename, in_dir, gt_dir, out_in, out_gt,
                limit, patch_size, stride):
    img_h = cv2.imread(os.path.join(in_dir, filename))
    img_g = cv2.imread(os.path.join(gt_dir, filename))
    if img_h is None or img_g is None:
        return 0

    H, W = img_h.shape[:2]
    candidates = [
        (y, x)
        for y in range(0, H - patch_size + 1, stride)
        for x in range(0, W - patch_size + 1, stride)
    ]
    random.shuffle(candidates)

    patches_needed = int(np.ceil(limit / 4))
    generated = 0
    base = os.path.splitext(filename)[0]

    for (y, x) in candidates[:patches_needed]:
        ph = img_h[y:y+patch_size, x:x+patch_size]
        pg = img_g[y:y+patch_size, x:x+patch_size]
        bname = f"{base}_p{y}_{x}"

        save_aug(ph, pg, out_in, out_gt, bname, 'orig');  generated += 1
        if generated < limit:
            save_aug(cv2.flip(ph, 1), cv2.flip(pg, 1), out_in, out_gt, bname, 'hflip'); generated += 1
        if generated < limit:
            save_aug(cv2.flip(ph, 0), cv2.flip(pg, 0), out_in, out_gt, bname, 'vflip'); generated += 1
        if generated < limit:
            save_aug(
                cv2.rotate(ph, cv2.ROTATE_90_CLOCKWISE),
                cv2.rotate(pg, cv2.ROTATE_90_CLOCKWISE),
                out_in, out_gt, bname, 'rot90'
            ); generated += 1
        if generated >= limit:
            break

    return generated


def main():
    args = parse_args()

    if os.path.exists(args.output):
        print(f'[augment] Removing existing output: {args.output}')
        shutil.rmtree(args.output)

    out_in = os.path.join(args.output, 'input')
    out_gt = os.path.join(args.output, 'ground_truth')
    os.makedirs(out_in, exist_ok=True)
    os.makedirs(out_gt, exist_ok=True)

    all_files = sorted([
        f for f in os.listdir(args.input)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    n_files = len(all_files)
    print(f'[augment] {n_files} source images | target: {args.target} patches')

    total = 0
    per_file = args.target // n_files

    for fname in tqdm(all_files, desc='Augmenting'):
        remaining      = args.target - total
        files_left     = n_files - all_files.index(fname)
        limit          = min(per_file + 20,
                             int(np.ceil(remaining / max(files_left, 1))))
        if remaining <= 0:
            break
        total += process_one(
            fname, args.input, args.gt, out_in, out_gt,
            limit, args.patch, args.stride
        )

    print(f'\n[augment] Done. {total} patches saved to {args.output}')


if __name__ == '__main__':
    main()
