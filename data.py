"""
data.py
Dataset classes for training and inference.
"""

import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class PairedDataset(Dataset):
    """
    Standard paired dataset: hazy input + ground truth.
    Used for Phase 1, 2, 4 training (256x256 patches).
    """
    def __init__(self, input_dir: str, gt_dir: str, subset: int = None):
        in_files = sorted(glob.glob(os.path.join(input_dir, '*.png')))
        gt_files = sorted(glob.glob(os.path.join(gt_dir, '*.png')))

        n = min(len(in_files), len(gt_files))
        if subset:
            n = min(n, subset)

        self.in_files = in_files[:n]
        self.gt_files = gt_files[:n]
        self.transform = T.ToTensor()

        if n == 0:
            raise ValueError(f"No images found in {input_dir}")
        print(f"[PairedDataset] {n} pairs loaded from {input_dir}")

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, idx):
        try:
            inp = self.transform(Image.open(self.in_files[idx]).convert('RGB'))
            gt  = self.transform(Image.open(self.gt_files[idx]).convert('RGB'))
            return inp, gt
        except Exception:
            return self.__getitem__(0)


class HighResDataset(Dataset):
    """
    Resizes patches to PATCH_SIZE x PATCH_SIZE.
    Used for Phase 3 high-resolution polish (512x512).
    """
    def __init__(self, input_dir: str, gt_dir: str,
                 patch_size: int = 512, subset: int = 3000):
        in_files = sorted(glob.glob(os.path.join(input_dir, '*.png')))
        gt_files = sorted(glob.glob(os.path.join(gt_dir, '*.png')))

        n = min(len(in_files), len(gt_files), subset)
        self.in_files = in_files[:n]
        self.gt_files = gt_files[:n]
        self.transform = T.Compose([
            T.Resize((patch_size, patch_size), Image.BICUBIC),
            T.ToTensor()
        ])
        print(f"[HighResDataset] {n} pairs at {patch_size}x{patch_size}")

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, idx):
        try:
            inp = self.transform(Image.open(self.in_files[idx]).convert('RGB'))
            gt  = self.transform(Image.open(self.gt_files[idx]).convert('RGB'))
            return inp, gt
        except Exception:
            return self.__getitem__(0)


class InferenceDataset(Dataset):
    """Single-image dataset for test-time inference (no GT needed)."""
    EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

    def __init__(self, input_dir: str):
        self.files = sorted([
            f for f in glob.glob(os.path.join(input_dir, '*'))
            if f.lower().endswith(self.EXTS)
        ])
        self.to_tensor = T.ToTensor()
        print(f"[InferenceDataset] {len(self.files)} images found in {input_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img  = Image.open(path).convert('RGB')
        return self.to_tensor(img), os.path.basename(path), img.size  # tensor, name, (W,H)
