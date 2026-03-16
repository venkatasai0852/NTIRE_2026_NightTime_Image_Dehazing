"""
model.py
DehazeFormer-B entry point.

Architecture is included in models/dehazeformer.py — no external clone needed.
Original paper: https://arxiv.org/abs/2208.11697
"""

import os
import torch
import torch.nn as nn
from collections import OrderedDict

from models.dehazeformer import dehazeformer_b


def build_model():
    """Instantiate DehazeFormer-B."""
    return dehazeformer_b()


def load_weights(model: nn.Module, ckpt_path: str, strict: bool = False) -> nn.Module:
    """
    Load weights from a checkpoint, handling:
      - 'state_dict' key wrapping
      - DataParallel 'module.' prefix
      - Minor shape mismatches (Conv2d kernel crop, 1D -> 4D reshape)
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)

    # Strip DataParallel prefix
    clean = OrderedDict()
    for k, v in state_dict.items():
        clean[k[7:] if k.startswith('module.') else k] = v

    # Shape patching
    model_state = model.state_dict()
    patched = OrderedDict()
    skipped = []
    for k, v in clean.items():
        if k not in model_state:
            continue
        target = model_state[k].shape
        if v.shape != target:
            # Conv2d: crop 3x3 -> 2x2
            if v.ndim == 4 and v.shape[2] == 3 and target[2] == 2:
                v = v[:, :, :2, :2]
            # Norm: [C] -> [1, C, 1, 1]
            elif v.ndim == 1 and len(target) == 4:
                v = v.view(1, -1, 1, 1)
        if v.shape == target:
            patched[k] = v
        else:
            skipped.append(k)

    if skipped:
        print(f"[load_weights] Skipped {len(skipped)} unresolvable keys: {skipped[:5]} ...")

    model.load_state_dict(patched, strict=strict)
    print(f"[load_weights] Loaded {len(patched)}/{len(model_state)} keys from {ckpt_path}")
    return model


def get_model(ckpt_path: str = None) -> nn.Module:
    """Build model and optionally load weights. Returns model in eval mode."""
    model = build_model()
    if ckpt_path:
        model = load_weights(model, ckpt_path)
    model.eval()
    return model
