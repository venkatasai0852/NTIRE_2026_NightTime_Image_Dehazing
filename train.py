"""
train.py
Unified training script for all 4 fine-tuning phases.

Usage:
    python train.py --phase 1 --pretrained ./weights/base.pth
    python train.py --phase 2 --pretrained ./weights/finetuned_phase1_mixup_ema.pth
    python train.py --phase 3 --pretrained ./weights/finetuned_phase2_nighttime.pth
    python train.py --phase 4 --pretrained ./weights/finetuned_phase3_highres_ema.pth

Phase Summary:
    1 — MixUp / CutMix + EMA              (L1 loss, LR=2e-4)
    2 — Nighttime domain adaptation       (Charbonnier, LR=5e-5)
    3 — High-res 512x512 polish + EMA     (Charbonnier + grad accum, LR=1e-5)
    4 — SSIM + Charbonnier polish + EMA   (LR=5e-6)
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config import (
    WEIGHTS_DIR, AUG_INPUT_DIR, AUG_GT_DIR,
    BATCH_SIZE, NUM_WORKERS, EMA_DECAY,
    P1_LR, P1_EPOCHS, P1_MIX_PROB, P1_CUTMIX_PROB, P1_OUTPUT,
    P2_LR, P2_EPOCHS, P2_NOISE_PROB, P2_GLOW_PROB, P2_OUTPUT,
    P3_LR, P3_EPOCHS, P3_PATCH_SIZE, P3_BATCH_SIZE, P3_ACCUM_STEPS,
    P3_SUBSET, P3_OUTPUT,
    P4_LR, P4_EPOCHS, P4_BATCH_SIZE, P4_SUBSET, P4_SSIM_WEIGHT, P4_OUTPUT,
)
from model import build_model, load_weights
from data import PairedDataset, HighResDataset
from losses import CharbonnierLoss, SSIMCharbonnierLoss
from training.utils import ModelEMA, apply_mixup_cutmix, apply_nighttime_physics
from evaluation.metrics import batch_psnr


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def save_ema_checkpoint(model, ema, path):
    ema.apply_shadow()
    m = model.module if isinstance(model, nn.DataParallel) else model
    torch.save(m.state_dict(), path)
    ema.restore()
    print(f'  ★ Saved EMA checkpoint → {path}')


def save_checkpoint(model, path):
    m = model.module if isinstance(model, nn.DataParallel) else model
    torch.save(m.state_dict(), path)
    print(f'  ★ Saved checkpoint → {path}')


def prepare_model(pretrained, device):
    model = build_model()
    if pretrained and os.path.exists(pretrained):
        model = load_weights(model, pretrained)
    else:
        print(f'[train] Warning: pretrained weights not found ({pretrained}). Starting from scratch.')
    if torch.cuda.device_count() > 1:
        print(f'[train] Using {torch.cuda.device_count()} GPUs')
        model = nn.DataParallel(model)
    return model.to(device)


# ─── PHASE 1: MixUp / CutMix + EMA ───────────────────────────────────────────

def phase1(pretrained, device):
    print('\n▶ PHASE 1: MixUp/CutMix + EMA')
    model     = prepare_model(pretrained, device)
    ema       = ModelEMA(model, decay=EMA_DECAY)
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=P1_LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, P1_EPOCHS, eta_min=1e-6)
    loader    = DataLoader(
        PairedDataset(AUG_INPUT_DIR, AUG_GT_DIR),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
    )
    out_path  = os.path.join(WEIGHTS_DIR, P1_OUTPUT)
    best_loss = float('inf')

    for epoch in range(P1_EPOCHS):
        model.train()
        ep_loss = ep_psnr = 0.0
        pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{P1_EPOCHS}')
        for inp, tgt in pbar:
            inp, tgt = inp.to(device), tgt.to(device)
            inp, tgt = apply_mixup_cutmix(inp, tgt, mix_prob=P1_MIX_PROB, cutmix_prob=P1_CUTMIX_PROB)
            optimizer.zero_grad()
            out  = model(inp)
            loss = criterion(out, tgt)
            loss.backward()
            optimizer.step()
            ema.update()
            ep_loss += loss.item()
            ep_psnr += batch_psnr(torch.clamp(out, 0, 1), tgt)
            pbar.set_postfix(loss=f'{loss.item():.4f}', psnr=f'{batch_psnr(torch.clamp(out,0,1),tgt):.2f}')
        scheduler.step()
        avg = ep_loss / len(loader)
        print(f'  Epoch {epoch+1} | Loss {avg:.5f} | PSNR {ep_psnr/len(loader):.2f} dB')
        if avg < best_loss:
            best_loss = avg
            save_ema_checkpoint(model, ema, out_path)


# ─── PHASE 2: Nighttime Domain Adaptation ────────────────────────────────────

def phase2(pretrained, device):
    print('\n▶ PHASE 2: Nighttime Domain Adaptation')
    model     = prepare_model(pretrained, device)
    criterion = CharbonnierLoss()
    optimizer = optim.AdamW(model.parameters(), lr=P2_LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, P2_EPOCHS, eta_min=1e-7)
    loader    = DataLoader(
        PairedDataset(AUG_INPUT_DIR, AUG_GT_DIR),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
    )
    out_path  = os.path.join(WEIGHTS_DIR, P2_OUTPUT)
    best_loss = float('inf')

    for epoch in range(P2_EPOCHS):
        model.train()
        ep_loss = 0.0
        pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{P2_EPOCHS}')
        for inp, tgt in pbar:
            inp, tgt = inp.to(device), tgt.to(device)
            inp = apply_nighttime_physics(inp, P2_NOISE_PROB, P2_GLOW_PROB)
            optimizer.zero_grad()
            out  = model(inp)
            loss = criterion(out, tgt)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.5f}')
        scheduler.step()
        avg = ep_loss / len(loader)
        print(f'  Epoch {epoch+1} | Loss {avg:.6f}')
        if avg < best_loss:
            best_loss = avg
            save_checkpoint(model, out_path)


# ─── PHASE 3: High-Res 512x512 Polish + EMA ──────────────────────────────────

def phase3(pretrained, device):
    print('\n▶ PHASE 3: High-Res 512×512 + EMA')
    model     = prepare_model(pretrained, device)
    ema       = ModelEMA(model, decay=EMA_DECAY)
    criterion = CharbonnierLoss()
    optimizer = optim.AdamW(model.parameters(), lr=P3_LR)
    loader    = DataLoader(
        HighResDataset(AUG_INPUT_DIR, AUG_GT_DIR, P3_PATCH_SIZE, P3_SUBSET),
        batch_size=P3_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    out_path  = os.path.join(WEIGHTS_DIR, P3_OUTPUT)
    best_loss = float('inf')

    for epoch in range(P3_EPOCHS):
        model.train()
        ep_loss = 0.0
        pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{P3_EPOCHS}')
        optimizer.zero_grad()
        for i, (inp, tgt) in enumerate(pbar):
            inp, tgt = inp.to(device), tgt.to(device)
            out  = model(inp)
            loss = criterion(out, tgt) / P3_ACCUM_STEPS
            loss.backward()
            if (i + 1) % P3_ACCUM_STEPS == 0 or (i + 1) == len(loader):
                optimizer.step()
                ema.update()
                optimizer.zero_grad()
            ep_loss += loss.item() * P3_ACCUM_STEPS
            pbar.set_postfix(loss=f'{loss.item()*P3_ACCUM_STEPS:.4f}')
        torch.cuda.empty_cache()
        avg = ep_loss / len(loader)
        print(f'  Epoch {epoch+1} | Loss {avg:.5f}')
        if avg < best_loss:
            best_loss = avg
            save_ema_checkpoint(model, ema, out_path)


# ─── PHASE 4: SSIM + Charbonnier Polish + EMA ────────────────────────────────

def phase4(pretrained, device):
    print('\n▶ PHASE 4: SSIM + Charbonnier Polish + EMA')
    model     = prepare_model(pretrained, device)
    ema       = ModelEMA(model, decay=EMA_DECAY)
    criterion = SSIMCharbonnierLoss(ssim_weight=P4_SSIM_WEIGHT)
    optimizer = optim.AdamW(model.parameters(), lr=P4_LR)
    loader    = DataLoader(
        PairedDataset(AUG_INPUT_DIR, AUG_GT_DIR, subset=P4_SUBSET),
        batch_size=P4_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    out_path  = os.path.join(WEIGHTS_DIR, P4_OUTPUT)
    best_loss = float('inf')

    for epoch in range(P4_EPOCHS):
        model.train()
        ep_loss = 0.0
        pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{P4_EPOCHS}')
        for inp, tgt in pbar:
            inp, tgt = inp.to(device), tgt.to(device)
            optimizer.zero_grad()
            out  = model(inp)
            loss = criterion(out, tgt)
            loss.backward()
            optimizer.step()
            ema.update()
            ep_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4f}')
        avg = ep_loss / len(loader)
        print(f'  Epoch {epoch+1} | Loss {avg:.5f}')
        if avg < best_loss:
            best_loss = avg
            save_ema_checkpoint(model, ema, out_path)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

PHASES = {1: phase1, 2: phase2, 3: phase3, 4: phase4}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--phase',      type=int, required=True, choices=[1, 2, 3, 4])
    p.add_argument('--pretrained', type=str, default=None,
                   help='Path to pretrained/previous phase checkpoint')
    return p.parse_args()


if __name__ == '__main__':
    args   = parse_args()
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[train] Device: {device}')
    PHASES[args.phase](args.pretrained, device)
    print('\n[train] Done.')
