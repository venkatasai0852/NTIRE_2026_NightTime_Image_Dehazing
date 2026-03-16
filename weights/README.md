# Weights

Pre-trained and fine-tuned checkpoints are hosted externally due to file size.

## Download

| Checkpoint | Description | Link |
|---|---|---|
| `finetuned_phase4_ssim_ema.pth` | **Best submission** — Phase 4 SSIM+Charb+EMA | _[Add link]_ |
| `finetuned_phase3_highres_ema.pth` | Phase 3 — 512×512 high-res polish | _[Add link]_ |
| `finetuned_phase2_nighttime.pth` | Phase 2 — Nighttime domain adaptation | _[Add link]_ |
| `finetuned_phase1_mixup_ema.pth` | Phase 1 — MixUp/CutMix + EMA | _[Add link]_ |

## Placement

After downloading, place all `.pth` files in this `weights/` directory.

## Training Chain

```
Base DehazeFormer-B (official)
        ↓
Phase 1: MixUp/CutMix + EMA  →  finetuned_phase1_mixup_ema.pth
        ↓
Phase 2: Nighttime Physics    →  finetuned_phase2_nighttime.pth
        ↓
Phase 3: 512×512 Polish + EMA →  finetuned_phase3_highres_ema.pth
        ↓
Phase 4: SSIM+Charb + EMA    →  finetuned_phase4_ssim_ema.pth  ← submission
```
