# NTIRE 2026 Nighttime Image Dehazing

**CVPR 2026 Workshop — NTIRE Challenge**  
Team: **[Your Team Name]** | Affiliation: SVNIT Surat  
**Achieved: 24.83 dB PSNR** on the validation leaderboard

---

## Method Overview

We fine-tune **DehazeFormer-B** ([Song et al., TPAMI 2023](https://arxiv.org/abs/2208.11697)) on the NTIRE 2026 nighttime dehazing training set using a 4-phase curriculum:

| Phase | Description | Loss | Key Technique |
|---|---|---|---|
| 1 | Structural generalization | L1 | MixUp + CutMix + EMA |
| 2 | Nighttime domain adaptation | Charbonnier | Synthetic glow & noise |
| 3 | High-resolution polish | Charbonnier | 512×512 patches + grad accum + EMA |
| 4 | Perceptual refinement | SSIM + Charbonnier | EMA |

**Inference**: 8-fold geometric TTA (4 flip combinations × 2 rotation axes) on full images with reflection padding.

---

## Setup

```bash
# 1. Clone this repo
git clone https://github.com/venkatasai0852/<repo-name>.git
cd <repo-name>

# 2. Clone DehazeFormer architecture
git clone https://github.com/IDKiro/DehazeFormer.git

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download weights (see weights/README.md)
```

Update `DEHAZEFORMER_REPO_PATH` in `configs/config.py` if needed (default: `./DehazeFormer`).

---

## Reproduce Submission

```bash
python test.py \
    --input  ./data/test/input \
    --output ./results \
    --weights ./weights/finetuned_phase4_ssim_ema.pth \
    --zip
```

This runs 8-fold TTA and saves a `results.zip` ready for submission.

---

## Training from Scratch

```bash
# Step 1: Generate augmented patches (~9000) from official 25 training images
python augment.py \
    --input ./data/train/input \
    --gt    ./data/train/gt

# Step 2–5: Run phases sequentially
python train.py --phase 1 --pretrained ./weights/base_dehazeformer_b.pth
python train.py --phase 2 --pretrained ./weights/finetuned_phase1_mixup_ema.pth
python train.py --phase 3 --pretrained ./weights/finetuned_phase2_nighttime.pth
python train.py --phase 4 --pretrained ./weights/finetuned_phase3_highres_ema.pth
```

---

## Evaluation

```bash
python evaluator.py --pred ./results --gt ./data/val/gt
```

---

## Repository Structure

```
ntire2026-nighttime-dehazing/
├── model.py          # DehazeFormer-B loader with weight patching
├── train.py          # Unified training script (all 4 phases)
├── test.py           # Inference with 8-fold TTA + tiled inference
├── augment.py        # Patch augmentation from official training set
├── evaluator.py      # PSNR / SSIM / LPIPS evaluation
├── data.py           # Dataset classes
├── losses/
│   ├── base_loss.py  # CharbonnierLoss, SSIMCharbonnierLoss
├── training/
│   └── utils.py      # ModelEMA, MixUp/CutMix, nighttime physics aug
├── evaluation/
│   └── metrics.py    # batch_psnr, compute_metrics
├── configs/
│   └── config.py     # All hyperparameters and paths
└── weights/          # Place downloaded checkpoints here
```

---

## Citation

If you use this code, please cite the original DehazeFormer:

```bibtex
@article{song2023vision,
  title={Vision Transformers for Single Image Dehazing},
  author={Song, Yuda and He, Zhuqing and Qian, Hui and Du, Xin},
  journal={IEEE Transactions on Image Processing},
  year={2023}
}
```
