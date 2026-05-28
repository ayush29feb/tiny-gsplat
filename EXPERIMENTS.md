# Experiments

## exp1_baseline

**Config:** `training.max_steps=30000 training.batch_size=4 data.val_ids=[4]`

- Default strategy, no app_opt, no bilateral grid, no antialiasing, no absgrad, no random_bkgd
- Random init (100k points, no SfM)
- 28 train images, 1 val image (image 4 — closest to test view 0)

**Results:**

| Metric | Value |
|--------|-------|
| Peak val PSNR | **21.63 dB** (step 7k) |
| Final val PSNR | 20.80 dB (step 30k) |
| Gaussians | 1.18M |
| Training time | 691s (~11.5 min) on H100 |

**Val PSNR curve:**
- Peaks at step 7k (21.63 dB), then slowly degrades to 20.80 dB by 30k — overfitting to training views
- Densification stops at step 15k (default `refine_stop_iter`), Gaussian count stabilizes at ~1.18M

**Test renders (step 7k — peak val PSNR):**

| Test view 0 | Test view 1 |
|-------------|-------------|
| ![](assets/outputs/exp1_baseline/renders/render_0000_step7000.png) | ![](assets/outputs/exp1_baseline/renders/render_0001_step7000.png) |

**Val render (image 4, step 7k — peak val PSNR):**

![](assets/outputs/exp1_baseline/val_renders/val_0004_step7000.png)

**Observations:**
- Foreground (furniture, walls, whiteboard) is well-reconstructed
- Ceiling/background still has artifacts — consistent with sparse 29-view indoor capture
- Val render (image 4) looks good overall — foreground chair in bottom-right has some smearing at the camera edge
- Model overfits after step 7k — val PSNR drops 0.8 dB over remaining 23k steps while train PSNR keeps climbing

---

## exp2_scale_factor

**Changelog since exp1:**
- Added per-Gaussian `splat_scale_factor` parameter (shape [N, 1], init 1.0, raw space)
- Applied in `Splats.activate()`: `means *= scale_factor`, `scales *= scale_factor`
- LR 2e-3 with exponential decay to 1% (same schedule as means)

**Config:** `training.max_steps=30000 training.batch_size=4 data.val_ids=[4]`

- Same as exp1 baseline but with per-Gaussian scale factor enabled

**Results:**

| | Baseline (exp1) | Scale Factor (exp2) |
|---|---|---|
| Peak val PSNR | **21.63 dB** (step 7k) | 21.45 dB (step 4k) |
| Final val PSNR | **20.80 dB** | 20.58 dB |
| Gaussians | 1.18M | 1.82M |
| Training time | **691s** | 860s |

**Comparison — val PSNR, train PSNR, Gaussian count:**

![](assets/outputs/exp2_scale_factor/comparison.png)

**Test renders (step 4k — peak val PSNR):**

| Test view 0 | Test view 1 |
|-------------|-------------|
| ![](assets/outputs/exp2_scale_factor/renders/render_0000_step4000.png) | ![](assets/outputs/exp2_scale_factor/renders/render_0001_step4000.png) |

**Val render (image 4, step 4k — peak val PSNR):**

![](assets/outputs/exp2_scale_factor/val_renders/val_0004_step4000.png)

**Observations:**
- Scale factor adds 53% more Gaussians (1.82M vs 1.18M) without improving quality — slight 0.2 dB regression
- Extra parameter gives the strategy more freedom to create Gaussians during densification but they don't contribute useful reconstruction
- Overfitting pattern similar to baseline (peak at 4-7k, gradual decline)
- Test and val renders look comparable to baseline — no visible improvement or degradation
- The additional wall time (860s vs 691s) is from rendering more Gaussians

---

## exp3_full_7k

**Config:** `training.max_steps=7000 training.batch_size=4 data.test_every=0`

- All 29 images used for training (no val holdout)
- 7k steps — corresponds to peak val PSNR step from exp1
- Default strategy, splat_scale_factor off

**Results:**

| Metric | Value |
|--------|-------|
| Gaussians | 777k |
| Training time | 128s (~2 min) on H100 |

**Test renders (step 7k):**

| Test view 0 | Test view 1 |
|-------------|-------------|
| ![](assets/outputs/exp3_full_7k/renders/render_0000_step7000.png) | ![](assets/outputs/exp3_full_7k/renders/render_0001_step7000.png) |

**Observations:**
- Best quality renders so far — all 29 views used for training, stopped at sweet spot before overfitting
- Fast training: 128s for the full run

---

## exp4_garden

**Config:** `data_dir=data/garden_capture training.max_steps=30000 training.batch_size=4`

- Mip-NeRF 360 garden dataset (COLMAP format, converted to capture format)
- 185 images at 4x downsample (1297x840), 161 train / 24 val
- Sanity check: standard benchmark to verify our training code works on well-posed data

**Results:**

| Metric | Value |
|--------|-------|
| Peak val PSNR | **26.91 dB** (step 29k) |
| Final val PSNR | 26.89 dB (step 30k) |
| Gaussians | 4.82M |
| Training time | 1504s (~25 min) on H100 |

**Val render (image 0, step 30k):**

![](assets/outputs/exp4_garden/val_renders/val_0000_step30000.png)

**Auto observations:**
- Val PSNR reaches ~26.9 dB — competitive with published 3DGS results on garden (~27 dB)
- No overfitting — val PSNR keeps improving through 30k steps, unlike our capture dataset
- 4.82M Gaussians (vs 1.18M for capture) — more views support more Gaussians effectively
- Confirms our training code is correct — the quality gap on the capture dataset is from the data (29 sparse views), not a bug

**User observations:**
- No overfitting curve like we see in our capture — validates that the standard training pipeline has no severe bugs
- The capture dataset quality gap is a data problem, not a code problem
- Next steps: determine whether the capture issue is (1) a coverage problem (too few views, sparse angular sampling) or (2) a camera pose accuracy problem (calibration errors in the capture poses)
