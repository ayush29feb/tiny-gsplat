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

**Test render (step 30k):**

![](assets/outputs/exp1_baseline/renders/render_0001_step30000.png)

**Val render (image 4, step 30k):**

![](assets/outputs/exp1_baseline/val_renders/val_0004_step30000.png)

**Observations:**
- Foreground (furniture, walls, whiteboard) is well-reconstructed
- Ceiling/background still has artifacts — consistent with sparse 29-view indoor capture
- Val render (image 4) looks good overall — foreground chair in bottom-right has some smearing at the camera edge
- Model overfits after step 7k — val PSNR drops 0.8 dB over remaining 23k steps while train PSNR keeps climbing
