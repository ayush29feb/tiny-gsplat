# tiny-gsplat

Simple Gaussian splat training pipeline built on [gsplat](https://github.com/nerfstudio-project/gsplat). Reads capture datasets with known camera poses, trains 3D Gaussian splats, and renders novel views.

## Setup

```bash
./create_env.sh          # creates micromamba env with all deps
micromamba activate tiny-gsplat
```

See `create_env.sh` for details — it installs Python 3.11, CUDA toolkit 12.8, PyTorch, gsplat (from source), and all other dependencies.

## Capture format

```
capture/
├── inputs/
│   ├── metadata.json    # camera poses, intrinsics, image list
│   └── rgb_*.png        # training images
└── outputs/
    └── cameras.json     # novel-view camera poses for rendering
```

`metadata.json` contains:
- `camera.camera_to_world` — per-image 4x4 camera-to-world matrices
- `camera.camera_to_pixel` — per-image 3x3 intrinsic matrices
- `camera.image_size_xy` — image dimensions
- `pixel_data.rgb` — indices mapping to `rgb_*.png` files

## Usage

```bash
# Train with defaults
python train.py

# Override training params
python train.py training.max_steps=30000

# Set validation split
python train.py train_data.val_ids=[20] val_data.val_ids=[20]

# Enable appearance optimization + bilateral grid
python train.py model.app_opt=true model.bilateral_grid=true model.antialiased=true

# Swap densification strategy
python train.py strategy._target_=gsplat.strategy.MCMCStrategy

# Disable a logger
python train.py ~loggers.test_image
```

Outputs go to `outputs/<date>/<time>/` (managed by Hydra), containing checkpoints, rendered images, loss/PSNR plots, and the full config.

## Project structure

```
train.py            Training loop entry point (@hydra.main)
model.py            GaussianSplatModel — splats, appearance opt, bilateral grid
dataset.py          CaptureDataset — index-style PyTorch dataset
logger.py           Composable loggers with .step()/.finalize() interface
configs/train.yaml  Default Hydra config
lib_bilagrid.py     Bilateral grid module (from gsplat)
EXPERIMENTS.md      Experiment log with results and observations
```

### Model (`model.py`)

`GaussianSplatModel` wraps all learnable state:
- Splat parameters (means, scales, quats, opacities, SH coefficients)
- Optional `AppearanceOptModule` — per-image appearance embeddings + MLP
- Optional `BilateralGrid` — per-image color correction

`forward()` encapsulates the full render pipeline (color computation + rasterization + post-processing), so callers just pass camera pose + intrinsics.

### Loggers (`logger.py`)

Each logger has `step(step, model, **ctx)` and `finalize()`. They internally decide when to act based on their `every` parameter. The training loop calls all loggers every step — zero branching.

| Logger | What it does |
|--------|-------------|
| `TrainMetricsLogger` | Tracks loss curve, saves `train_loss.png` |
| `ValMetricsLogger` | Computes val PSNR, saves `val_psnr.png` |
| `ValImageLogger` | Renders val images at eval steps |
| `TestImageLogger` | Renders novel-view test cameras |
| `CheckpointLogger` | Saves model checkpoints |

### Config (`configs/train.yaml`)

Hydra config with structured groups: `model`, `lr`, `training`, `strategy`, `loggers`, `train_data`, `val_data`. All loggers use `_target_` for Hydra instantiate. Override anything from the CLI.

## Experiment tracking

See `EXPERIMENTS.md` for documented experiments with configs, results, and observations.
