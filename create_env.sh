#!/bin/bash
set -e

ENV_NAME="${1:-tiny-gsplat}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Find micromamba
if command -v micromamba &> /dev/null; then
    MICROMAMBA_BIN="$(command -v micromamba)"
    echo "Found micromamba in PATH: $MICROMAMBA_BIN"
elif [ -x "$MAMBA_EXE" ]; then
    MICROMAMBA_BIN="$MAMBA_EXE"
    echo "Found micromamba at: $MICROMAMBA_BIN"
    export PATH="$(dirname "$MICROMAMBA_BIN"):$PATH"
else
    echo "Searching for micromamba..."
    MICROMAMBA_BIN="$(find "$HOME" -name "micromamba" -type f -executable -print -quit 2>/dev/null || true)"
    if [ -n "$MICROMAMBA_BIN" ]; then
        echo "Found micromamba at: $MICROMAMBA_BIN"
        export PATH="$(dirname "$MICROMAMBA_BIN"):$PATH"
    else
        echo "Error: micromamba not found"
        exit 1
    fi
fi

# Check if environment already exists
if micromamba env list | grep -w "$ENV_NAME" > /dev/null 2>&1; then
    echo "Error: Environment '$ENV_NAME' already exists!"
    echo "To recreate: micromamba env remove -n $ENV_NAME"
    exit 1
fi

echo "=== Creating micromamba environment: $ENV_NAME ==="
micromamba create -n "$ENV_NAME" python=3.11 -y -c conda-forge

echo "Activating environment..."
eval "$(micromamba shell hook --shell bash)"
micromamba activate "$ENV_NAME"

echo "Installing build tools and CUDA toolkit..."
micromamba install -y pip wheel setuptools cmake ninja cuda-toolkit=12.8 -c nvidia -c conda-forge

echo "Installing PyTorch with CUDA 12.8..."
pip install torch --index-url https://download.pytorch.org/whl/cu128

echo "Verifying PyTorch..."
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA={torch.version.cuda}')"

GSPLAT_DIR="${GSPLAT_DIR:-$HOME/code/gsplat}"
echo "Initializing gsplat submodules..."
(cd "$GSPLAT_DIR" && git submodule update --init --recursive)

echo "Installing gsplat from source ($GSPLAT_DIR)..."
pip install --no-build-isolation -e "$GSPLAT_DIR"

echo "Installing tiny-gsplat and remaining dependencies..."
pip install --no-build-isolation -e "$SCRIPT_DIR"

echo ""
echo "=== Verifying installation ==="
python -c "import torch; print(f'  torch={torch.__version__}')"
python -c "import gsplat; print('  gsplat ok')"
python -c "from fused_ssim import fused_ssim; print('  fused_ssim ok')"
python -c "import tyro; print('  tyro ok')"

echo ""
echo "=== Done! ==="
echo "  micromamba activate $ENV_NAME"
echo "  cd $SCRIPT_DIR"
echo "  python train.py --data_dir /path/to/capture --result_dir ./results"
