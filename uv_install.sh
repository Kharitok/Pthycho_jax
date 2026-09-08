#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# USER CONFIGURATION: Set your cluster scratch path here
# ==============================================================================
# Edit this variable to point to your cluster scratch/work directory.
# Examples: "/scratch/$USER", "/work/$USER", "/gpfs/scratch/$USER"
SCRATCH_DIR=  # PUT YOUR CLUSTER SCRATCH PATH HERE
# ==============================================================================

echo "=== Setting up Ptycho JAX Environment ==="
echo "[1/4] Using scratch directory: $SCRATCH_DIR"

# 1. Create scratch directories to bypass home directory disk quota limits
mkdir -p "$SCRATCH_DIR/.cache/uv" "$SCRATCH_DIR/.local/share/uv/python"

# 2. Export environment variables for the current session
export UV_CACHE_DIR="$SCRATCH_DIR/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$SCRATCH_DIR/.local/share/uv/python"
export PATH="$HOME/.local/bin:$PATH"

# Persist variables in ~/.bashrc if not already present
if ! grep -q "UV_CACHE_DIR" ~/.bashrc 2>/dev/null; then
    echo "Adding uv scratch configuration to ~/.bashrc..."
    {
        echo ""
        echo "# uv cluster settings for Ptycho JAX"
        echo "export UV_CACHE_DIR=\"$SCRATCH_DIR/.cache/uv\""
        echo "export UV_PYTHON_INSTALL_DIR=\"$SCRATCH_DIR/.local/share/uv/python\""
        echo 'export PATH="$HOME/.local/bin:$PATH"'
    } >> ~/.bashrc
fi

# 3. Install uv in user-space if not already installed
if ! command -v uv &> /dev/null; then
    echo "[2/4] Installing uv in user space..."
    curl -sSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "[2/4] uv is already installed."
fi

# 4. Create .venv and install all dependencies
echo "[3/4] Syncing environment dependencies..."
uv sync

# 5. Register the kernel for Jupyter server discovery
echo "[4/4] Registering Jupyter kernel..."
uv run python -m ipykernel install --user --name=ptycho_jax_cu12 --display-name "Ptycho_JAX_cu12"

echo ""
echo "=== Setup complete! ==="
echo "To verify GPU support, run:"
echo "  uv run python -c \"import jax; print(jax.devices())\""