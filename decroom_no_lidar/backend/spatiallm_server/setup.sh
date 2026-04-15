#!/bin/bash
# SpatialLM GPU Server Setup Script
# Run this on a Vast.ai instance (or any CUDA machine)
#
# Usage: bash setup.sh

set -e

echo "=== SpatialLM Server Setup ==="

# 1. System packages
echo ">>> Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq git python3-pip python3-venv > /dev/null

# 2. Create virtual environment
echo ">>> Creating virtual environment..."
python3 -m venv /opt/spatiallm-env
source /opt/spatiallm-env/bin/activate

# 3. Install PyTorch with CUDA
echo ">>> Installing PyTorch..."
pip install --quiet torch==2.4.1 torchvision --index-url https://download.pytorch.org/whl/cu124

# 4. Install SpatialLM and dependencies
echo ">>> Installing SpatialLM..."
pip install --quiet transformers==4.46.1 open3d>=0.18.0 scipy scikit-learn numpy
pip install --quiet einops safetensors timm addict
pip install --quiet fastapi uvicorn python-multipart requests

# Install spconv for CUDA 12.4
echo ">>> Installing spconv..."
pip install --quiet spconv-cu120 || echo "WARNING: spconv install failed, will use fallback inference"

# Install torch-scatter
echo ">>> Installing torch-scatter..."
pip install --quiet torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu124.html || echo "WARNING: torch-scatter install failed"

# Install flash-attn (optional, needs Ampere+ GPU)
echo ">>> Installing flash-attn (optional)..."
pip install --quiet flash-attn --no-build-isolation 2>/dev/null || echo "NOTE: flash-attn not installed (needs Ampere+ GPU), will work without it"

# Install SpatialLM package
echo ">>> Installing SpatialLM package..."
pip install --quiet git+https://github.com/manycore-research/SpatialLM.git || echo "WARNING: SpatialLM package install failed, will use basic inference"

# 5. Pre-download the model weights
echo ">>> Downloading model weights (this may take a few minutes)..."
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = 'manycore-research/SpatialLM1.1-Qwen-0.5B'
print('Downloading tokenizer...')
AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
print('Downloading model...')
AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
print('Model downloaded successfully!')
"

# 6. Copy server script
echo ">>> Setting up server..."
mkdir -p /opt/spatiallm-server
cp server.py /opt/spatiallm-server/server.py

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To start the server:"
echo "  source /opt/spatiallm-env/bin/activate"
echo "  cd /opt/spatiallm-server"
echo "  python server.py"
echo ""
echo "Server will listen on port 8100"
echo "Test with: curl http://localhost:8100/health"
