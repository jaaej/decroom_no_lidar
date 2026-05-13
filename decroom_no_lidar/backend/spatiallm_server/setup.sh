#!/bin/bash
# SpatialLM GPU Server Setup Script
# Run this on a fresh RunPod / Vast.ai instance (CUDA 12.4 PyTorch image).
#
# Installs everything into /workspace (the RunPod volume) so it survives
# pod stop/destroy. Earlier versions of this script installed to /opt,
# which lives on the ephemeral container disk and was wiped whenever the
# pod was destroyed-and-recreated.
#
# Usage (from inside the pod):
#   bash setup.sh
#
# Prerequisites: server.py copied to /workspace/server.py before running
# (or sitting in the same directory as this script).

set -e

VENV=/workspace/spatiallm-env
HF=/workspace/.cache/huggingface
SERVER_DIR=/workspace/spatiallm-server

echo "=== SpatialLM Server Setup (persistent /workspace install) ==="

# 1. System packages
echo ">>> Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq git python3-pip python3-venv > /dev/null

# 2. Persistent caches + venv on /workspace
echo ">>> Creating venv + cache dirs on /workspace..."
mkdir -p "$HF" "$SERVER_DIR"
python3 -m venv "$VENV"
source "$VENV/bin/activate"
export HF_HOME="$HF"

# 3. PyTorch with CUDA
echo ">>> Installing PyTorch..."
pip install --quiet torch==2.4.1 torchvision --index-url https://download.pytorch.org/whl/cu124

# 4. Core deps + pinned transformers/tokenizers (5.x breaks SpatialLM)
echo ">>> Installing core deps..."
pip install --quiet transformers==4.46.1 tokenizers==0.20.3 \
    open3d>=0.18.0 scipy scikit-learn numpy einops safetensors timm addict
pip install --quiet fastapi uvicorn python-multipart requests
pip install --quiet wheel setuptools --upgrade

# 5. spconv + torch-scatter
# IMPORTANT: spconv must match the CUDA version of torch (12.4). spconv-cu120
# imports cleanly but its sparse-conv kernels SIGFPE on cu124 PyTorch.
echo ">>> Installing spconv (cu124 — must match torch's CUDA version)..."
pip install --quiet spconv-cu124 || echo "WARNING: spconv-cu124 install failed"
echo ">>> Installing torch-scatter..."
pip install --quiet torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu124.html \
    || echo "WARNING: torch-scatter install failed"

# 6. flash-attn (optional, needs Ampere+ GPU)
echo ">>> Installing flash-attn (optional)..."
pip install --quiet flash-attn --no-build-isolation 2>/dev/null \
    || echo "NOTE: flash-attn skipped (works without on most GPUs)"

# 7. SpatialLM package — workaround for newer pip
# pip 24+ rejects 'torch>=2.4.1+cu124' because '+cu124' is a local version
# label and recent pip only accepts those with == or !=. Downgrade pip
# temporarily, install with --no-deps (we already have all deps), then
# restore.
echo ">>> Installing SpatialLM package (with pip downgrade workaround)..."
pip install --quiet 'pip<24'
pip install --quiet --no-deps git+https://github.com/manycore-research/SpatialLM.git
pip install --quiet --upgrade pip

# 8. Pre-cache model weights via huggingface_hub
# We use snapshot_download instead of AutoModelForCausalLM.from_pretrained
# because the latter requires the spatiallm package to be already imported
# to register the 'spatiallm_qwen' model_type — and on a fresh shell it
# isn't auto-registered, so AutoConfig fails with KeyError. snapshot_download
# just pulls all repo files into the HF cache, which is what we actually want.
echo ">>> Downloading model weights to $HF (this may take a few minutes)..."
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('manycore-research/SpatialLM1.1-Qwen-0.5B'); print('Model cached.')"

# Plan 3c: pre-cache Grounding DINO so server.py startup doesn't block on download.
echo ">>> Downloading Grounding DINO weights..."
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('IDEA-Research/grounding-dino-tiny'); print('Grounding DINO cached.')"

# 9. Copy server.py + code_template.txt into place
echo ">>> Setting up server..."
if [ -f /workspace/server.py ]; then
    cp /workspace/server.py "$SERVER_DIR/server.py"
elif [ -f "$(dirname "$0")/server.py" ]; then
    cp "$(dirname "$0")/server.py" "$SERVER_DIR/server.py"
else
    echo "WARNING: server.py not found at /workspace/server.py or alongside this script"
    echo "         scp it from your Mac before starting the server:"
    echo "         scp -P <PORT> -i ~/.ssh/id_ed25519 \\"
    echo "             <repo>/decroom_no_lidar/backend/spatiallm_server/server.py \\"
    echo "             root@<IP>:$SERVER_DIR/server.py"
fi

# code_template.txt is REQUIRED — server.py refuses to start without it.
# It lives next to server.py so the model gets the output schema in its prompt.
# Without it the model degenerates to single-token loops (Wall(11,11,11,...)).
echo ">>> Downloading SpatialLM code template..."
curl -fsSL -o "$SERVER_DIR/code_template.txt" \
    https://raw.githubusercontent.com/manycore-research/SpatialLM/main/code_template.txt \
    && echo "    code_template.txt downloaded ($(wc -c < "$SERVER_DIR/code_template.txt") bytes)" \
    || echo "WARNING: code_template.txt download failed — server will refuse to start"

# 10. Persistent launcher script
echo ">>> Writing /workspace/start-spatiallm.sh..."
printf '%s\n' \
    '#!/bin/bash' \
    "export HF_HOME=$HF" \
    "source $VENV/bin/activate" \
    "cd $SERVER_DIR" \
    'python server.py' \
    > /workspace/start-spatiallm.sh
chmod +x /workspace/start-spatiallm.sh

echo ""
echo "=== Setup Complete ==="
echo ""
echo "All of this lives on /workspace and persists across pod destroy."
echo ""
echo "To start the server:"
echo "  bash /workspace/start-spatiallm.sh"
echo ""
echo "Server listens on port 8100. Test with:"
echo "  curl http://localhost:8100/health"
