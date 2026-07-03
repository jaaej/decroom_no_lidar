#!/usr/bin/env bash
set -euxo pipefail
# Run ON THE POD. Python 3.11, torch >= 2.5.1 (spec constraint).
conda create -y -n phase0 python=3.11
source activate phase0
pip install "torch>=2.5.1" torchvision --index-url https://download.pytorch.org/whl/cu124
pip install "transformers>=4.55" accelerate pillow opencv-python-headless open3d numpy scipy av decord huggingface_hub matplotlib
# SAM 2 (video predictor needs the repo package, not just HF weights)
git clone https://github.com/facebookresearch/sam2 /root/sam2 && pip install -e /root/sam2
# MoGe
pip install git+https://github.com/microsoft/MoGe.git
# Orient Anything V1 (repo code, no pip package)
git clone https://github.com/SpatialVision/Orient-Anything /root/orient-anything
mkdir -p /root/ckpts
python - <<'EOF'
from huggingface_hub import hf_hub_download, snapshot_download
snapshot_download("iSEE-Laboratory/llmdet_large", local_dir="/root/ckpts/llmdet_large")
hf_hub_download("facebook/sam2.1-hiera-large", "sam2.1_hiera_large.pt", local_dir="/root/ckpts/sam2")
snapshot_download("Ruicheng/moge-2-vitl-normal", local_dir="/root/ckpts/moge2")
# Orient Anything V1 Large ckpt (croplargeEX2); Base fallback file is cropbaseEx03 (README's cropbaseEx032 link is broken)
hf_hub_download("Viglong/Orient-Anything", "croplargeEX2/dino_weight.pt", local_dir="/root/ckpts/orient")
EOF
