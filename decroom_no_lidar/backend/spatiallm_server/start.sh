#!/bin/bash
# Start the SpatialLM inference server (persistent /workspace install).
# Equivalent to /workspace/start-spatiallm.sh — kept here for parity
# with the repo so anyone reading the source can start the pod from
# either location.
export HF_HOME=/workspace/.cache/huggingface
source /workspace/spatiallm-env/bin/activate
cd /workspace/spatiallm-server
python server.py
