#!/bin/bash
# Start the SpatialLM inference server
source /opt/spatiallm-env/bin/activate
cd /opt/spatiallm-server
python server.py
