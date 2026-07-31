#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python scripts/train_agri.py \
  --model ultralytics/cfg/models/v8/yolov8n-p2-eca-simam-agri.yaml \
  --data data/agri_uav.yaml \
  --imgsz 640 \
  --epochs 350 \
  --batch 32 \
  --device 0 \
  --optimizer SGD \
  amp=False \
  pretrained=False \
  --name agri_p2_eca_simam \
  "$@"



