#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

python scripts/train_agri.py \
  --model ultralytics/cfg/models/v8/yolov8-p2.yaml \
  --data data/agri_uav.yaml \
  --imgsz 640 \
  --epochs 300 \
  --batch 16 \
  --device 0 \
  --optimizer SGD \
  amp=False \
  pretrained=False \
  --name agri_p2 \
  "$@"
