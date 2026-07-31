#!/usr/bin/env python3
"""Validate a trained AgriUAV YOLOv8 detector on a YOLO-format dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

DEFAULT_MODEL = "runs/detect/agri_p2_eca_simam/weights/best.pt"
DEFAULT_DATA = "data/agri_uav.yaml"


def parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_extra_args(items: list[str]) -> dict[str, Any]:
    extras: dict[str, Any] = {}
    i = 0
    while i < len(items):
        item = items[i]
        if "=" in item and not item.startswith("--"):
            key, value = item.split("=", 1)
            extras[key.replace("-", "_")] = parse_scalar(value)
            i += 1
            continue
        if item.startswith("--"):
            key = item[2:].replace("-", "_")
            if i + 1 < len(items) and not items[i + 1].startswith("--"):
                extras[key] = parse_scalar(items[i + 1])
                i += 2
            else:
                extras[key] = True
                i += 1
            continue
        raise SystemExit(f"Unsupported passthrough argument: {item}")
    return extras


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Trained .pt weights path.")
    parser.add_argument("--data", default=DEFAULT_DATA, help="YOLO dataset YAML path.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Dataset split from data YAML.")
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300, dest="max_det")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="agri_val")
    parser.add_argument("--plots", type=parse_scalar, default=True)
    parser.add_argument("--save-json", action="store_true", dest="save_json")
    parser.add_argument("--save-txt", action="store_true", dest="save_txt")
    parser.add_argument("--exist-ok", action="store_true", dest="exist_ok")
    return parser


def main() -> None:
    args, unknown = build_parser().parse_known_args()
    val_args = vars(args)
    model_path = val_args.pop("model")
    val_args.update(parse_extra_args(unknown))
    val_args = {k: v for k, v in val_args.items() if v is not None}

    model = YOLO(model_path)
    metrics = model.val(**val_args)
    save_dir = getattr(metrics, "save_dir", None) or Path(val_args["project"]) / val_args["name"]
    if save_dir:
        print(f"Validation results saved to {save_dir}")
    results = getattr(metrics, "results_dict", {})
    if results:
        print("Validation summary:")
        for key in ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)", "fitness"):
            if key in results:
                print(f"  {key}: {results[key]:.6f}")


if __name__ == "__main__":
    main()
