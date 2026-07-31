#!/usr/bin/env python3
"""Train AgriUAV YOLOv8 model variants."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
DEFAULT_MODEL = "ultralytics/cfg/models/v8/yolov8n-p2-eca-simam-agri.yaml"
DEFAULT_DATA = "data/agri_uav.yaml"


def parse_scalar(value: str) -> Any:
    """Parse simple CLI scalar values for passthrough Ultralytics args."""
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
    """Accept extra args as key=value, --key value, or --flag."""
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
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model YAML or weights path.")
    parser.add_argument("--data", default=DEFAULT_DATA, help="Dataset YAML path.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default=None)
    parser.add_argument("--name", default="agri_p2_eca_simam")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--pretrained", type=parse_scalar, default=None)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--optimizer", default="SGD", help="Optimizer name passed to Ultralytics train.")
    parser.add_argument("--amp", type=parse_scalar, default=None, help="Enable automatic mixed precision training.")
    parser.add_argument("--exist-ok", action="store_true", dest="exist_ok")
    return parser


def main() -> None:
    args, unknown = build_parser().parse_known_args()
    train_args = vars(args)
    model_path = train_args.pop("model")
    train_args.update(parse_extra_args(unknown))
    train_args = {k: v for k, v in train_args.items() if v is not None}
    train_args.setdefault("amp", False)
    train_args.setdefault("pretrained", False)

    model = YOLO(model_path)
    model.train(**train_args)


if __name__ == "__main__":
    main()
