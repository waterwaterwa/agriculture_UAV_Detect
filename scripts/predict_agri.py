#!/usr/bin/env python3
"""Run AgriUAV YOLOv8 inference and save visualized predictions."""

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
    parser.add_argument("source", nargs="?", default=None, help="Image, directory, video, or glob source.")
    parser.add_argument("--source", dest="source_opt", default=None, help="Source path alternative.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Trained .pt weights path.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default=0)
    parser.add_argument("--max-det", type=int, default=300, dest="max_det")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="agri_predict")
    parser.add_argument("--save", type=parse_scalar, default=True)
    parser.add_argument("--save-txt", action="store_true", dest="save_txt")
    parser.add_argument("--save-conf", action="store_true", dest="save_conf")
    parser.add_argument("--show-labels", type=parse_scalar, default=True)
    parser.add_argument("--show-conf", type=parse_scalar, default=True)
    parser.add_argument("--line-width", type=int, default=None, dest="line_width")
    parser.add_argument("--exist-ok", action="store_true", dest="exist_ok")
    return parser


def main() -> None:
    args, unknown = build_parser().parse_known_args()
    predict_args = vars(args)
    source = predict_args.pop("source_opt") or predict_args.pop("source")
    if source is None:
        raise SystemExit("source is required, for example: scripts/predict_agri.py /path/to/images")
    model_path = predict_args.pop("model")
    predict_args["source"] = source
    predict_args.update(parse_extra_args(unknown))
    predict_args = {k: v for k, v in predict_args.items() if v is not None}

    model = YOLO(model_path)
    results = model.predict(**predict_args)
    save_dir = getattr(model.predictor, "save_dir", None)
    total_boxes = sum(len(result.boxes) for result in results)
    print(f"Predicted {len(results)} image(s), {total_boxes} detection(s).")
    if save_dir:
        print(f"Visualized predictions saved to {save_dir}")


if __name__ == "__main__":
    main()
