"""Tests for AgriUAV YOLOv8 model variants."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import pytest
import torch

from ultralytics.nn.modules import ECA, SimAM
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import get_flops

ROOT = Path(__file__).resolve().parents[1]
YOLOV8N = ROOT / "ultralytics/cfg/models/v8/yolov8.yaml"
YOLOV8N_P2 = ROOT / "ultralytics/cfg/models/v8/yolov8-p2.yaml"
YOLOV8N_P2_ECA = ROOT / "ultralytics/cfg/models/v8/yolov8n-p2-eca-agri.yaml"
YOLOV8N_P2_SIMAM = ROOT / "ultralytics/cfg/models/v8/yolov8n-p2-simam-agri.yaml"
YOLOV8N_P2_ECA_SIMAM = ROOT / "ultralytics/cfg/models/v8/yolov8n-p2-eca-simam-agri.yaml"


def count_parameters(model: torch.nn.Module) -> int:
    """Return the total model parameter count."""
    return sum(p.numel() for p in model.parameters())


def forward_features(model: DetectionModel, imgsz: int = 640) -> list[torch.Tensor]:
    """Run a random forward pass and return Detect feature tensors."""
    model.train()
    with torch.no_grad():
        outputs = model(torch.randn(1, 3, imgsz, imgsz))
    return outputs["feats"] if isinstance(outputs, dict) else outputs


def model_gflops(model: DetectionModel, imgsz: int = 640) -> float:
    """Return GFLOPs from the current Ultralytics FLOPs helper."""
    return float(get_flops(model, imgsz=imgsz))


def test_eca_shape_params_and_kernel_validation():
    """Validate ECA shape preservation, parameter count, and kernel checks."""
    x = torch.randn(2, 32, 20, 24)
    eca = ECA(3)
    y = eca(x)
    assert y.shape == x.shape
    assert count_parameters(eca) == 3
    with pytest.raises(ValueError):
        ECA(0)
    with pytest.raises(ValueError):
        ECA(4)


def test_simam_shape_params_and_lambda_validation():
    """Validate SimAM shape preservation, parameter count, and lambda checks."""
    x = torch.randn(2, 32, 20, 24)
    simam = SimAM(1e-4)
    y = simam(x)
    assert y.shape == x.shape
    assert count_parameters(simam) == 0
    assert torch.isfinite(y).all().item()
    with pytest.raises(ValueError):
        SimAM(0)


def test_yolov8n_p2_eca_model_build_stride_and_forward():
    """Build YOLOv8n-P2-ECA and validate four strides plus finite 640 forward outputs."""
    model = DetectionModel(str(YOLOV8N_P2_ECA), ch=3, nc=5, verbose=False)
    assert model.stride.tolist() == [4.0, 8.0, 16.0, 32.0]
    assert model.model[-1].nc == 5
    assert model.model[-1].nl == 4
    assert "ECA" in model.model[19].type

    feats = forward_features(model, imgsz=640)
    assert len(feats) == 4
    assert [tuple(x.shape[-2:]) for x in feats] == [(160, 160), (80, 80), (40, 40), (20, 20)]
    assert all(torch.isfinite(x).all().item() for x in feats)


def test_yolov8n_p2_simam_model_build_stride_and_forward():
    """Build YOLOv8n-P2-SimAM and validate P5 attention plus finite 640 forward outputs."""
    model = DetectionModel(str(YOLOV8N_P2_SIMAM), ch=3, nc=5, verbose=False)
    assert model.stride.tolist() == [4.0, 8.0, 16.0, 32.0]
    assert model.model[-1].nc == 5
    assert model.model[-1].nl == 4
    assert "SimAM" in model.model[28].type

    feats = forward_features(model, imgsz=640)
    assert len(feats) == 4
    assert [tuple(x.shape[-2:]) for x in feats] == [(160, 160), (80, 80), (40, 40), (20, 20)]
    assert all(torch.isfinite(x).all().item() for x in feats)


def test_yolov8n_p2_eca_simam_model_build_stride_and_forward():
    """Build YOLOv8n-P2-ECA-SimAM and validate P2 attention stack plus finite 640 forward outputs."""
    model = DetectionModel(str(YOLOV8N_P2_ECA_SIMAM), ch=3, nc=5, verbose=False)
    assert model.stride.tolist() == [4.0, 8.0, 16.0, 32.0]
    assert model.model[-1].nc == 5
    assert model.model[-1].nl == 4
    assert "ECA" in model.model[19].type
    assert "SimAM" in model.model[20].type

    feats = forward_features(model, imgsz=640)
    assert len(feats) == 4
    assert [tuple(x.shape[-2:]) for x in feats] == [(160, 160), (80, 80), (40, 40), (20, 20)]
    assert all(torch.isfinite(x).all().item() for x in feats)


def test_agri_model_variant_comparison():
    """Compare parameters, GFLOPs, and random forward timing for baseline variants."""
    variants = {
        "yolov8n": YOLOV8N,
        "yolov8n-p2": YOLOV8N_P2,
        "yolov8n-p2-eca": YOLOV8N_P2_ECA,
        "yolov8n-p2-simam": YOLOV8N_P2_SIMAM,
        "yolov8n-p2-eca-simam": YOLOV8N_P2_ECA_SIMAM,
    }
    results = {}
    for name, cfg in variants.items():
        model = DetectionModel(str(cfg), ch=3, nc=5, verbose=False)
        gflops = model_gflops(model, imgsz=640)
        start = perf_counter()
        feats = forward_features(model, imgsz=640)
        elapsed_ms = (perf_counter() - start) * 1000
        results[name] = {
            "params": count_parameters(model),
            "gflops": gflops,
            "time_ms": elapsed_ms,
            "strides": [float(x) for x in model.stride],
            "shapes": [tuple(x.shape) for x in feats],
        }
        assert all(torch.isfinite(x).all().item() for x in feats)

    assert results["yolov8n"]["strides"] == [8.0, 16.0, 32.0]
    assert results["yolov8n-p2"]["strides"] == [4.0, 8.0, 16.0, 32.0]
    assert results["yolov8n-p2-eca"]["strides"] == [4.0, 8.0, 16.0, 32.0]
    assert results["yolov8n-p2-simam"]["strides"] == [4.0, 8.0, 16.0, 32.0]
    assert results["yolov8n-p2-eca-simam"]["strides"] == [4.0, 8.0, 16.0, 32.0]
    assert results["yolov8n-p2-eca"]["params"] == results["yolov8n-p2"]["params"] + 3
    assert results["yolov8n-p2-simam"]["params"] == results["yolov8n-p2"]["params"]
    assert results["yolov8n-p2-eca-simam"]["params"] == results["yolov8n-p2"]["params"] + 3
    print(f"Agri model comparison: {results}")


