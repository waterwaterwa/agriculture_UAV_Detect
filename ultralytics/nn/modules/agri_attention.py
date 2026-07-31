# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Agricultural UAV attention modules."""

from __future__ import annotations

import torch
import torch.nn as nn


class ECA(nn.Module):
    """Efficient Channel Attention that preserves BCHW feature dimensions."""

    def __init__(self, kernel_size: int = 3) -> None:
        """Initialize ECA with a positive odd Conv1d kernel size."""
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention to a BCHW tensor without changing its shape."""
        y = self.pool(x).squeeze(-1).transpose(-1, -2)
        y = self.act(self.conv(y)).transpose(-1, -2).unsqueeze(-1)
        return x * y


class SimAM(nn.Module):
    """Parameter-free SimAM attention that preserves BCHW feature dimensions."""

    def __init__(self, e_lambda: float = 1e-4) -> None:
        """Initialize SimAM with a positive numerical stability term."""
        super().__init__()
        if e_lambda <= 0:
            raise ValueError("e_lambda must be positive")
        self.e_lambda = float(e_lambda)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SimAM attention to a BCHW tensor without changing its shape."""
        mean = x.mean(dim=(2, 3), keepdim=True)
        deviation = (x - mean).pow(2)
        n = max(x.shape[2] * x.shape[3] - 1, 1)
        energy = deviation / (4 * (deviation.sum(dim=(2, 3), keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.act(energy)

