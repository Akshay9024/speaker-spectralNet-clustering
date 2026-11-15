"""Siamese encoder, trained with a contrastive loss to learn the affinity metric."""

from __future__ import annotations

import torch
import torch.nn as nn


class SiameseNetModel(nn.Module):
    """Shared-weight encoder trained with a contrastive loss to learn the affinity metric."""

    def __init__(self, hiddens: list[int], input_dim: int):
        super().__init__()
        self.layers = nn.ModuleList()
        current = input_dim
        for width in hiddens:
            self.layers.append(nn.Sequential(nn.Linear(current, width), nn.ReLU()))
            current = width

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_once(x1), self.forward_once(x2)
