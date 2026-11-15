"""Laplacian and the Siamese contrastive loss for the speaker-count backend."""

from __future__ import annotations

import torch
import torch.nn as nn


def laplacian(w: torch.Tensor, kind: str) -> torch.Tensor:
    degree = w.sum(dim=1)
    if kind == "unnormalized":
        return torch.diag(degree) - w
    inv_sqrt = degree.clamp_min(1e-12).rsqrt()
    normalized = w * inv_sqrt.unsqueeze(1) * inv_sqrt.unsqueeze(0)
    return torch.eye(w.shape[0], dtype=w.dtype, device=w.device) - normalized


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, out1: torch.Tensor, out2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        euclidean = nn.functional.pairwise_distance(out1, out2)
        positive = euclidean.pow(2)
        negative = torch.clamp(self.margin - euclidean, min=0.0).pow(2)
        return torch.mean(label * positive + (1 - label) * negative)
