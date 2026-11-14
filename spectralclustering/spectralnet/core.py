"""Affinity, Laplacian and losses for the SpectralNet speaker-count backend.

Affinity construction (Gaussian kernel over a kNN graph, local scaling) and the
spectral loss are vendored from shaham-lab/SpectralNet; the Laplacian matches the
`unnormalized | sym_norm` choice already used elsewhere in this project.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors


def nearest_neighbors(x: torch.Tensor, k: int) -> tuple[np.ndarray, np.ndarray]:
    x_np = x.detach().cpu().numpy()
    k = min(k, len(x_np))
    nbrs = NearestNeighbors(n_neighbors=k).fit(x_np)
    distances, indices = nbrs.kneighbors(x_np)
    return distances, indices


def compute_scale(distances: np.ndarray, k: int, is_local: bool) -> np.ndarray | float:
    if is_local:
        return np.median(distances, axis=1)
    return float(np.median(distances[:, min(k, distances.shape[1]) - 1]))


def gaussian_kernel(
    d: torch.Tensor, scale, indices: np.ndarray, device: torch.device, is_local: bool
) -> torch.Tensor:
    if is_local:
        scale_t = torch.tensor(scale, dtype=d.dtype, device=device).clamp_min(1e-7)
        w = torch.exp(-d.pow(2) / (scale_t.unsqueeze(1) * scale_t.unsqueeze(0)))
    else:
        w = torch.exp(-d.pow(2) / (float(scale) ** 2))
    n, k = indices.shape
    mask = torch.zeros((n, n), dtype=d.dtype, device=device)
    rows = torch.arange(n, device=device).repeat_interleave(k)
    cols = torch.as_tensor(indices, device=device).reshape(-1)
    mask[rows, cols] = 1.0
    w = w * mask
    return (w + w.T) / 2.0


def affinity_matrix(
    x: torch.Tensor, n_neighbors: int, scale_k: int, is_local: bool, device: torch.device
) -> torch.Tensor:
    """Symmetric Gaussian-kernel affinity over the kNN graph of ``x``."""
    dx = torch.cdist(x, x)
    distances, indices = nearest_neighbors(x, k=n_neighbors + 1)
    scale = compute_scale(distances, k=scale_k, is_local=is_local)
    return gaussian_kernel(dx, scale, indices, device=device, is_local=is_local)


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


class SpectralNetLoss(nn.Module):
    def forward(self, w: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m = y.size(0)
        dy = torch.cdist(y, y)
        return torch.sum(w * dy.pow(2)) / (2 * m)
