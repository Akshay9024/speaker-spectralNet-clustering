"""SpectralNet models, vendored and trimmed from shaham-lab/SpectralNet (ICLR 2018)."""

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


class SpectralNetModel(nn.Module):
    """Maps inputs to an orthonormalized spectral embedding; last layer width = k."""

    def __init__(self, hiddens: list[int], input_dim: int):
        super().__init__()
        self.layers = nn.ModuleList()
        self.orthonorm_weights: torch.Tensor | None = None
        current = input_dim
        for i, width in enumerate(hiddens):
            if i == len(hiddens) - 1:
                self.layers.append(nn.Sequential(nn.Linear(current, width), nn.Tanh()))
            else:
                self.layers.append(nn.Sequential(nn.Linear(current, width), nn.LeakyReLU()))
                current = width

    def _make_orthonorm_weights(self, y: torch.Tensor) -> torch.Tensor:
        m = y.shape[0]
        _, r = torch.linalg.qr(y)
        signs = torch.sign(torch.diag(r))
        r = signs.unsqueeze(1) * r
        return m**0.5 * torch.inverse(r)

    def forward(self, x: torch.Tensor, should_update_orth_weights: bool = True) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        y_tilde = x
        if should_update_orth_weights or self.orthonorm_weights is None:
            self.orthonorm_weights = self._make_orthonorm_weights(y_tilde)
        return y_tilde @ self.orthonorm_weights
