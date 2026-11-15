"""Per-recording speaker count via NME-SC (Park et al. 2020, arXiv:2003.02405, Algorithm 1).

Affinity is cosine similarity over the Siamese-learned embedding (or raw ECAPA when
``clustering.use_siamese`` is False). The count is the eigengap of the Laplacian at the
auto-tuned binarisation p* = argmin_p (p / normalised-max-eigengap). Labels come from
k-means on the Laplacian eigenvectors at p* (classic spectral clustering, no deep net).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.sparse.csgraph import connected_components

from ..config import ClusteringConfig
from ..embedding import resolve_device
from .core import laplacian
from .models import SiameseNetModel


@dataclass
class SpectralNetResult:
    k: int
    labels: np.ndarray
    n_segments: int
    mean_cosine: float
    single_speaker_shortcut: bool
    n_components: int  # connected components of the binarised affinity at p* (0 on trivial paths)
    eigenvalues: np.ndarray
    eigengaps: np.ndarray


def _trivial(n: int, mean_cosine: float, shortcut: bool) -> SpectralNetResult:
    empty = np.zeros(0, dtype=np.float64)
    return SpectralNetResult(
        k=1,
        labels=np.zeros(n, dtype=np.int64),
        n_segments=n,
        mean_cosine=mean_cosine,
        single_speaker_shortcut=shortcut,
        n_components=0,
        eigenvalues=empty,
        eigengaps=empty,
    )


def _p_grid(n: int, cfg: ClusteringConfig) -> np.ndarray:
    p_max = int(math.floor(n * cfg.p_max_ratio))
    if cfg.p_max_abs is not None:
        p_max = min(p_max, cfg.p_max_abs)
    p_max = max(cfg.p_min, min(p_max, n - 1))
    return np.arange(cfg.p_min, p_max + 1, max(1, cfg.p_stride), dtype=np.int64)


def _kmeans(spectral: np.ndarray, k: int, cfg: ClusteringConfig) -> np.ndarray:
    if k <= 1:
        return np.zeros(spectral.shape[0], dtype=np.int64)
    from sklearn.cluster import KMeans

    model = KMeans(n_clusters=k, n_init=cfg.kmeans_n_init, random_state=cfg.seed)
    return model.fit_predict(spectral).astype(np.int64)


def estimate(
    embeddings: np.ndarray,
    cfg: ClusteringConfig,
    siamese_net: SiameseNetModel,
    progress=None,
) -> SpectralNetResult:
    n = int(embeddings.shape[0])
    if n == 0:
        return _trivial(0, float("nan"), False)

    device = resolve_device(cfg.device)
    dtype = torch.float64 if cfg.dtype == "float64" else torch.float32
    emb = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
    emb = torch.nn.functional.normalize(emb, p=2.0, dim=-1)

    affinity = emb @ emb.T
    off_diagonal = affinity.sum() - torch.diagonal(affinity).sum()
    mean_cosine = float(off_diagonal / max(1, n * (n - 1))) if n > 1 else float("nan")

    if n < max(2, cfg.min_segments):
        return _trivial(n, mean_cosine, False)
    if cfg.single_speaker.enabled and mean_cosine > cfg.single_speaker.mean_cosine_threshold:
        return _trivial(n, mean_cosine, True)

    if cfg.use_siamese:
        torch.manual_seed(cfg.seed)
        with torch.no_grad():
            z = siamese_net.forward_once(emb)
        z = torch.nn.functional.normalize(z, p=2.0, dim=-1)
        affinity = z @ z.T

    affinity = affinity.to(dtype)
    if cfg.zero_diagonal:
        affinity.fill_diagonal_(0.0)

    grid = _p_grid(n, cfg)
    _, order = torch.topk(affinity, k=int(grid[-1]), dim=1)

    gap_cap = min(cfg.k_max, n - 1)
    binary = torch.zeros((n, n), dtype=dtype, device=device)
    r_values = np.empty(len(grid), dtype=np.float64)
    for i, p in enumerate(grid):
        binary.zero_()
        binary.scatter_(1, order[:, : int(p)], 1.0)
        ev = torch.linalg.eigvalsh(laplacian(0.5 * (binary + binary.T), cfg.laplacian))
        gaps = (ev[1:] - ev[:-1])[:gap_cap]
        g = float(gaps.max()) / (float(ev[-1]) + cfg.eps)  # normalised max eigengap
        r_values[i] = float(p) / (g + cfg.eps)
        if progress is not None:
            progress(i + 1, len(grid))

    best_p = int(grid[int(np.argmin(r_values))])
    binary.zero_()
    binary.scatter_(1, order[:, :best_p], 1.0)
    symmetric = 0.5 * (binary + binary.T)
    adjacency = (symmetric > 0.0).cpu().numpy()
    n_components = int(connected_components(adjacency, directed=False, return_labels=False))

    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian(symmetric, cfg.laplacian))
    gaps = eigenvalues[1:] - eigenvalues[:-1]
    k = int(torch.argmax(gaps[:gap_cap])) + 1

    spectral = eigenvectors[:, :k]
    if cfg.row_normalize_spectral_embedding:
        spectral = torch.nn.functional.normalize(spectral, p=2.0, dim=-1)
    labels = _kmeans(spectral.double().cpu().numpy(), k, cfg)

    return SpectralNetResult(
        k=k,
        labels=labels,
        n_segments=n,
        mean_cosine=mean_cosine,
        single_speaker_shortcut=False,
        n_components=n_components,
        eigenvalues=eigenvalues.double().cpu().numpy(),
        eigengaps=gaps.double().cpu().numpy(),
    )
