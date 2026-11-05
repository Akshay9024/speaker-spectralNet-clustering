"""Per-recording speaker count + labels via the SpectralNet backend.

Count comes from the eigengap of the Laplacian of the Siamese-learned affinity;
labels come from k-means on a spectral map fit at that count.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.sparse.csgraph import connected_components

from ..config import ClusteringConfig
from ..embedding import resolve_device
from .core import SpectralNetLoss, affinity_matrix, laplacian
from .models import SiameseNetModel, SpectralNetModel


@dataclass
class SpectralNetResult:
    k: int
    labels: np.ndarray
    n_segments: int
    mean_cosine: float
    single_speaker_shortcut: bool
    n_components: int
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


def _fit_spectral(
    x: torch.Tensor, w: torch.Tensor, k: int, cfg: ClusteringConfig, device: str, progress=None
) -> np.ndarray:
    sp = cfg.spectral
    net = SpectralNetModel(list(sp.hidden) + [k], input_dim=x.shape[1]).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=sp.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=sp.lr_decay, patience=sp.patience
    )
    criterion = SpectralNetLoss()
    for epoch in range(sp.epochs):
        net.eval()
        net(x, should_update_orth_weights=True)
        net.train()
        optimizer.zero_grad()
        y = net(x, should_update_orth_weights=False)
        loss = criterion(w, y)
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())
        if progress is not None:
            progress(epoch + 1, sp.epochs)
        if optimizer.param_groups[0]["lr"] <= sp.min_lr:
            break
    net.eval()
    with torch.no_grad():
        y = net(x, should_update_orth_weights=True)
    return y.detach().cpu().numpy()


def _kmeans(spectral: np.ndarray, k: int, cfg: ClusteringConfig) -> np.ndarray:
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
    emb = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
    emb = torch.nn.functional.normalize(emb, p=2.0, dim=-1)

    affinity = emb @ emb.T
    off_diagonal = affinity.sum() - torch.diagonal(affinity).sum()
    mean_cosine = float(off_diagonal / max(1, n * (n - 1))) if n > 1 else float("nan")

    if n < max(2, cfg.min_segments):
        return _trivial(n, mean_cosine, False)
    if cfg.single_speaker.enabled and mean_cosine > cfg.single_speaker.mean_cosine_threshold:
        return _trivial(n, mean_cosine, True)

    torch.manual_seed(cfg.seed)
    with torch.no_grad():
        z = siamese_net.forward_once(emb)

    w = affinity_matrix(
        z,
        n_neighbors=min(cfg.n_neighbors, n - 1),
        scale_k=cfg.scale_k,
        is_local=cfg.is_local_scale,
        device=torch.device(device),
    )

    eig_dtype = torch.float64 if cfg.dtype == "float64" else torch.float32
    lap = laplacian(w.to(eig_dtype), cfg.laplacian)
    eigenvalues = torch.linalg.eigvalsh(lap)
    gaps = eigenvalues[1:] - eigenvalues[:-1]
    gap_cap = min(cfg.k_max, n - 1)
    k = int(torch.argmax(gaps[:gap_cap])) + 1

    adjacency = (w > 0.0).cpu().numpy()
    n_components = int(connected_components(adjacency, directed=False, return_labels=False))

    if k <= 1:
        labels = np.zeros(n, dtype=np.int64)
    else:
        spectral = _fit_spectral(emb, w.to(torch.float32), k, cfg, device, progress=progress)
        labels = _kmeans(spectral, k, cfg)

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
