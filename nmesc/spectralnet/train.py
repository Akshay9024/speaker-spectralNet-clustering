"""Global Siamese training for the SpectralNet affinity metric (self-supervised kNN pairs)."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset, random_split

from ..config import ClusteringConfig
from .core import ContrastiveLoss
from .models import SiameseNetModel


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _knn_pairs(x: np.ndarray, n_neighbors: int) -> np.ndarray:
    """(anchor, other, label) index triples: n_neighbors positives and n_neighbors negatives each."""
    n = len(x)
    all_idx = np.arange(n)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm="ball_tree").fit(x)
    _, neighbors = nbrs.kneighbors(x)

    pairs = np.empty((n * 2 * n_neighbors, 3), dtype=np.int64)
    cursor = 0
    for i in range(n):
        pos = neighbors[i, 1 : n_neighbors + 1]
        non = np.setdiff1d(all_idx, neighbors[i], assume_unique=False)
        neg = np.random.choice(non, n_neighbors, replace=len(non) < n_neighbors)
        for j in pos:
            pairs[cursor] = (i, j, 1)
            cursor += 1
        for j in neg:
            pairs[cursor] = (i, j, 0)
            cursor += 1
    return pairs


class _PairDataset(Dataset):
    def __init__(self, x: torch.Tensor, pairs: np.ndarray):
        self.x = x
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b, label = self.pairs[idx]
        return self.x[a], self.x[b], torch.tensor(float(label))


def train_siamese(
    embeddings: np.ndarray, cfg: ClusteringConfig, device: str, out_path: Path
) -> SiameseNetModel:
    """Train the Siamese metric net on pooled embeddings and save its weights."""
    seed_everything(cfg.seed)
    sc = cfg.siamese
    x = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
    net = SiameseNetModel(sc.hiddens, input_dim=x.shape[1]).to(device)

    pairs = _knn_pairs(embeddings, sc.n_neighbors)
    dataset = _PairDataset(x, pairs)
    n_train = int(0.9 * len(dataset))
    train_set, valid_set = random_split(dataset, [n_train, len(dataset) - n_train])
    train_loader = DataLoader(train_set, batch_size=sc.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=sc.batch_size, shuffle=False)

    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=sc.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=sc.lr_decay, patience=sc.patience
    )

    for epoch in range(sc.epochs):
        net.train()
        for x1, x2, label in train_loader:
            optimizer.zero_grad()
            out1, out2 = net(x1, x2)
            loss = criterion(out1, out2, label.to(device))
            loss.backward()
            optimizer.step()

        net.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for x1, x2, label in valid_loader:
                out1, out2 = net(x1, x2)
                valid_loss += criterion(out1, out2, label.to(device)).item()
        valid_loss /= max(1, len(valid_loader))
        scheduler.step(valid_loss)
        lr = optimizer.param_groups[0]["lr"]
        print(f"  siamese epoch {epoch + 1}/{sc.epochs}  valid_loss={valid_loss:.6f}  lr={lr:.2e}", flush=True)
        if lr <= sc.min_lr:
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "hiddens": sc.hiddens,
                "input_dim": x.shape[1], "seed": cfg.seed}, out_path)
    return net


_siamese_cache: dict[tuple[str, str], SiameseNetModel] = {}


def load_siamese(cfg: ClusteringConfig, root: Path, device: str) -> SiameseNetModel:
    path = root / cfg.siamese_weights
    key = (str(path), device)
    if key in _siamese_cache:
        return _siamese_cache[key]
    if not path.exists():
        raise FileNotFoundError(
            f"Siamese weights not found at {path}; run `run.py train` first to fit the metric net."
        )
    ckpt = torch.load(path, map_location=device)
    net = SiameseNetModel(ckpt["hiddens"], input_dim=ckpt["input_dim"]).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    _siamese_cache[key] = net
    return net
