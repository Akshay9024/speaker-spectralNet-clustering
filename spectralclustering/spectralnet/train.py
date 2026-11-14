"""Global Siamese training for the SpectralNet affinity metric.

Supervised: pairs are built from reference speaker labels (same speaker -> positive,
different speaker -> negative) within each recording, then pooled across recordings.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from ..config import ClusteringConfig
from .core import ContrastiveLoss
from .models import SiameseNetModel


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def segment_labels(segments: list, turns: list) -> list[str | None]:
    """Speaker of each segment: the reference turn with the most overlap on its window."""
    labels: list[str | None] = []
    for s, e, _ in segments:
        best, best_overlap = None, 0.0
        for t in turns:
            overlap = min(e, t.end) - max(s, t.start)
            if overlap > best_overlap:
                best_overlap, best = overlap, t.speaker
        labels.append(best)
    return labels


def _supervised_pairs(labels: list[str | None], n_per: int, offset: int, rng) -> list[tuple[int, int, int]]:
    """Same/different-speaker index pairs within one recording (indices shifted by offset)."""
    by_speaker: dict[str, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        if lab is not None:
            by_speaker[lab].append(i)
    if len(by_speaker) < 2:  # need >=2 speakers to form negatives
        return []
    valid = [i for i, lab in enumerate(labels) if lab is not None]
    pairs: list[tuple[int, int, int]] = []
    for i in valid:
        same = [j for j in by_speaker[labels[i]] if j != i]
        diff = [j for j in valid if labels[j] != labels[i]]
        for j in rng.choice(same, min(n_per, len(same)), replace=False) if same else []:
            pairs.append((offset + i, offset + int(j), 1))
        for j in rng.choice(diff, min(n_per, len(diff)), replace=False) if diff else []:
            pairs.append((offset + i, offset + int(j), 0))
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


def _fit_siamese(x: torch.Tensor, pairs: np.ndarray, cfg: ClusteringConfig, device: str) -> SiameseNetModel:
    sc = cfg.siamese
    net = SiameseNetModel(sc.hiddens, input_dim=x.shape[1]).to(device)
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
    return net


def train_siamese(records: list[tuple[np.ndarray, list]], cfg: ClusteringConfig, device: str,
                  out_path: Path) -> SiameseNetModel:
    """Train the Siamese on supervised same/different-speaker pairs pooled across recordings.

    records: list of (embeddings[N,D], per-segment speaker labels[N]) for each recording.
    """
    seed_everything(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    mats, pair_blocks, offset = [], [], 0
    for emb, labels in records:
        if emb.shape[0] == 0:
            continue
        pair_blocks.extend(_supervised_pairs(labels, cfg.siamese.n_neighbors, offset, rng))
        mats.append(emb)
        offset += emb.shape[0]
    if not pair_blocks:
        raise RuntimeError("no supervised pairs built (need recordings with >=2 labelled speakers)")

    x = torch.as_tensor(np.vstack(mats), dtype=torch.float32, device=device)
    pairs = np.asarray(pair_blocks, dtype=np.int64)
    n_pos = int((pairs[:, 2] == 1).sum())
    print(f"supervised pairs: {len(pairs):,} ({n_pos:,} positive / {len(pairs) - n_pos:,} negative) "
          f"over {x.shape[0]:,} segments", flush=True)

    net = _fit_siamese(x, pairs, cfg, device)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "hiddens": cfg.siamese.hiddens,
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
