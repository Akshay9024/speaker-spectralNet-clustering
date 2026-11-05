from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import audio, oracle_vad, rttm, spectralnet
from .config import Config
from .embedding import EcapaEmbedder, resolve_device
from .segmentation import Segment, segment_intervals

STAGES = ("segment", "embed", "cluster", "evaluate")


def train_siamese(cfgs: list[Config]) -> Path:
    """Pool cached embeddings across the given (dev) configs and fit the global Siamese net."""
    pools = []
    for cfg in cfgs:
        for rec in recording_ids(cfg):
            emb_path = _artifact(cfg, "embeddings", rec, ".npy")
            if not emb_path.exists():
                raise FileNotFoundError(
                    f"{emb_path} missing; run `run.py run --stages segment,embed --config ...` on "
                    f"{cfg.dataset.name} before training the Siamese net"
                )
            emb = np.load(emb_path)
            if emb.shape[0]:
                pools.append(emb)
    if not pools:
        raise RuntimeError("no embeddings found to train the Siamese net")
    pooled = np.vstack(pools)
    base = cfgs[0]
    out = Path(base.root) / base.clustering.siamese_weights
    print(f"training Siamese on {pooled.shape[0]:,} pooled embeddings -> {out}", flush=True)
    spectralnet.train_siamese(pooled, base.clustering, resolve_device(base.clustering.device), out)
    return out


def recording_ids(cfg: Config) -> list[str]:
    rttm_dir = cfg.path(cfg.dataset.rttm_dir)
    if not rttm_dir.is_dir():
        raise FileNotFoundError(f"rttm_dir not found: {rttm_dir}")
    return sorted(p.stem for p in rttm_dir.glob("*.rttm"))


def reference_rttm_path(cfg: Config, rec: str) -> Path:
    return cfg.path(cfg.dataset.rttm_dir, f"{rec}.rttm")


def audio_path(cfg: Config, rec: str) -> Path:
    return cfg.path(cfg.dataset.audio_dir, cfg.dataset.audio_template.format(rec=rec))


def _artifact(cfg: Config, stage: str, rec: str, suffix: str) -> Path:
    path = cfg.work / stage / f"{rec}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------- stage: segment


def stage_segment(cfg: Config, rec: str, force: bool = False, vad=None) -> dict:
    out = _artifact(cfg, "segments", rec, ".json")
    if out.exists() and not force:
        return json.loads(out.read_text())

    wav = audio_path(cfg, rec)
    if not wav.exists():
        raise FileNotFoundError(f"audio not found for {rec}: {wav}")
    duration = audio.probe_duration(wav)

    turns = rttm.read_rttm(reference_rttm_path(cfg, rec))
    if cfg.vad.source == "pyannote":
        if vad is None:
            raise RuntimeError("vad.source=pyannote but no VAD instance passed to stage_segment")
        intervals = vad.intervals(wav)
    else:
        intervals = oracle_vad.speech_intervals(turns, duration, cfg.segmentation.merge_gap_sec)
    segments = segment_intervals(intervals, cfg.segmentation)

    payload = {
        "recording_id": rec,
        "duration": duration,
        "k_true": rttm.speaker_count(turns),
        "n_reference_turns": len(turns),
        "speech_seconds": oracle_vad.total_duration(intervals),
        "intervals": [[s, e] for s, e in intervals],
        "segments": [[s.start, s.end, s.interval_index] for s in segments],
        "segmentation": asdict(cfg.segmentation),
        "vad_source": cfg.vad.source,
    }
    out.write_text(json.dumps(payload))
    return payload


def load_segments(payload: dict) -> list[Segment]:
    return [Segment(s, e, int(i)) for s, e, i in payload["segments"]]


# ---------------------------------------------------------------- stage: embed


def stage_embed(cfg: Config, rec: str, embedder: EcapaEmbedder, force: bool = False) -> np.ndarray:
    out = _artifact(cfg, "embeddings", rec, ".npy")
    if out.exists() and not force:
        return np.load(out)

    payload = stage_segment(cfg, rec)
    segments = load_segments(payload)
    if not segments:
        embeddings = np.zeros((0, 192), dtype=np.float32)
    else:
        waveform = audio.load_mono(audio_path(cfg, rec), cfg.embedding.sample_rate)
        embeddings = embedder.encode(waveform, segments, cfg.segmentation.window_sec)
    np.save(out, embeddings)
    return embeddings


# ---------------------------------------------------------------- stage: cluster


def stage_cluster(cfg: Config, rec: str, force: bool = False, progress=None) -> dict:
    out = _artifact(cfg, "clustering", rec, ".npz")
    if out.exists() and not force:
        with np.load(out, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}

    embeddings_path = _artifact(cfg, "embeddings", rec, ".npy")
    if not embeddings_path.exists():
        raise FileNotFoundError(f"{embeddings_path} missing; run the embed stage for {rec} first")
    embeddings = np.load(embeddings_path)
    device = resolve_device(cfg.clustering.device)
    siamese_net = spectralnet.load_siamese(cfg.clustering, Path(cfg.root), device)
    started = time.perf_counter()
    result = spectralnet.estimate(embeddings, cfg.clustering, siamese_net, progress=progress)
    elapsed = time.perf_counter() - started

    payload = {
        "k_pred": np.int64(result.k),
        "n_segments": np.int64(result.n_segments),
        "mean_cosine": np.float64(result.mean_cosine),
        "single_speaker_shortcut": np.bool_(result.single_speaker_shortcut),
        "n_components": np.int64(result.n_components),
        "elapsed_sec": np.float64(elapsed),
        "eigenvalues": result.eigenvalues,
        "eigengaps": result.eigengaps,
        "labels": result.labels,
    }
    np.savez_compressed(out, **payload)
    return payload


# ---------------------------------------------------------------- stage: evaluate


def stage_evaluate(cfg: Config, recs: list[str]) -> "object":
    import pandas as pd

    rows = []
    for rec in recs:
        payload = stage_segment(cfg, rec)
        cluster = stage_cluster(cfg, rec)
        k_pred = int(cluster["k_pred"])
        n_segments = int(cluster["n_segments"])
        gap_cap = min(cfg.clustering.k_max, n_segments - 1)
        row = {
            "recording_id": rec,
            "k_true": int(payload["k_true"]),
            "k_pred": k_pred,
            "n_segments": n_segments,
            "n_components": int(cluster["n_components"]) if "n_components" in cluster else -1,
            "saturated": bool(k_pred > 1 and k_pred == gap_cap),
            "mean_cosine": float(cluster["mean_cosine"]),
            "single_speaker_shortcut": bool(cluster["single_speaker_shortcut"]),
            "duration_sec": float(payload["duration"]),
            "speech_sec": float(payload["speech_seconds"]),
            "cluster_sec": float(cluster["elapsed_sec"]),
        }
        rows.append(row)

    frame = pd.DataFrame(rows)
    cfg.results.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cfg.results / "predictions.csv", index=False)
    return frame
