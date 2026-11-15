#!/usr/bin/env python3
"""SpectralNet speaker-count pipeline: stage runner, Siamese trainer and inspector."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from spectralclustering import pipeline, report  # noqa: E402
from spectralclustering.config import load_config  # noqa: E402
from spectralclustering.embedding import EcapaEmbedder, resolve_device  # noqa: E402


def _parse_override(text: str):
    key, _, raw = text.partition("=")
    if not _:
        raise argparse.ArgumentTypeError(f"--set expects key=value, got {text!r}")
    try:
        return key, json.loads(raw)
    except json.JSONDecodeError:
        return key, raw


def _select(cfg, args) -> list[str]:
    recs = pipeline.recording_ids(cfg)
    if args.only:
        missing = set(args.only) - set(recs)
        if missing:
            raise SystemExit(f"unknown recording ids: {sorted(missing)}")
        recs = [r for r in recs if r in set(args.only)]
    if args.limit:
        recs = recs[: args.limit]
    return recs


def cmd_run(args) -> None:
    from tqdm import tqdm

    cfg = load_config(args.config, dict(args.set or []))
    stages = args.stages.split(",")
    unknown = set(stages) - set(pipeline.STAGES)
    if unknown:
        raise SystemExit(f"unknown stages {sorted(unknown)}; choose from {pipeline.STAGES}")

    recs = _select(cfg, args)
    print(f"dataset={cfg.dataset.name} recordings={len(recs)} stages={stages}", flush=True)

    if "segment" in stages:
        vad = None
        if cfg.vad.source == "pyannote":
            from spectralclustering.vad import SegmentationVad

            vad = SegmentationVad(
                model=cfg.vad.model,
                device=cfg.vad.device,
                min_duration_on=cfg.vad.min_duration_on,
                min_duration_off=cfg.vad.min_duration_off,
            )
            print(f"  VAD: {cfg.vad.model} on {cfg.vad.device}", flush=True)
        total = 0
        for rec in tqdm(recs, desc="segment"):
            total += len(pipeline.stage_segment(cfg, rec, force=args.force, vad=vad)["segments"])
        print(f"  segments: {total:,}", flush=True)

    if "embed" in stages:
        embedder = EcapaEmbedder(cfg.embedding, Path(cfg.root))
        print(f"  embedding device: {embedder.device}", flush=True)
        for rec in tqdm(recs, desc="embed"):
            pipeline.stage_embed(cfg, rec, embedder, force=args.force)

    if "cluster" in stages:
        print(f"  clustering device: {resolve_device(cfg.clustering.device)}", flush=True)
        outer = tqdm(recs, desc="cluster")
        for rec in outer:
            bar = {"inner": None}

            def progress(done, total, bar=bar, rec=rec):
                if bar["inner"] is None:
                    bar["inner"] = tqdm(total=total, desc=f"  {rec} spectral", leave=False)
                bar["inner"].update(done - bar["inner"].n)

            result = pipeline.stage_cluster(cfg, rec, force=args.force, progress=progress)
            if bar["inner"] is not None:
                bar["inner"].close()
            outer.set_postfix(k=int(result["k_pred"]))

    if "evaluate" in stages:
        frame = pipeline.stage_evaluate(cfg, recs)
        path = report.write(cfg, frame)
        print(f"\n{report.build(cfg, frame)}", flush=True)
        print(f"wrote {cfg.results / 'predictions.csv'} and {path}", flush=True)


def cmd_report(args) -> None:
    import pandas as pd

    cfg = load_config(args.config, dict(args.set or []))
    csv = cfg.results / "predictions.csv"
    if not csv.exists():
        raise SystemExit(f"{csv} not found; run the evaluate stage first")
    frame = pd.read_csv(csv)
    print(report.build(cfg, frame))
    print(f"\nwrote {report.write(cfg, frame)}", flush=True)


def cmd_train(args) -> None:
    cfgs = [load_config(c, dict(args.set or [])) for c in args.config]
    out = pipeline.train_siamese(cfgs)
    print(f"wrote {out}", flush=True)


def cmd_meet(args) -> None:
    from spectralclustering import meet

    meet.run(args.meet_config, token=args.token)


def cmd_inspect(args) -> None:
    cfg = load_config(args.config, dict(args.set or []))
    rec = args.recording

    payload = pipeline.stage_segment(cfg, rec)
    print(f"=== {rec} ===")
    print(f"duration        : {payload['duration']:.2f} s")
    print(f"reference K     : {payload['k_true']}  ({payload['n_reference_turns']} turns)")
    print(f"oracle speech   : {payload['speech_seconds']:.2f} s "
          f"({100 * payload['speech_seconds'] / payload['duration']:.1f}%)")
    print(f"speech intervals: {len(payload['intervals'])}")
    print(f"segments        : {len(payload['segments'])}")

    emb_path = cfg.work / "embeddings" / f"{rec}.npy"
    if emb_path.exists():
        embeddings = np.load(emb_path)
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"embeddings      : {embeddings.shape} dtype={embeddings.dtype} "
              f"L2 norm in [{norms.min():.4f}, {norms.max():.4f}]")

    npz_path = pipeline.clustering_artifact(cfg, rec)
    if not npz_path.exists():
        print("clustering      : not run yet")
        return

    with np.load(npz_path) as data:
        k_pred = int(data["k_pred"])
        print(f"K̂               : {k_pred}   (true {payload['k_true']})")
        if "n_components" in data.files:
            print(f"affinity comps  : {int(data['n_components'])}  (connected components of the Siamese affinity graph)")
        gap_cap = min(cfg.clustering.k_max, int(data["n_segments"]) - 1)
        if k_pred > 1 and k_pred == gap_cap:
            print(f"K_max saturation: YES (K̂ == min(k_max, N-1) == {gap_cap}) -- estimator hit the ceiling")
        print(f"mean off-diag cos: {float(data['mean_cosine']):.4f}")
        print(f"clustering time : {float(data['elapsed_sec']):.2f} s")

        eigenvalues, eigengaps = data["eigenvalues"], data["eigengaps"]
        if len(eigenvalues):
            top = min(len(eigengaps), cfg.clustering.k_max + 3)
            print(f"\nsmallest eigenvalues and gaps of the Siamese-affinity Laplacian (k_max={cfg.clustering.k_max}):")
            print(f"  {'i':>4} {'lambda_i':>14} {'gap_i = l(i+1)-l(i)':>22}")
            for i in range(top):
                mark = "  <- K̂" if i + 1 == int(data["k_pred"]) else ""
                capped = "" if i < cfg.clustering.k_max else "  (beyond k_max)"
                print(f"  {i + 1:>4} {eigenvalues[i]:>14.6f} {eigengaps[i]:>22.6f}{mark}{capped}")

        labels = data["labels"]
        if len(labels):
            uniq, counts = np.unique(labels, return_counts=True)
            print(f"\nsegment labels  : {dict(zip(uniq.tolist(), counts.tolist()))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--config", required=True)
        p.add_argument("--set", action="append", type=_parse_override, metavar="KEY=VALUE",
                       help="dotted config override, e.g. --set clustering.laplacian=sym_norm")

    run = sub.add_parser("run", help="execute pipeline stages")
    common(run)
    run.add_argument("--stages", default=",".join(pipeline.STAGES))
    run.add_argument("--only", nargs="+", metavar="REC_ID")
    run.add_argument("--limit", type=int)
    run.add_argument("--force", action="store_true", help="recompute even if artifacts exist")
    run.set_defaults(func=cmd_run)

    train = sub.add_parser("train", help="train the global Siamese affinity net on pooled dev embeddings")
    train.add_argument("--config", required=True, action="append", metavar="CONFIG",
                       help="dev config(s); repeat to pool across datasets, e.g. --config a.yaml --config b.yaml")
    train.add_argument("--set", action="append", type=_parse_override, metavar="KEY=VALUE")
    train.set_defaults(func=cmd_train)

    inspect = sub.add_parser("inspect", help="dump intermediates for one recording")
    common(inspect)
    inspect.add_argument("recording")
    inspect.set_defaults(func=cmd_inspect)

    rep = sub.add_parser("report", help="rebuild summary.md from predictions.csv")
    common(rep)
    rep.set_defaults(func=cmd_report)

    meet = sub.add_parser("meet", help="render results_meet plots and relative-error accuracy")
    meet.add_argument("--meet-config", default="results_meet/config.yaml")
    meet.add_argument("--token", default=None, help="HF token for pyannote/segmentation-3.0")
    meet.set_defaults(func=cmd_meet)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
