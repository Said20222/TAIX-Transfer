"""Evaluate the chosen checkpoints on the TAIX-Ray TEST split, once, and compare them.

For each run directory, loads best.pt (picked on validation by train.py), predicts on the test split,
and reports AUROC per label (8), per finding (5) and their means. Uncertainty comes from a
patient-level bootstrap: patients are resampled with replacement, because many patients have
several radiographs. With two runs, the same resamples give a paired 95% CI and a two-sided
bootstrap p-value for the AUROC difference (second run minus first).

The test split is for the final comparison only. The script refuses to overwrite an existing result
unless --force is given; use --split val for a dry run.

Usage:
    python scripts/evaluate_test.py outputs/runs/densenet121_s224_... outputs/runs/eva_x_tiny_s224_...
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from common import (FINDINGS, LABEL_COLS, MODEL_DEFAULTS, TaixDataset, build_model, finding_arrays, load_split,
                    metrics, predict)

NAMES = {"densenet121": "DenseNet-121 (ImageNet)", "eva_x_tiny": "EVA-X-Tiny (CXR MIM)"}


def fast_auroc(y: np.ndarray, p: np.ndarray) -> float:
    """Mann-Whitney AUROC with tie-averaged ranks (same value as sklearn's roc_auc_score)."""
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(p)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def all_scores(probs8, targets8):
    """26 scores for one set of predictions: AUROC for 8 labels + 5 findings, then AUPRC for the same 13."""
    p5, t5 = finding_arrays(probs8, targets8)
    P, T = np.concatenate([probs8, p5], 1), np.concatenate([targets8, t5], 1)
    roc = [fast_auroc(T[:, j], P[:, j]) for j in range(P.shape[1])]
    pr = [average_precision_score(T[:, j], P[:, j]) if 0 < T[:, j].sum() < len(T) else np.nan
          for j in range(P.shape[1])]
    return np.array(roc + pr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run directories containing best.pt")
    ap.add_argument("--data-dir", default=os.environ.get("TAIX_DATA_DIR", "data/TAIX-Ray"))
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--out-dir", default=None, help="default: outputs/<split>_eval")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="overwrite an existing result")
    args = ap.parse_args()

    out_dir = Path(args.out_dir or f"outputs/{args.split}_eval").resolve()
    result_path = out_dir / "results.json"
    if result_path.exists() and not args.force:
        raise SystemExit(f"{result_path} already exists. The {args.split} split should be evaluated once; "
                         f"pass --force only if you really mean to re-run it.")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    data_dir = Path(args.data_dir).resolve()
    ds = load_split(data_dir, args.split, extra_cols=["PatientID"])
    uids, patients = list(ds["UID"]), np.array(ds["PatientID"])
    print(f"{args.split}: {len(ds)} images from {len(set(patients))} patients")

    preds, info = {}, []
    targets = None
    for run in map(lambda r: Path(r).resolve(), args.runs):
        ckpt = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
        a = ckpt["args"]
        name = a.get("model", "densenet121")  # the first baseline run predates --model
        mean = ckpt.get("mean", MODEL_DEFAULTS[name]["mean"])
        std = ckpt.get("std", MODEL_DEFAULTS[name]["std"])
        model = build_model(name, device, pretrained=False, dropout=a.get("dropout") or 0.0)
        model.load_state_dict(ckpt["model"])
        loader = DataLoader(TaixDataset(ds, a["size"], a["threshold"], augment=False), batch_size=512,
                            num_workers=args.workers, pin_memory=True)
        t0 = time.time()
        p, t = predict(model, loader, device, torch.bfloat16, mean, std)
        print(f"  {run.name}: predicted in {time.time() - t0:.0f}s (best epoch {ckpt['epoch']})")
        if targets is not None and not np.array_equal(targets, t):
            raise RuntimeError("runs use different targets (threshold?) and cannot be compared")
        targets = t
        key = f"{name}:{run.name}"
        preds[key] = p
        info.append({"key": key, "model": name, "label": NAMES.get(name, name), "run": run.name,
                     "best_epoch": ckpt["epoch"], "val_mean_auroc_8": ckpt.get("val_metrics", {}).get("mean_auroc_8")})
        pd.DataFrame(p, columns=[f"p_{c}" for c in LABEL_COLS]).assign(
            UID=uids, PatientID=patients, **{c: t[:, j] for j, c in enumerate(LABEL_COLS)}
        ).to_parquet(out_dir / f"predictions_{run.name}.parquet", index=False)

    metric_names = LABEL_COLS + list(FINDINGS)  # 8 labels, then 5 findings
    n = len(metric_names)
    keys = list(preds)

    # Patient-level bootstrap, shared across models so differences are paired.
    # Columns: AUROC for the 13 metrics, then AUPRC for the same 13.
    rng = np.random.default_rng(args.seed)
    groups = list(pd.Series(np.arange(len(patients))).groupby(patients).indices.values())
    boot = {k: np.empty((args.bootstrap, 2 * n)) for k in keys}
    t0 = time.time()
    for b in range(args.bootstrap):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        for k in keys:
            boot[k][b] = all_scores(preds[k][idx], targets[idx])
    print(f"bootstrap: {args.bootstrap} patient resamples in {time.time() - t0:.0f}s")

    def ci(x):
        return [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]

    def boot_series(B):
        """Bootstrap distribution of every reported quantity, keyed like the results."""
        out = {"mean_auroc_8": B[:, :8].mean(1), "mean_auroc_5": B[:, 8:n].mean(1),
               "mean_auprc_8": B[:, n:n + 8].mean(1), "mean_auprc_5": B[:, n + 8:].mean(1)}
        out.update({f"auroc/{m}": B[:, j] for j, m in enumerate(metric_names)})
        out.update({f"auprc/{m}": B[:, n + j] for j, m in enumerate(metric_names)})
        return out

    def point_series(pt):
        out = {k: pt[k] for k in ("mean_auroc_8", "mean_auroc_5", "mean_auprc_8", "mean_auprc_5")}
        out.update({f"auroc/{m}": v for m, v in {**pt["auroc_8"], **pt["auroc_5"]}.items()})
        out.update({f"auprc/{m}": v for m, v in {**pt["auprc_8"], **pt["auprc_5"]}.items()})
        return out

    point = {k: point_series(metrics(preds[k], targets)) for k in keys}
    bs = {k: boot_series(boot[k]) for k in keys}
    pt_any = metrics(preds[keys[0]], targets)
    prevalence = {**pt_any["prevalence_8"], **pt_any["prevalence_5"]}

    results = {"split": args.split, "n_images": len(ds), "n_patients": int(len(groups)),
               "bootstrap": args.bootstrap, "seed": args.seed, "prevalence": prevalence, "models": []}
    for k, meta in zip(keys, info):
        results["models"].append({**meta, "scores": {q: {"value": float(point[k][q]), "ci": ci(bs[k][q])}
                                                     for q in point[k]}})

    if len(keys) == 2:
        a, b = keys
        results["comparison"] = {
            "difference": f"{info[1]['label']} minus {info[0]['label']}",
            "metrics": {q: {"diff": float(point[b][q] - point[a][q]), "ci": ci(bs[b][q] - bs[a][q]),
                            "p_two_sided": float(min(1.0, 2 * min(np.mean(bs[b][q] - bs[a][q] <= 0),
                                                                  np.mean(bs[b][q] - bs[a][q] >= 0))))}
                        for q in point[a]}}

    result_path.write_text(json.dumps(results, indent=2))

    # Human-readable tables: primary metric (AUROC), then secondary (AUPRC, with prevalence = chance level).
    names = {"mean_8": "**Mean of 8 labels**", "mean_5": "**Mean of 5 findings**"}
    lines = [f"# {args.split.capitalize()} results ({len(ds):,} images, {len(groups):,} patients)", "",
             f"95% CIs from {args.bootstrap} patient-level bootstrap resamples. "
             "The difference is paired (same resamples for both models) with a two-sided bootstrap p-value.", ""]
    for metric, title in [("auroc", "Primary metric: AUROC"), ("auprc", "Secondary metric: AUPRC (average precision)")]:
        hdr = "| Metric | " + ("Prevalence | " if metric == "auprc" else "") + " | ".join(m["label"] for m in results["models"])
        hdr += " | Difference (2nd minus 1st) | p |" if "comparison" in results else " |"
        lines += [f"## {title}", "", hdr, "|" + "---|" * (hdr.count("|") - 1)]
        rows = [(names["mean_8"], f"mean_{metric}_8", None), (names["mean_5"], f"mean_{metric}_5", None)]
        rows += [(m, f"{metric}/{m}", prevalence[m]) for m in metric_names]
        for label, q, prev in rows:
            line = f"| {label} | "
            if metric == "auprc":
                line += (f"{prev:.3f} | " if prev is not None else "– | ")
            line += " | ".join(f"{e['scores'][q]['value']:.3f} ({e['scores'][q]['ci'][0]:.3f}–{e['scores'][q]['ci'][1]:.3f})"
                               for e in results["models"])
            if "comparison" in results:
                c = results["comparison"]["metrics"][q]
                p = c["p_two_sided"]
                p_txt = f"< {1 / args.bootstrap:.3f}" if p == 0 else f"{p:.3f}"  # 0 resamples crossed zero
                line += f" | {c['diff']:+.3f} ({c['ci'][0]:+.3f} to {c['ci'][1]:+.3f}) | {p_txt} |"
            else:
                line += " |"
            lines.append(line)
        lines.append("")
    (out_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved {result_path}")


if __name__ == "__main__":
    main()
