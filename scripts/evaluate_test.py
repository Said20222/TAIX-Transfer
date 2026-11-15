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


def all_aurocs(probs8, targets8):
    """13 AUROCs (8 labels, then 5 findings) for one set of predictions."""
    p5, t5 = finding_arrays(probs8, targets8)
    a8 = [fast_auroc(targets8[:, j], probs8[:, j]) for j in range(probs8.shape[1])]
    a5 = [fast_auroc(t5[:, j], p5[:, j]) for j in range(p5.shape[1])]
    return np.array(a8 + a5)


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
        model = build_model(name, device, pretrained=False)
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

    metric_names = LABEL_COLS + list(FINDINGS)
    keys = list(preds)

    # Patient-level bootstrap, shared across models so differences are paired.
    rng = np.random.default_rng(args.seed)
    groups = list(pd.Series(np.arange(len(patients))).groupby(patients).indices.values())
    boot = {k: np.empty((args.bootstrap, len(metric_names))) for k in keys}
    t0 = time.time()
    for b in range(args.bootstrap):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        for k in keys:
            boot[k][b] = all_aurocs(preds[k][idx], targets[idx])
    print(f"bootstrap: {args.bootstrap} patient resamples in {time.time() - t0:.0f}s")

    def ci(x):
        return [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]

    results = {"split": args.split, "n_images": len(ds), "n_patients": int(len(groups)),
               "bootstrap": args.bootstrap, "seed": args.seed, "models": []}
    for k, meta in zip(keys, info):
        point = metrics(preds[k], targets)
        b = boot[k]
        entry = {**meta,
                 "mean_auroc_8": point["mean_auroc_8"], "mean_auroc_8_ci": ci(b[:, :8].mean(1)),
                 "mean_auroc_5": point["mean_auroc_5"], "mean_auroc_5_ci": ci(b[:, 8:].mean(1)),
                 "auroc": {**point["auroc_8"], **point["auroc_5"]},
                 "auroc_ci": {m: ci(b[:, j]) for j, m in enumerate(metric_names)}}
        results["models"].append(entry)

    if len(keys) == 2:
        a, b = keys
        d_all = boot[b] - boot[a]
        diffs = {"mean_auroc_8": boot[b][:, :8].mean(1) - boot[a][:, :8].mean(1),
                 "mean_auroc_5": boot[b][:, 8:].mean(1) - boot[a][:, 8:].mean(1),
                 **{m: d_all[:, j] for j, m in enumerate(metric_names)}}
        pa, pb = results["models"]
        point_d = {"mean_auroc_8": pb["mean_auroc_8"] - pa["mean_auroc_8"],
                   "mean_auroc_5": pb["mean_auroc_5"] - pa["mean_auroc_5"],
                   **{m: pb["auroc"][m] - pa["auroc"][m] for m in metric_names}}
        results["comparison"] = {
            "difference": f"{info[1]['label']} minus {info[0]['label']}",
            "metrics": {m: {"diff": float(point_d[m]), "ci": ci(x),
                            "p_two_sided": float(min(1.0, 2 * min(np.mean(x <= 0), np.mean(x >= 0))))}
                        for m, x in diffs.items()}}

    result_path.write_text(json.dumps(results, indent=2))

    # Human-readable table.
    lines = [f"# {args.split.capitalize()} results ({len(ds):,} images, {len(groups):,} patients)", "",
             "AUROC with 95% patient-level bootstrap CI "
             f"({args.bootstrap} resamples).", ""]
    hdr = "| Metric | " + " | ".join(m["label"] for m in results["models"])
    if "comparison" in results:
        hdr += " | Difference (2nd minus 1st) | p |"
    else:
        hdr += " |"
    lines += [hdr, "|" + "---|" * (hdr.count("|") - 1)]
    rows = [("Mean of 8 labels", "mean_auroc_8"), ("Mean of 5 findings", "mean_auroc_5")] + [(m, m) for m in metric_names]
    for label, m in rows:
        cells = []
        for e in results["models"]:
            v, (lo, hi) = (e[m], e[f"{m}_ci"]) if m.startswith("mean_") else (e["auroc"][m], e["auroc_ci"][m])
            cells.append(f"{v:.3f} ({lo:.3f}–{hi:.3f})")
        line = f"| {label} | " + " | ".join(cells)
        if "comparison" in results:
            c = results["comparison"]["metrics"][m]
            p = c["p_two_sided"]
            p_txt = f"< {1 / args.bootstrap:.3f}" if p == 0 else f"{p:.3f}"  # 0 resamples crossed zero
            line += f" | {c['diff']:+.3f} ({c['ci'][0]:+.3f} to {c['ci'][1]:+.3f}) | {p_txt} |"
        else:
            line += " |"
        lines.append(line)
    (out_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved {result_path}")


if __name__ == "__main__":
    main()
