"""Fine-tune DenseNet-121 (ImageNet) or EVA-X-Tiny (chest-X-ray MIM) on TAIX-Ray.

Multi-label abnormality detection with 8 outputs. The best epoch is chosen by mean validation
AUROC (primary metric), with early stopping; mean validation AUPRC is logged as the secondary
metric. This script never touches the test split; evaluate the chosen checkpoints once with
scripts/evaluate_test.py.

Data pipeline (labels, preprocessing, augmentation) is shared by both models and described in
scripts/common.py. Each model gets its own optimizer recipe (MODEL_DEFAULTS there); any value can
be overridden on the command line and is saved to config.json. scripts/hpo.py calls train_model()
directly to search these values.

Usage:
    python scripts/train.py --model densenet121
    python scripts/train.py --model eva_x_tiny
    python scripts/train.py --model densenet121 --lr 3e-4 --weight-decay 1e-3 --dropout 0.2
"""
import argparse
import json
import math
import os
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import (FINDINGS, LABEL_COLS, MODEL_DEFAULTS, TaixDataset, build_model, load_split, metrics,
                    param_groups, predict, to_input)

TUNABLE = ("batch_size", "epochs", "lr", "weight_decay", "layer_decay", "drop_path", "dropout",
           "warmup_steps", "warmup_epochs")


def resolve_config(model: str, overrides: dict) -> dict:
    """Model defaults, overridden by any non-None value in `overrides`.

    Warmup is given either in steps or in epochs; an explicit value beats the model default.
    """
    d = MODEL_DEFAULTS[model]
    cfg = {"model": model, "size": 224, "patience": 5, "threshold": 2, "augment": True, "amp": "bf16",
           "seed": 0, "max_train_steps": 0, "workers": 16}
    cfg.update({k: v for k, v in overrides.items() if k not in TUNABLE and v is not None})
    explicit = {k for k in TUNABLE if overrides.get(k) is not None}
    for k in TUNABLE:
        cfg[k] = overrides[k] if k in explicit else d[k]
    if "warmup_epochs" in explicit and "warmup_steps" not in explicit:
        cfg["warmup_steps"] = None
    cfg["mean"], cfg["std"] = d["mean"], d["std"]
    return cfg


def make_loaders(cfg: dict, data_dir: Path):
    """Train/val datasets and loaders. Reusable across runs with the same batch size."""
    train_ds, val_ds = load_split(data_dir, "train"), load_split(data_dir, "val")
    common = dict(num_workers=cfg["workers"], pin_memory=True, persistent_workers=True, prefetch_factor=2)
    train_loader = DataLoader(TaixDataset(train_ds, cfg["size"], cfg["threshold"], cfg["augment"]),
                              batch_size=cfg["batch_size"], shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(TaixDataset(val_ds, cfg["size"], cfg["threshold"], False),
                            batch_size=min(cfg["batch_size"] * 2, 1024), shuffle=False, **common)
    return train_ds, val_ds, train_loader, val_loader


def train_model(cfg: dict, train_loader, val_loader, val_uids=None, run_dir: Path = None, report=None,
                log=print) -> dict:
    """Train one configuration; returns {"history": [...], "best": {...}}.

    run_dir:  if given, writes history.csv, best.pt, last.pt and validation predictions there.
    report:   optional callback report(epoch, row) after each epoch; it may raise to stop the run
              (scripts/hpo.py uses it for successive-halving pruning).
    """
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    amp_dtype = torch.bfloat16 if cfg["amp"] == "bf16" else torch.float16
    mean, std = cfg["mean"], cfg["std"]

    steps_per_epoch = len(train_loader) if not cfg["max_train_steps"] else min(cfg["max_train_steps"], len(train_loader))
    total_steps = steps_per_epoch * cfg["epochs"]
    warmup = (cfg["warmup_steps"] if cfg["warmup_steps"] is not None
              else int(round(cfg["warmup_epochs"] * steps_per_epoch)))
    cfg["resolved_warmup_steps"], cfg["steps_per_epoch"] = warmup, steps_per_epoch

    model = build_model(cfg["model"], device, pretrained=True, drop_path=cfg["drop_path"], dropout=cfg["dropout"])
    groups = param_groups(model, cfg["model"], cfg["weight_decay"], cfg["layer_decay"])
    for g in groups:
        g["lr"] = cfg["lr"] * g.pop("lr_scale")
    opt = torch.optim.AdamW(groups, lr=cfg["lr"])

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    crit = nn.BCEWithLogitsLoss()

    history, best, stale = [], None, 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        t0, loss_sum, n = time.time(), 0.0, 0
        for step, (x, y) in enumerate(train_loader):
            if step >= steps_per_epoch:
                break
            x, y = to_input(x, device, mean, std), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype):
                loss = crit(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item()
            n += 1
        train_secs = time.time() - t0
        if not math.isfinite(loss_sum):
            raise FloatingPointError(f"training loss became {loss_sum} in epoch {epoch}")

        t1 = time.time()
        probs, targets = predict(model, val_loader, device, amp_dtype, mean, std)
        val_loss = float(nn.functional.binary_cross_entropy(torch.from_numpy(probs).clamp(1e-7, 1 - 1e-7),
                                                            torch.from_numpy(targets)))
        m = metrics(probs, targets)
        row = {"epoch": epoch, "train_loss": loss_sum / n, "val_loss": val_loss,
               "val_mean_auroc_8": m["mean_auroc_8"], "val_mean_auroc_5": m["mean_auroc_5"],
               "val_mean_auprc_8": m["mean_auprc_8"], "val_mean_auprc_5": m["mean_auprc_5"],
               **{f"val_auroc_{k}": v for k, v in m["auroc_8"].items()},
               **{f"val_auprc_{k}": v for k, v in m["auprc_8"].items()},
               "lr": max(sched.get_last_lr()), "train_secs": train_secs, "val_secs": time.time() - t1,
               "train_img_per_sec": n * cfg["batch_size"] / train_secs}
        history.append(row)

        improved = best is None or m["mean_auroc_8"] > best["val_mean_auroc_8"]
        if improved:
            best, stale = {**row, "val_metrics": m}, 0
            if run_dir is not None:
                torch.save({"model": model.state_dict(), "epoch": epoch, "val_metrics": m, "args": cfg,
                            "mean": mean, "std": std}, run_dir / "best.pt")
                pd.DataFrame(probs, columns=[f"p_{c}" for c in LABEL_COLS]).assign(
                    UID=val_uids, **{c: targets[:, j] for j, c in enumerate(LABEL_COLS)}
                ).to_parquet(run_dir / "val_predictions_best.parquet", index=False)
        else:
            stale += 1
        if run_dir is not None:
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        log(f"epoch {epoch:2d}/{cfg['epochs']} | train loss {row['train_loss']:.4f} | val loss {val_loss:.4f} | "
            f"val AUROC 8={m['mean_auroc_8']:.4f} 5={m['mean_auroc_5']:.4f} | "
            f"AUPRC 8={m['mean_auprc_8']:.4f} | {train_secs / 60:.1f}+{row['val_secs'] / 60:.1f} min | "
            f"{row['train_img_per_sec']:.0f} img/s{' *best*' if improved else ''}")
        if report is not None:
            report(epoch, row)
        if stale >= cfg["patience"]:
            log(f"Early stopping: no improvement for {cfg['patience']} epochs.")
            break

    if run_dir is not None:
        torch.save({"model": model.state_dict(), "epoch": epoch, "args": cfg, "mean": mean, "std": std},
                   run_dir / "last.pt")
    return {"history": history, "best": best, "epochs_run": epoch}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_DEFAULTS), default="densenet121")
    ap.add_argument("--data-dir", default=os.environ.get("TAIX_DATA_DIR", "data/TAIX-Ray"))
    ap.add_argument("--runs-dir", default="outputs/runs")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--size", type=int, default=224)
    # Model-specific defaults (None = take from MODEL_DEFAULTS[--model]).
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--layer-decay", type=float, default=None)
    ap.add_argument("--drop-path", type=float, default=None)
    ap.add_argument("--dropout", type=float, default=None, help="DenseNet-121 classifier dropout")
    ap.add_argument("--warmup-steps", type=int, default=None)
    ap.add_argument("--warmup-epochs", type=float, default=None)
    ap.add_argument("--patience", type=int, default=5, help="epochs without val improvement before stopping")
    ap.add_argument("--threshold", type=int, default=2, help="grade >= threshold counts as positive")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--amp", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train-steps", type=int, default=0, help="debug: cap steps per epoch")
    args = ap.parse_args()

    overrides = {k: v for k, v in vars(args).items() if k not in ("model", "data_dir", "runs_dir", "run_name", "no_augment")}
    overrides["augment"] = not args.no_augment
    cfg = resolve_config(args.model, overrides)

    run_name = args.run_name or f"{args.model}_s{cfg['size']}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = Path(args.runs_dir).resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    env = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
           "python": platform.python_version(), "os": platform.platform()}
    try:
        import timm
        env["timm"] = timm.__version__
    except ImportError:
        pass

    data_dir = Path(args.data_dir).resolve()
    train_ds, val_ds, train_loader, val_loader = make_loaders(cfg, data_dir)
    pos_rate = {c: float(np.mean(np.array(train_ds[c]) >= cfg["threshold"])) for c in LABEL_COLS}
    print(f"Run dir: {run_dir}")
    print(f"{cfg['model']} | GPU {env['gpu']} | " + " ".join(f"{k}={cfg[k]}" for k in TUNABLE))
    print(f"train={len(train_ds)} val={len(val_ds)} | train positive rate: "
          + ", ".join(f"{c}={100 * r:.1f}%" for c, r in pos_rate.items()), flush=True)

    (run_dir / "config.json").write_text(json.dumps({"args": cfg, "env": env, "labels": LABEL_COLS,
                                                     "findings": FINDINGS}, indent=2))
    t_run = time.time()
    out = train_model(cfg, train_loader, val_loader, val_uids=list(val_ds["UID"]), run_dir=run_dir,
                      log=lambda s: print(s, flush=True))
    (run_dir / "config.json").write_text(json.dumps({"args": cfg, "env": env, "labels": LABEL_COLS,
                                                     "findings": FINDINGS}, indent=2))
    b = out["best"]
    summary = {"model": cfg["model"], "best_epoch": b["epoch"], "epochs_run": out["epochs_run"],
               "minutes": round((time.time() - t_run) / 60, 1), "val": b["val_metrics"],
               "train_positive_rate": pos_rate}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    v = summary["val"]
    print(f"\nBest epoch {b['epoch']} | val mean AUROC (8) {v['mean_auroc_8']:.4f} (5) {v['mean_auroc_5']:.4f} | "
          f"val mean AUPRC (8) {v['mean_auprc_8']:.4f} (5) {v['mean_auprc_5']:.4f}")
    for k in v["auroc_8"]:
        print(f"  {k:26s} AUROC {v['auroc_8'][k]:.4f}  AUPRC {v['auprc_8'][k]:.4f}")
    print(f"Done in {summary['minutes']} min -> {run_dir}")


if __name__ == "__main__":
    main()
