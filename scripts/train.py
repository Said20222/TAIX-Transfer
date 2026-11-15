"""Fine-tune DenseNet-121 (ImageNet) or EVA-X-Tiny (chest-X-ray MIM) on TAIX-Ray.

Multi-label abnormality detection with 8 outputs. The best epoch is chosen by mean validation
AUROC, with early stopping. This script never touches the test split; evaluate the chosen
checkpoints once with scripts/evaluate_test.py.

Data pipeline (labels, preprocessing, augmentation) is shared by both models and described in
scripts/common.py. Each model gets its own standard optimizer recipe (MODEL_DEFAULTS there);
any value can be overridden on the command line and is saved to config.json.

Usage:
    python scripts/train.py --model densenet121
    python scripts/train.py --model eva_x_tiny
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

    d = MODEL_DEFAULTS[args.model]
    keys = ("batch_size", "epochs", "lr", "weight_decay", "layer_decay", "drop_path", "warmup_steps", "warmup_epochs")
    explicit = {k for k in keys if getattr(args, k) is not None}
    for key in keys:
        if key not in explicit:
            setattr(args, key, d[key])
    # Warmup is set either in steps or in epochs; a flag given on the command line beats the model default.
    if "warmup_epochs" in explicit and "warmup_steps" not in explicit:
        args.warmup_steps = None
    mean, std = d["mean"], d["std"]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16

    run_name = args.run_name or f"{args.model}_s{args.size}_{datetime.now():%Y%m%d_%H%M%S}"
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
    train_ds, val_ds = load_split(data_dir, "train"), load_split(data_dir, "val")
    pos_rate = {c: float(np.mean(np.array(train_ds[c]) >= args.threshold)) for c in LABEL_COLS}

    common = dict(num_workers=args.workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    train_loader = DataLoader(TaixDataset(train_ds, args.size, args.threshold, not args.no_augment),
                              batch_size=args.batch_size, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(TaixDataset(val_ds, args.size, args.threshold, False),
                            batch_size=min(args.batch_size * 2, 1024), shuffle=False, **common)

    steps_per_epoch = len(train_loader) if not args.max_train_steps else min(args.max_train_steps, len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    warmup = args.warmup_steps if args.warmup_steps is not None else int(round(args.warmup_epochs * steps_per_epoch))

    (run_dir / "config.json").write_text(json.dumps({
        "args": vars(args), "resolved": {"mean": mean, "std": std, "warmup_steps": warmup,
                                         "steps_per_epoch": steps_per_epoch},
        "env": env, "labels": LABEL_COLS, "findings": FINDINGS}, indent=2))
    print(f"Run dir: {run_dir}")
    print(f"{args.model} | GPU {env['gpu']} | size={args.size} batch={args.batch_size} epochs={args.epochs} "
          f"lr={args.lr} wd={args.weight_decay} layer_decay={args.layer_decay} drop_path={args.drop_path} "
          f"warmup={warmup} steps | threshold>={args.threshold} augment={not args.no_augment}")
    print(f"train={len(train_ds)} val={len(val_ds)} | train positive rate: "
          + ", ".join(f"{c}={100 * r:.1f}%" for c, r in pos_rate.items()), flush=True)

    model = build_model(args.model, device, pretrained=True, drop_path=args.drop_path)
    groups = param_groups(model, args.model, args.weight_decay, args.layer_decay)
    for g in groups:
        g["lr"] = args.lr * g.pop("lr_scale")
    opt = torch.optim.AdamW(groups, lr=args.lr)

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    crit = nn.BCEWithLogitsLoss()

    history, best, best_epoch, stale = [], -1.0, -1, 0
    t_run = time.time()
    for epoch in range(1, args.epochs + 1):
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

        t1 = time.time()
        probs, targets = predict(model, val_loader, device, amp_dtype, mean, std)
        val_loss = float(nn.functional.binary_cross_entropy(torch.from_numpy(probs).clamp(1e-7, 1 - 1e-7),
                                                            torch.from_numpy(targets)))
        m = metrics(probs, targets)
        row = {"epoch": epoch, "train_loss": loss_sum / n, "val_loss": val_loss,
               "val_mean_auroc_8": m["mean_auroc_8"], "val_mean_auroc_5": m["mean_auroc_5"],
               **{f"val_auroc_{k}": v for k, v in m["auroc_8"].items()},
               "lr": max(sched.get_last_lr()), "train_secs": train_secs, "val_secs": time.time() - t1,
               "train_img_per_sec": n * args.batch_size / train_secs}
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

        improved = m["mean_auroc_8"] > best
        if improved:
            best, best_epoch, stale = m["mean_auroc_8"], epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_metrics": m, "args": vars(args),
                        "mean": mean, "std": std}, run_dir / "best.pt")
            pd.DataFrame(probs, columns=[f"p_{c}" for c in LABEL_COLS]).assign(
                UID=list(val_ds["UID"]), **{c: targets[:, j] for j, c in enumerate(LABEL_COLS)}
            ).to_parquet(run_dir / "val_predictions_best.parquet", index=False)
        else:
            stale += 1
        print(f"epoch {epoch:2d}/{args.epochs} | train loss {row['train_loss']:.4f} | val loss {val_loss:.4f} | "
              f"val AUROC 8={m['mean_auroc_8']:.4f} 5={m['mean_auroc_5']:.4f} | "
              f"{train_secs / 60:.1f}+{row['val_secs'] / 60:.1f} min | {row['train_img_per_sec']:.0f} img/s"
              f"{' *best*' if improved else ''}", flush=True)
        if stale >= args.patience:
            print(f"Early stopping: no improvement for {args.patience} epochs.")
            break

    torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args), "mean": mean, "std": std},
               run_dir / "last.pt")
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    summary = {"model": args.model, "best_epoch": best_epoch, "epochs_run": epoch,
               "minutes": round((time.time() - t_run) / 60, 1), "val": ckpt["val_metrics"],
               "train_positive_rate": pos_rate}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    v = summary["val"]
    print(f"\nBest epoch {best_epoch} | val mean AUROC (8 labels) {v['mean_auroc_8']:.4f} | "
          f"(5 findings) {v['mean_auroc_5']:.4f}")
    for k, a in {**v["auroc_8"], **v["auroc_5"]}.items():
        print(f"  {k:26s} {a:.4f}")
    print(f"Done in {summary['minutes']} min -> {run_dir}")


if __name__ == "__main__":
    main()
