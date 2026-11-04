"""Train DenseNet-121 (ImageNet init) on TAIX-Ray for multi-label abnormality detection.

Selects the best epoch by mean validation AUROC. The test split is evaluated only with --eval-test,
once, on the best checkpoint; do not use it for model selection.

PROVISIONAL DEFAULTS (team policy not final; everything is a CLI flag and saved to config.json):
  targets   8 binary outputs, positive = grade >= 2 on each original ordinal column
            (also reports the EDA's 5-finding AUROC: bilateral score = max(left, right))
  input     per-image min-max to [0, 1], zero-pad to square (keeps aspect), resize to --size
  augment   small random affine (rotation/translation/scale), NO horizontal flip (labels are L/R)

Usage:
    python scripts/train_densenet.py                       # 224 px, up to 20 epochs
    python scripts/train_densenet.py --size 512 --batch-size 48 --epochs 15
"""
import argparse
import io
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
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

LABEL_COLS = [
    "HeartSize", "PulmonaryCongestion", "PleuralEffusion_Right", "PleuralEffusion_Left",
    "PulmonaryOpacities_Right", "PulmonaryOpacities_Left", "Atelectasis_Right", "Atelectasis_Left",
]
# EDA's five findings: single columns, or (right, left) pairs combined with max / OR.
FINDINGS = {
    "Cardiomegaly": ["HeartSize"],
    "Congestion": ["PulmonaryCongestion"],
    "PleuralEffusion": ["PleuralEffusion_Right", "PleuralEffusion_Left"],
    "PulmonaryOpacity": ["PulmonaryOpacities_Right", "PulmonaryOpacities_Left"],
    "Atelectasis": ["Atelectasis_Right", "Atelectasis_Left"],
}
IMAGENET_MEAN, IMAGENET_STD = 0.449, 0.226  # grey-level averages of the ImageNet RGB stats


def load_split(data_dir: Path, split: str):
    import datasets

    ds = datasets.load_dataset(
        "parquet", data_files={split: str(data_dir / "data" / f"{split}-*.parquet")}, split=split
    )
    return ds.select_columns(["UID", "Image", *LABEL_COLS]).cast_column("Image", datasets.Image(decode=False))


class TaixDataset(Dataset):
    def __init__(self, hf_ds, size: int, threshold: int, augment: bool):
        self.ds, self.size, self.threshold, self.augment = hf_ds, size, threshold, augment
        self.affine = None
        if augment:
            from torchvision.transforms import v2

            self.affine = v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        row = self.ds[int(i)]
        with Image.open(io.BytesIO(row["Image"]["bytes"])) as im:
            arr = np.asarray(im, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[..., :3].mean(axis=2)
        lo, hi = float(arr.min()), float(arr.max())
        arr = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)
        h, w = arr.shape
        side = max(h, w)
        if h != w:  # zero-pad to a centred square so the anatomy is not stretched
            sq = np.zeros((side, side), dtype=np.float32)
            top, left = (side - h) // 2, (side - w) // 2
            sq[top:top + h, left:left + w] = arr
            arr = sq
        if side != self.size:
            arr = np.asarray(Image.fromarray(arr, mode="F").resize((self.size, self.size), Image.BILINEAR))
        x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
        if self.affine is not None:
            x = self.affine(x)
        y = torch.tensor([float(row[c] >= self.threshold) for c in LABEL_COLS], dtype=torch.float32)
        return x.to(torch.float16), y


def build_model(device):
    from torchvision.models import DenseNet121_Weights, densenet121

    m = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    m.classifier = nn.Linear(m.classifier.in_features, len(LABEL_COLS))
    return m.to(device, memory_format=torch.channels_last)


def to_input(x, device):
    x = x.to(device, non_blocking=True).float()
    x = ((x - IMAGENET_MEAN) / IMAGENET_STD).expand(-1, 3, -1, -1)
    return x.contiguous(memory_format=torch.channels_last)


def auroc(y, p):
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def metrics(probs: np.ndarray, targets: np.ndarray) -> dict:
    per_label = {c: auroc(targets[:, j], probs[:, j]) for j, c in enumerate(LABEL_COLS)}
    idx = {c: j for j, c in enumerate(LABEL_COLS)}
    per_finding = {}
    for name, cols in FINDINGS.items():
        js = [idx[c] for c in cols]
        per_finding[name] = auroc(targets[:, js].max(axis=1), probs[:, js].max(axis=1))
    return {
        "mean_auroc_8": float(np.nanmean(list(per_label.values()))),
        "mean_auroc_5": float(np.nanmean(list(per_finding.values()))),
        "auroc_8": per_label, "auroc_5": per_finding,
    }


@torch.no_grad()
def predict(model, loader, device, amp_dtype):
    model.eval()
    probs, targets = [], []
    with torch.autocast("cuda", dtype=amp_dtype):
        for x, y in loader:
            probs.append(torch.sigmoid(model(to_input(x, device)).float()).cpu())
            targets.append(y)
    return torch.cat(probs).numpy(), torch.cat(targets).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("TAIX_DATA_DIR", "data/TAIX-Ray"))
    ap.add_argument("--runs-dir", default="outputs/runs")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5, help="epochs without val improvement before stopping")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--threshold", type=int, default=2, help="grade >= threshold counts as positive")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--amp", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train-steps", type=int, default=0, help="debug: cap steps per epoch")
    ap.add_argument("--eval-test", action="store_true", help="evaluate the best checkpoint on test once")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16

    run_name = args.run_name or f"densenet121_s{args.size}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = Path(args.runs_dir).resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    env = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
           "python": platform.python_version(), "os": platform.platform()}
    (run_dir / "config.json").write_text(json.dumps({"args": vars(args), "env": env,
                                                     "labels": LABEL_COLS, "findings": FINDINGS}, indent=2))
    print(f"Run dir: {run_dir}")
    print(f"GPU: {env['gpu']} | torch {env['torch']} | size={args.size} batch={args.batch_size} "
          f"epochs={args.epochs} lr={args.lr} threshold>={args.threshold} augment={not args.no_augment}")

    data_dir = Path(args.data_dir).resolve()
    train_ds, val_ds = load_split(data_dir, "train"), load_split(data_dir, "val")
    pos_rate = {c: float(np.mean(np.array(train_ds[c]) >= args.threshold)) for c in LABEL_COLS}
    print(f"train={len(train_ds)} val={len(val_ds)} | train positive rate: "
          + ", ".join(f"{c}={100 * r:.1f}%" for c, r in pos_rate.items()))

    common = dict(num_workers=args.workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    train_loader = DataLoader(TaixDataset(train_ds, args.size, args.threshold, not args.no_augment),
                              batch_size=args.batch_size, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(TaixDataset(val_ds, args.size, args.threshold, False),
                            batch_size=args.batch_size * 2, shuffle=False, **common)

    model = build_model(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader) if not args.max_train_steps else min(args.max_train_steps, len(train_loader))
    total_steps = steps_per_epoch * args.epochs

    def lr_lambda(step):
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
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
            x, y = to_input(x, device), y.to(device, non_blocking=True)
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
        probs, targets = predict(model, val_loader, device, amp_dtype)
        val_loss = float(nn.functional.binary_cross_entropy(torch.from_numpy(probs).clamp(1e-7, 1 - 1e-7),
                                                            torch.from_numpy(targets)))
        m = metrics(probs, targets)
        row = {"epoch": epoch, "train_loss": loss_sum / n, "val_loss": val_loss,
               "val_mean_auroc_8": m["mean_auroc_8"], "val_mean_auroc_5": m["mean_auroc_5"],
               **{f"val_auroc_{k}": v for k, v in m["auroc_8"].items()},
               "lr": sched.get_last_lr()[0], "train_secs": train_secs, "val_secs": time.time() - t1,
               "train_img_per_sec": n * args.batch_size / train_secs}
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

        improved = m["mean_auroc_8"] > best
        if improved:
            best, best_epoch, stale = m["mean_auroc_8"], epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_metrics": m, "args": vars(args)},
                       run_dir / "best.pt")
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

    torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)}, run_dir / "last.pt")
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    summary = {"best_epoch": best_epoch, "epochs_run": epoch, "minutes": round((time.time() - t_run) / 60, 1),
               "val": ckpt["val_metrics"], "train_positive_rate": pos_rate}

    if args.eval_test:
        print("Evaluating best checkpoint on TEST (once) ...")
        model.load_state_dict(ckpt["model"])
        test_ds = load_split(data_dir, "test")
        test_loader = DataLoader(TaixDataset(test_ds, args.size, args.threshold, False),
                                 batch_size=args.batch_size * 2, num_workers=args.workers, pin_memory=True)
        p, t = predict(model, test_loader, device, amp_dtype)
        summary["test"] = metrics(p, t)
        pd.DataFrame(p, columns=[f"p_{c}" for c in LABEL_COLS]).assign(
            UID=list(test_ds["UID"]), **{c: t[:, j] for j, c in enumerate(LABEL_COLS)}
        ).to_parquet(run_dir / "test_predictions.parquet", index=False)

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    v = summary["val"]
    print(f"\nBest epoch {best_epoch} | val mean AUROC (8 labels) {v['mean_auroc_8']:.4f} | "
          f"(5 findings) {v['mean_auroc_5']:.4f}")
    for k, a in v["auroc_8"].items():
        print(f"  {k:26s} {a:.4f}")
    for k, a in v["auroc_5"].items():
        print(f"  {k:26s} {a:.4f}")
    if "test" in summary:
        print(f"TEST mean AUROC (8) {summary['test']['mean_auroc_8']:.4f} | (5) {summary['test']['mean_auroc_5']:.4f}")
    print(f"Done in {summary['minutes']} min -> {run_dir}")


if __name__ == "__main__":
    main()
