"""Short DenseNet-121 training benchmark on the local TAIX-Ray `default` config.

Measures (1) GPU-only throughput on synthetic tensors, (2) end-to-end training throughput on real
data including decoding/resizing in DataLoader workers, and (3) a quick validation-subset AUROC as a
sanity check that the model is learning. It is a throughput/smoke benchmark, not a training run.

PROVISIONAL TARGETS: 8 binary outputs, positive = grade >= 2 on each original ordinal column.
The team's final label/uncertainty policy is not decided yet; this does not affect throughput.

PROVISIONAL PREPROCESSING: per-image min-max to [0, 1], bilinear resize to --size, grey -> 3 channels,
ImageNet mean/std. Not the final training preprocessing.

Usage:
    python scripts/bench_densenet.py --size 224 --batch-size 128 --steps 300
    python scripts/bench_densenet.py --size 512 --batch-size 32 --steps 300
"""
import argparse
import io
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

LABEL_COLS = [
    "HeartSize", "PulmonaryCongestion", "PleuralEffusion_Right", "PleuralEffusion_Left",
    "PulmonaryOpacities_Right", "PulmonaryOpacities_Left", "Atelectasis_Right", "Atelectasis_Left",
]
IMAGENET_MEAN, IMAGENET_STD = 0.449, 0.226  # grey-level averages of the ImageNet RGB stats


def load_split(data_dir: Path, split: str):
    """Memory-mapped HF Arrow dataset for one split; images left undecoded (bytes)."""
    import datasets

    ds = datasets.load_dataset(
        "parquet", data_files={split: str(data_dir / "data" / f"{split}-*.parquet")}, split=split
    )
    return ds.select_columns(["Image", *LABEL_COLS]).cast_column("Image", datasets.Image(decode=False))


class TaixDataset(Dataset):
    def __init__(self, hf_ds, size: int):
        self.ds, self.size = hf_ds, size

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        row = self.ds[int(i)]
        with Image.open(io.BytesIO(row["Image"]["bytes"])) as im:
            arr = np.asarray(im, dtype=np.float32)
        if arr.ndim == 3:  # defensive: collapse any RGB/RGBA to grey
            arr = arr[..., :3].mean(axis=2)
        lo, hi = float(arr.min()), float(arr.max())
        arr = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)
        if arr.shape != (self.size, self.size):
            arr = np.asarray(Image.fromarray(arr, mode="F").resize((self.size, self.size), Image.BILINEAR))
        x = torch.tensor(arr, dtype=torch.float16).unsqueeze(0)  # copies; PIL arrays are read-only
        y = torch.tensor([float(row[c] >= 2) for c in LABEL_COLS], dtype=torch.float32)
        return x, y


def build_model(device):
    from torchvision.models import DenseNet121_Weights, densenet121

    m = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    m.classifier = nn.Linear(m.classifier.in_features, len(LABEL_COLS))
    return m.to(device, memory_format=torch.channels_last)


def to_input(x, device):
    x = x.to(device, non_blocking=True).float()
    x = ((x - IMAGENET_MEAN) / IMAGENET_STD).expand(-1, 3, -1, -1)
    return x.contiguous(memory_format=torch.channels_last)


def train_steps(model, opt, batches, steps, warmup, device, amp_dtype):
    """Run warmup+steps optimizer steps; return per-step timings (after warmup) and losses."""
    crit = nn.BCEWithLogitsLoss()
    model.train()
    data_t, step_t, losses = [], [], []
    it = iter(batches)
    for s in range(warmup + steps):
        t0 = time.perf_counter()
        x, y = next(it)
        t1 = time.perf_counter()
        x, y = to_input(x, device), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype):
            loss = crit(model(x).float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        if s >= warmup:
            data_t.append(t1 - t0)
            step_t.append(t2 - t0)
        losses.append(loss.item())
    return np.array(data_t), np.array(step_t), losses


def summarize(name, data_t, step_t, bs):
    total = step_t.sum()
    r = {
        "images_per_sec": round(len(step_t) * bs / total, 1),
        "ms_per_step_median": round(1000 * float(np.median(step_t)), 1),
        "data_wait_fraction": round(float(data_t.sum() / total), 3),
    }
    print(f"  {name}: {r['images_per_sec']} img/s | {r['ms_per_step_median']} ms/step | "
          f"data wait {100 * r['data_wait_fraction']:.1f}% of step time")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("TAIX_DATA_DIR", "data/TAIX-Ray"))
    ap.add_argument("--out-dir", default="outputs/benchmark")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--val-images", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--amp", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    data_dir = Path(args.data_dir).resolve()
    bs = args.batch_size
    print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__} (CUDA {torch.version.cuda})")
    print(f"Config: size={args.size} batch={bs} steps={args.steps} (+{args.warmup} warmup) "
          f"workers={args.workers} amp={args.amp}")

    # 1) GPU-only upper bound: same model/optimizer, synthetic batch already on the GPU.
    print("[1/3] Synthetic GPU-only throughput")
    model = build_model(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    xs = torch.rand(bs, 1, args.size, args.size, dtype=torch.float16, device=device)
    ys = torch.randint(0, 2, (bs, len(LABEL_COLS)), device=device).float()
    torch.cuda.reset_peak_memory_stats()
    syn = summarize("synthetic", *train_steps(model, opt, iter(lambda: (xs, ys), None),
                                              args.steps, args.warmup, device, amp_dtype)[:2], bs)
    syn["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    print(f"  peak GPU memory: {syn['peak_mem_gb']} GB")

    # 2) Real data, fresh ImageNet-initialised model.
    print("[2/3] Real-data training (TAIX-Ray train split)")
    t0 = time.time()
    train_ds, val_ds = load_split(data_dir, "train"), load_split(data_dir, "val")
    print(f"  loaded train={len(train_ds)} val={len(val_ds)} in {time.time() - t0:.0f}s")
    loader = DataLoader(
        TaixDataset(train_ds, args.size), batch_size=bs, shuffle=True, num_workers=args.workers,
        pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=4,
    )
    model = build_model(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    torch.cuda.reset_peak_memory_stats()
    data_t, step_t, losses = train_steps(model, opt, loader, args.steps, args.warmup, device, amp_dtype)
    real = summarize("real data", data_t, step_t, bs)
    real["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    k = min(20, len(losses) // 4)
    real["loss_first"] = round(float(np.mean(losses[:k])), 4)
    real["loss_last"] = round(float(np.mean(losses[-k:])), 4)
    real["images_seen"] = (args.steps + args.warmup) * bs
    print(f"  loss (mean of {k} steps): {real['loss_first']} -> {real['loss_last']}")

    # 3) Quick validation AUROC on a fixed random subset of val, to confirm learning.
    print(f"[3/3] Validation subset AUROC ({args.val_images} images)")
    from sklearn.metrics import roc_auc_score

    idx = np.random.default_rng(args.seed).choice(len(val_ds), size=min(args.val_images, len(val_ds)), replace=False)
    vloader = DataLoader(TaixDataset(val_ds.select(np.sort(idx)), args.size), batch_size=bs * 2,
                         num_workers=args.workers, pin_memory=True)
    model.eval()
    probs, targets = [], []
    t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype):
        for x, y in vloader:
            probs.append(torch.sigmoid(model(to_input(x, device)).float()).cpu())
            targets.append(y)
    p, t = torch.cat(probs).numpy(), torch.cat(targets).numpy()
    aucs = {c: (round(float(roc_auc_score(t[:, j], p[:, j])), 3) if 0 < t[:, j].sum() < len(t) else None)
            for j, c in enumerate(LABEL_COLS)}
    valid = [v for v in aucs.values() if v is not None]
    val = {"auroc": aucs, "mean_auroc": round(float(np.mean(valid)), 3),
           "eval_images_per_sec": round(len(t) / (time.time() - t0), 1)}
    for c, v in aucs.items():
        print(f"  {c:26s} {v}")
    print(f"  mean AUROC: {val['mean_auroc']} (after only {real['images_seen']} training images)")

    # Rough epoch-time projection from measured real-data throughput.
    epoch_min = len(train_ds) / real["images_per_sec"] / 60
    print(f"Projected full train epoch at this throughput: ~{epoch_min:.1f} min")

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "config": vars(args),
        "env": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                "python": platform.python_version(), "os": platform.platform(),
                "cpu_count": os.cpu_count()},
        "synthetic": syn, "real": real, "val_subset": val,
        "projected_epoch_minutes": round(epoch_min, 1),
    }
    path = out / f"densenet121_s{args.size}_b{bs}.json"
    path.write_text(json.dumps(result, indent=2))
    print("Saved", path)


if __name__ == "__main__":
    main()
