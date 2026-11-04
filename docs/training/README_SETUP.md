# Training Setup — Local GPU (RTX 5090)

How to set up a local GPU machine for TAIX-Ray training, confirm the full dataset loads, and run the
DenseNet-121 throughput benchmark. Last verified on 2026-11-04.

## Reference machine

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32 GB (Blackwell, compute capability 12.0 / `sm_120`) |
| CPU / RAM | AMD Ryzen 9 9950X (16 cores / 32 threads), 64 GB |
| OS | Windows 11 Pro, NVIDIA driver 610.88 |
| Software | Python 3.12.13, PyTorch 2.14.1 + CUDA 13.0, torchvision 0.29.1, cuDNN 9.24, `datasets` 4.8.5 |

## 1. Environment

RTX 50-series GPUs require a PyTorch build with CUDA ≥ 12.8. Older builds install without error but
fail with `no kernel image is available for execution on the device`.

```bash
conda create -n taix python=3.12 -y
conda activate taix
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements-train.txt
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"  # must list sm_120
```

## 2. Download the dataset

```bash
python scripts/download_taix.py            # -> data/TAIX-Ray/data/*.parquet  (gitignored)
```

- Downloads the Hugging Face **`default` config**: all 215,381 images as 59 parquet files (**58.3 GB**),
  pinned to revision `7f43720e`. The images are 16-bit greyscale PNGs, 512 px wide.
- The `original` config (full-resolution images) is **1.27 TB** and is not needed at 224 or 512 px.
- The script can be re-run safely; it skips files that are already complete. Use `--data-dir` or
  `TAIX_DATA_DIR` to store the data elsewhere.
- A VPN can be a bottleneck. On the reference machine, Cloudflare WARP held downloads to about 2–5 MB/s.
  With WARP off, they ran at about 80 MB/s.

## 3. Confirm the full dataset loads

```bash
python scripts/validate_taix.py --workers 30
```

The script checks metadata and decodes **every** image. It writes `outputs/validation/summary.json`,
`image_stats.parquet` and `failures.csv`, and exits non-zero if any check fails.

**Result (2026-11-04): all 10 checks passed**

| Check | Result |
|---|---|
| Rows (train / val / test) | 137,593 / 34,860 / 42,928 = **215,381** ✔ (matches paper) |
| Patients | **47,724**, with zero overlap between any pair of splits ✔ |
| UIDs | unique; split column consistent with file names ✔ |
| Labels | no missing values; HeartSize grades 0–3, all other findings 0–4 ✔ |
| Image decode | **215,381 / 215,381 decoded, 0 failures**, in 48 s with 30 processes (≈ 4,500 img/s) |
| Image format | all PNG `I;16`; most common sizes 512×421 (52%) and 512×419 (20%); width 373–512, height 266–512 |

Note that pixel ranges differ between images: some are 12-bit (0–4095) and some use the full 16-bit
range. This matches the acquisition shift identified in the EDA, so preprocessing must normalize each
image (or each source) separately.

## 4. DenseNet-121 training benchmark

```bash
python scripts/bench_densenet.py --size 224 --batch-size 128 --steps 300
python scripts/bench_densenet.py --size 512 --batch-size 32  --steps 300
```

The benchmark runs three stages:

1. A GPU-only run on synthetic tensors, which gives an upper bound.
2. 300 training steps on the real train split, with PNG decoding and resizing in 14 DataLoader
   workers. The model is DenseNet-121 initialized from ImageNet, trained with AdamW, bf16 autocast
   and channels-last.
3. AUROC on a fixed 4,096-image validation subset, as a check that the model is learning.

Results are saved as JSON in `outputs/benchmark/`.

| Input | Batch | GPU-only | **Real data** | Data-loader wait | Peak VRAM | Projected epoch (137.6k imgs) | Loss (first → last 20 steps) | Val-subset mean AUROC* |
|---|---|---|---|---|---|---|---|---|
| 224×224 | 128 | 2,006 img/s | **1,906 img/s** | 0.3% | 8.8 GB | **≈ 1.2 min** | 0.654 → 0.469 | 0.839 (after 41k imgs) |
| 512×512 | 32 | 384 img/s | **377 img/s** | 0.2% | 11.4 GB | **≈ 6.1 min** | 0.661 → 0.492 | 0.819 (after 10k imgs) |

\*These AUROCs come from a few hundred steps with **provisional** targets and are not results. They
only show that the pipeline learns.

**Takeaways**

- The GPU is the bottleneck. Real-data throughput is 95–98% of the synthetic figure, and the loader
  waits less than 0.5% of the time. The CPU decodes PNGs fast enough even at 224 px.
- A full DenseNet-121 epoch takes about 1 minute at 224 px or about 6 minutes at 512 px. A 30-epoch run
  takes about 40 minutes at 224 px or about 3 hours at 512 px, plus evaluation time.
- VRAM has headroom. Memory grows roughly in proportion to batch size, so at 512 px a batch of about
  64 (≈ 23 GB) should fit in 32 GB, but 128 would not (≈ 45 GB). At 224 px, batches of 256 or more fit.

**Provisional choices in the benchmark (not final pipeline decisions)**

- Targets: 8 binary outputs, positive when grade ≥ 2 on each original ordinal column.
- Preprocessing: per-image min-max scaling to [0, 1], then a bilinear resize to a square. This stretches
  the roughly 512×420 images vertically; padding or cropping to a square is an open choice.
- No augmentation and no learning-rate schedule.

**Disk use:** the parquet files take 58 GB in `data/`. On first use, `datasets` also builds an Arrow
cache of about 44 GB in `~/.cache/huggingface/datasets/parquet`. Later runs load it in about 5 s.
