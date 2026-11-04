"""Confirm the full local TAIX-Ray `default` config loads: schema, counts, splits, labels, and
that every one of the 215,381 images decodes.

Usage:
    python scripts/validate_taix.py [--data-dir data/TAIX-Ray] [--workers 30] [--skip-decode]

Writes outputs/validation/{summary.json, image_stats.parquet, failures.csv}.
Exit code is 0 only if every check passes.
"""
import argparse
import io
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

EXPECTED = {"train": 137593, "val": 34860, "test": 42928}
EXPECTED_TOTAL, EXPECTED_PATIENTS = 215381, 47724
META_COLS = ["UID", "Fold", "Split", "PatientID", "PhysicianID", "StudyDate", "Age", "Sex"]
LABEL_COLS = [
    "HeartSize", "PulmonaryCongestion", "PleuralEffusion_Right", "PleuralEffusion_Left",
    "PulmonaryOpacities_Right", "PulmonaryOpacities_Left", "Atelectasis_Right", "Atelectasis_Left",
]


def split_of(path: Path) -> str:
    return path.name.split("-")[0]


def decode_file(path_str: str):
    """Decode every image in one parquet shard; return per-image stats and failures."""
    from PIL import Image

    rows, failures = [], []
    pf = pq.ParquetFile(path_str)
    for batch in pf.iter_batches(batch_size=256, columns=["UID", "Image"]):
        uids = batch.column("UID").to_pylist()
        imgs = batch.column("Image").to_pylist()
        for uid, img in zip(uids, imgs):
            try:
                data = img["bytes"] if isinstance(img, dict) else img
                if not data:
                    raise ValueError("empty image bytes")
                with Image.open(io.BytesIO(data)) as im:
                    im.load()  # forces full decode, not just the header
                    rows.append((uid, im.width, im.height, im.mode, im.format, len(data)))
            except Exception as e:  # noqa: BLE001 - we want to record every failure kind
                failures.append((uid, Path(path_str).name, f"{type(e).__name__}: {e}"))
    return rows, failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("TAIX_DATA_DIR", "data/TAIX-Ray"))
    ap.add_argument("--out-dir", default="outputs/validation")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--skip-decode", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted((data_dir / "data").glob("*.parquet"))
    checks = {}

    print(f"[1/3] Metadata from {len(files)} parquet files in {data_dir / 'data'}")
    checks["n_files == 59"] = len(files) == 59
    meta = pd.concat(
        [pq.read_table(f, columns=META_COLS + LABEL_COLS).to_pandas().assign(_file_split=split_of(f))
         for f in files],
        ignore_index=True,
    )
    counts = meta["Split"].value_counts().to_dict()
    print("  rows per split:", counts, "| total:", len(meta))
    checks["split sizes match paper"] = {k: counts.get(k, 0) for k in EXPECTED} == EXPECTED
    checks["total rows == 215,381"] = len(meta) == EXPECTED_TOTAL
    checks["Split column matches file name"] = bool((meta["Split"] == meta["_file_split"]).all())
    checks["UIDs unique"] = meta["UID"].is_unique
    n_pat = meta["PatientID"].nunique()
    checks["patients == 47,724"] = n_pat == EXPECTED_PATIENTS
    pats = {s: set(meta.loc[meta["Split"] == s, "PatientID"]) for s in EXPECTED}
    overlap = {f"{a}/{b}": len(pats[a] & pats[b])
               for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}
    checks["patient-disjoint splits"] = all(v == 0 for v in overlap.values())
    checks["no missing labels"] = int(meta[LABEL_COLS].isna().sum().sum()) == 0
    label_values = {c: sorted(int(v) for v in meta[c].dropna().unique()) for c in LABEL_COLS}
    print("  patients:", n_pat, "| patient overlap:", overlap)
    print("  label grade values:", label_values)

    summary = {
        "data_dir": str(data_dir), "n_files": len(files), "rows": len(meta),
        "split_counts": counts, "patients": n_pat, "patient_overlap": overlap,
        "label_values": label_values,
        "missing_per_column": {k: int(v) for k, v in meta.isna().sum().items() if v},
    }

    if not args.skip_decode:
        print(f"[2/3] Decoding every image with {args.workers} processes ...")
        t0 = time.time()
        rows, failures, done = [], [], 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(decode_file, str(f)): f for f in files}
            for fut in as_completed(futs):
                r, fl = fut.result()
                rows += r
                failures += fl
                done += 1
                print(f"  {done:2d}/{len(files)} {futs[fut].name}: {len(r)} ok, {len(fl)} failed "
                      f"({time.time() - t0:.0f}s)", flush=True)
        secs = time.time() - t0
        stats = pd.DataFrame(rows, columns=["UID", "width", "height", "mode", "format", "bytes"])
        stats.to_parquet(out_dir / "image_stats.parquet", index=False)
        pd.DataFrame(failures, columns=["UID", "file", "error"]).to_csv(out_dir / "failures.csv", index=False)
        checks["all images decoded"] = len(failures) == 0 and len(stats) == EXPECTED_TOTAL
        checks["decoded UIDs == metadata UIDs"] = set(stats["UID"]) == set(meta["UID"]) - {f[0] for f in failures}
        summary["decode"] = {
            "decoded": len(stats), "failed": len(failures), "seconds": round(secs, 1),
            "images_per_sec": round(len(stats) / secs, 1),
            "modes": dict(Counter(stats["mode"])), "formats": dict(Counter(stats["format"])),
            "top_sizes": {f"{w}x{h}": int(n) for (w, h), n in
                          stats.groupby(["width", "height"]).size().nlargest(5).items()},
            "width_range": [int(stats["width"].min()), int(stats["width"].max())],
            "height_range": [int(stats["height"].min()), int(stats["height"].max())],
            "mean_kb": round(stats["bytes"].mean() / 1024, 1),
        }
        print("  ", json.dumps(summary["decode"]))

    print("[3/3] Checks")
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    summary["checks"] = checks
    summary["all_passed"] = all(checks.values())
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("ALL CHECKS PASSED" if summary["all_passed"] else "SOME CHECKS FAILED",
          "->", out_dir / "summary.json")
    sys.exit(0 if summary["all_passed"] else 1)


if __name__ == "__main__":
    main()
