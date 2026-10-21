# TAIX-Transfer — Exploratory Data Analysis (Data Understanding)

This section is ready to copy into the project's main `README.md`, or it can be linked as a standalone `README_EDA.md`.

## Overview

`01_data_understanding_clean.ipynb` implements **CRISP-DM Phase 2: Data Understanding** for [TAIX-Ray](https://huggingface.co/datasets/TLAIM/TAIX-Ray). It covers official patient-disjoint splits, eight ordinal label distributions, exploratory five-finding associations, demographics, longitudinal/temporal patterns, image dimensions, 16-bit grayscale intensity characteristics, and data integrity checks.

The full-metadata analyses cover **215,381 radiographs from 47,724 patients**. Image-level analyses use a **reproducible, split-balanced sample of 900 radiographs** (300 each from train, validation, and test); they do not cover all 215,381 image files.

The notebook contains an interactive table of contents and requires no changes to its analysis cells. All configuration is centralized in its first configuration cell.

## Requirements

- Python **3.10+** (recommended: 3.11), Jupyter / Google Colab.
- Internet access to Hugging Face **only when caches or sampled PNGs need to be created**.
- Sufficient local or Google Drive storage for the streamed metadata cache and 900 sampled PNGs. Downloads can be slow: **streaming may traverse many images**, despite saving only 900.
- **No GPU is necessary** for this EDA; CPU and storage I/O dominate. A Colab T4 is not required.

Install the dependencies:

```bash
python -m pip install -r requirements-eda.txt
```

For a local Jupyter environment, also install/register a kernel if necessary:

```bash
python -m ipykernel install --user --name taix-eda --display-name "Python (TAIX EDA)"
```

## Run in Google Colab

1. Open `01_data_understanding_clean.ipynb` in Colab (upload directly, or open it from GitHub/Drive).
2. Install dependencies in a new temporary Colab cell, then restart the runtime if Colab asks:

   ```python
   %pip install datasets numpy pandas pyarrow matplotlib pillow tqdm huggingface_hub ipykernel
   ```

   If you opened the notebook from a cloned repository, you can instead run `%pip install -r requirements-eda.txt` from the repository root.
3. In the notebook's **Configuration** cell, keep `USE_GOOGLE_DRIVE = True`. Authorize the `drive.mount()` request. Its default folder is `/content/drive/MyDrive/TAIX-Ray/`.
4. If Hugging Face authentication is required, save a Colab secret named **`HF_TOKEN`**. The notebook reads it automatically if present; do **not** paste tokens into source cells or commit them.
5. On the **first run**, keep `REBUILD_METADATA_CACHE = False`: the notebook will stream the raw metadata from Hugging Face and create `taix_metadata_eda.parquet` unless it already exists.
6. To download the image sample on first use, set **`DOWNLOAD_MISSING_IMAGES = True`**. Once the 900 PNG files exist, set it back to `False` to avoid repeated remote scans. The notebook reuses the manifest and skips already downloaded files.
7. Run **Runtime → Run all**. Later runs reuse the Parquet/CSV/image caches and do not require HF streaming if everything is already present.
8. For a fresh decoding audit, set `VERIFY_ALL_SAMPLE_PNGS = True` (optional; this checks all 900 local files and may take several minutes on Drive).

If you only need **metadata EDA**, leave `DOWNLOAD_MISSING_IMAGES = False`; the notebook will report missing image files and skip the image-dependent analyses, rather than failing.

## Run locally (macOS, Linux, Windows)

From the repository root, create a Python environment (example for macOS/Linux):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-eda.txt
jupyter lab
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

If `jupyter lab` is not installed, install `jupyterlab` separately (`python -m pip install jupyterlab`) or open the notebook in VS Code with the Jupyter extension.

The default local data directory is `./data/TAIX-Ray/`, **relative to the Jupyter working directory**. For consistent behavior regardless of where the notebook launches, set an absolute `TAIX_DATA_DIR` environment variable **before starting Jupyter**:

```bash
export TAIX_DATA_DIR="$HOME/datasets/TAIX-Ray"  # macOS / Linux
```

```powershell
$env:TAIX_DATA_DIR = "C:\data\TAIX-Ray"  # Windows PowerShell
```

Then open `01_data_understanding_clean.ipynb`, select the project's Python kernel, and run all cells. Set `USE_GOOGLE_DRIVE = False` (this is already the notebook's automatic default outside Colab).

For optional Hugging Face authentication, set `HF_TOKEN` in the environment or log in using the Hugging Face CLI. **Never store it in the notebook or Git.**

## Files and caching

The notebook automatically creates this directory tree under `TAIX_DATA_DIR` (or the appropriate default):

```text
TAIX-Ray/
├── taix_metadata_eda.parquet         # original 16 metadata columns; all 215,381 rows
├── eda_image_sample_manifest.csv     # stable UID, Split, PatientID sample selection
├── eda_image_sample/                 # downloaded UID.png files (up to 900)
├── eda_image_characteristics.csv     # cached image geometry + PNG format/mode
├── eda_pixel_characteristics.csv     # cached per-image pixel summary statistics
└── eda_outputs/
    └── figures/                      # generated plots (PNG)
```

**Cache behavior:**

- `taix_metadata_eda.parquet`: reused unless missing or `REBUILD_METADATA_CACHE = True`.
- `eda_image_sample_manifest.csv`: reused if it already exists; otherwise regenerated from metadata with `random_state=0`, 300 rows per split.
- `eda_image_sample/`: image files are only downloaded with `DOWNLOAD_MISSING_IMAGES = True` and existing files are not redownloaded.
- `eda_image_characteristics.csv` / `eda_pixel_characteristics.csv`: reused if their UIDs match the manifest; otherwise generated from local images (or forced with `REBUILD_IMAGE_STATS = True`).
- `eda_outputs/figures/`: written when `SAVE_FIGURES = True`.

**To reproduce exactly the same 900-image selection**, use the same raw dataset revision and the same manifest. If a different manifest is supplied, the notebook will use that manifest and check its UIDs and split assignments against metadata. Optionally set `DATASET_REVISION` in the configuration cell to a fixed Hugging Face revision/commit hash for a long-term archival run.

> **Important:** the notebook's optional verification cell is off by default to save time. The earlier team run decoded 900/900 files successfully, but colleagues must enable the cell if they want to independently confirm that result.

## Scope, methodology, and limits

- The **eight original severity columns are ordinal**. Grade 1 means borderline (`HeartSize`) or questionable (other findings). EDA's exploratory five-finding positives (`grade >= 2`, bilateral OR) are **not finalized training labels**.
- Age summaries are computed **per image** and **per unique patient** (earliest recorded image), so their denominators differ.
- Dates are de-identified/shifted. Year trends, exact years near dataset boundaries, and associations with acquisition technology are approximate.
- The shift in numerical pixel ranges around 2015–2017 aligns with a reported change in imaging technology, but **does not establish that the intensity ranges uniquely identify detector models**.
- The per-image 1st–99th percentile normalization in the six-image visualization is **display-only**, not final EVA-X/DenseNet training preprocessing.
- Valid grade codes and PNG decoding do **not** prove the clinical correctness of annotations or image quality across the complete dataset.
- The notebook uses the **official train/validation/test splits**, not random new patient splits, and must not be used to select a model based on test-set results.

## GitHub hygiene

**Commit:** the cleaned `.ipynb`, this README section, `requirements-eda.txt`, and optionally lightweight, non-sensitive aggregate plots or the deterministic manifest when agreed by the team.

**Do not commit:** the 900 raw X-rays, Hugging Face tokens, large caches, downloaded model weights, or patient-level exports. Add the following to the repository's `.gitignore` (adjust if your repo deliberately tracks a non-data file in `data/`):

```gitignore
.venv/
.ipynb_checkpoints/
__pycache__/
.env
.env.*
data/
**/eda_image_sample/
**/eda_outputs/
**/taix_metadata_eda.parquet
**/eda_image_characteristics.csv
**/eda_pixel_characteristics.csv
```

When sharing summary plots of patient images, follow the dataset's applicable license and institutional privacy policy. Do not upload patient-level data to GitHub merely because identifiers are pseudonymized.

## References

- [Truhn et al., TAIX-Ray, *Scientific Data* (2026)](https://www.nature.com/articles/s41597-026-07271-7)
- [TLAIM/TAIX-Ray on Hugging Face](https://huggingface.co/datasets/TLAIM/TAIX-Ray)
- [Hugging Face Datasets — streaming guide](https://huggingface.co/docs/datasets/stream)
- [Hugging Face Datasets — images and decoding](https://huggingface.co/docs/datasets/image_load)
