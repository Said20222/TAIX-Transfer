# TAIX-Transfer

**Comparing Conventional and Self-Supervised Chest X-Ray Models for ICU Abnormality Detection**

DS 504 — Data Mining and Decision Support

## Project Overview

This project investigates how conventional ImageNet-based transfer learning compares with chest-X-ray-specific self-supervised pretraining for multi-label abnormality detection in ICU bedside radiographs.

Our primary research question is:

> Does chest-X-ray-specific self-supervised pretraining improve multi-label abnormality detection on previously unseen ICU bedside radiographs compared with conventional ImageNet transfer learning?

### Dataset

We use [TAIX-Ray](https://huggingface.co/datasets/TLAIM/TAIX-Ray), containing 215,381 chest radiographs from 47,724 patients.

The dataset provides eight ordinal radiographic findings and official patient-disjoint training, validation, and test splits.

### Models

- **DenseNet-121:** Conventional convolutional neural network initialized with ImageNet-pretrained weights.
- **EVA-X:** Vision Transformer pretrained using chest-X-ray-specific self-supervised learning.

The current comparison focuses on DenseNet-121 and EVA-X-Tiny. Final modeling and preprocessing configurations will be documented as development progresses.

## Project Structure

```text
TAIX-Transfer/
├── notebooks/
│   └── 01_data_understanding.ipynb
├── docs/
│   └── eda/
│       ├── README_EDA.md
│       └── TAIX-Ray_EDA_Part_II.pdf
├── src/
├── requirements-eda.txt
├── .gitignore
└── README.md
```

## Current Progress

**CRISP-DM Phase: Data Understanding — Completed**

The exploratory analysis covers:

- Dataset structure and patient-disjoint splits
- Severity distributions and multi-label relationships
- Demographic and temporal characteristics
- Image dimensions, pixel intensities, and acquisition-related variation
- Missing-value and integrity checks

### Reproducing the EDA

See [EDA Setup and Reproduction Instructions](docs/eda/README_EDA.md).

The notebook supports Google Colab and local Jupyter environments.

```bash
pip install -r requirements-eda.txt
```

The dataset images and generated caches are excluded from Git. Follow the EDA instructions to obtain the data and reproduce the analysis.

## Upcoming Work

- Finalize target-label and uncertainty-handling policies
- Finalize preprocessing and normalization
- Implement DenseNet-121 and EVA-X training pipelines
- Evaluate using patient-disjoint validation and test data
- Compare predictive performance and robustness

## References

- [TAIX-Ray dataset](https://huggingface.co/datasets/TLAIM/TAIX-Ray)
- [TAIX-Ray publication](https://www.nature.com/articles/s41597-026-07271-7)
- [EVA-X implementation](https://github.com/hustvl/EVA-X)
- [EVA-X publication](https://www.nature.com/articles/s41746-025-02032-z)

## Team

Course project developed collaboratively in a team of 3 students for DS 504 — Data Mining and Decision Support.

