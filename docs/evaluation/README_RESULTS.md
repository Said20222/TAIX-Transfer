# Results — DenseNet-121 vs EVA-X-Tiny on the TAIX-Ray Test Set

**Research question:** does chest-X-ray-specific self-supervised pretraining (EVA-X) detect abnormalities
on unseen ICU radiographs better than conventional ImageNet transfer learning (DenseNet-121)?

**Short answer from the first runs:** yes, but only by a small margin. EVA-X-Tiny scores higher on all
13 AUROCs. The gain in mean AUROC is **+0.003** (95% CI +0.002 to +0.003, p < 0.001). The clearest
gains are for cardiomegaly (+0.007) and congestion (+0.004). For pleural effusion, opacity and
atelectasis, the confidence intervals of the left/right-combined findings include zero.

Evaluated on 2026-11-15. The test split was used once, after both checkpoints had been chosen on
validation.

## Setup

Both models use **the same data pipeline** (`scripts/common.py`), so the comparison isolates the
architecture and pretraining:

- **Targets:** 8 binary outputs, positive when grade ≥ 2 on each original ordinal column. The
  5 EDA findings combine left and right: the label is either side positive, and the score is the
  higher of the two.
- **Input:** each image is min-max scaled, zero-padded to a square and resized to 224 × 224.
- **Augmentation:** random rotation ±10°, shift ±5% and zoom 0.9–1.1. There is no horizontal flip,
  because the labels are left/right specific.
- **Selection:** the best epoch by mean validation AUROC, with early stopping after 5 epochs
  without improvement.

| | DenseNet-121 | EVA-X-Tiny |
|---|---|---|
| Architecture | CNN, 6.96M parameters | ViT-Ti/16 with SwiGLU + RoPE, 5.53M parameters |
| Pretraining | ImageNet-1k, supervised (torchvision) | 520k chest X-rays, masked image modelling ([MapleF/eva_x](https://huggingface.co/MapleF/eva_x), rev `35ddcd6d`) |
| Optimizer | AdamW, lr 1e-4, wd 1e-4, batch 128 | AdamW, lr 5e-4, wd 0.05, layer decay 0.55, drop path 0.2, batch 512 |
| Schedule | 500 warmup steps, cosine, max 20 epochs | 5 warmup epochs, cosine, max 30 epochs |
| Normalization | ImageNet mean/std (grey) | EVA-X CXR mean/std (0.492 / 0.285) |
| Best epoch (of run) | 6 (early-stopped at 11) | 28 (ran all 30) |
| Validation mean AUROC (8 / 5) | 0.850 / 0.837 | 0.852 / 0.840 |
| Training time on RTX 5090 | 19.5 min | 42.6 min |

The EVA-X recipe is the authors' ChestX-ray14 recipe for EVA-X-Ti (lr 1e-3 at batch 1024), with the
learning rate scaled linearly to batch 512.

## Test results

42,928 images from 9,545 patients. Each AUROC has a 95% CI from 1,000 patient-level bootstrap
resamples. Patients are resampled rather than images because many patients have several
radiographs. The difference is EVA-X-Tiny minus DenseNet-121, computed on the same resamples
(paired). Its p-value is two-sided.

| Metric | DenseNet-121 | EVA-X-Tiny | Difference | p |
|---|---|---|---|---|
| **Mean of 8 labels** | 0.848 (0.845–0.850) | **0.850** (0.848–0.853) | **+0.003** (+0.002 to +0.003) | < 0.001 |
| **Mean of 5 findings** | 0.834 (0.831–0.837) | **0.837** (0.834–0.840) | **+0.003** (+0.002 to +0.004) | < 0.001 |
| Cardiomegaly (HeartSize) | 0.861 (0.855–0.868) | 0.868 (0.862–0.874) | +0.007 (+0.005 to +0.008) | < 0.001 |
| Congestion | 0.784 (0.777–0.790) | 0.788 (0.782–0.794) | +0.004 (+0.002 to +0.006) | < 0.001 |
| Pleural effusion (either side) | 0.883 (0.879–0.888) | 0.884 (0.880–0.888) | +0.001 (−0.001 to +0.002) | 0.278 |
| Pulmonary opacity (either side) | 0.842 (0.836–0.848) | 0.843 (0.837–0.849) | +0.001 (−0.001 to +0.003) | 0.216 |
| Atelectasis (either side) | 0.801 (0.794–0.808) | 0.803 (0.796–0.810) | +0.002 (−0.001 to +0.004) | 0.214 |

<details>
<summary>All 8 side-specific labels</summary>

| Label | DenseNet-121 | EVA-X-Tiny | Difference | p |
|---|---|---|---|---|
| HeartSize | 0.861 (0.855–0.868) | 0.868 (0.862–0.874) | +0.007 (+0.005 to +0.008) | < 0.001 |
| PulmonaryCongestion | 0.784 (0.777–0.790) | 0.788 (0.782–0.794) | +0.004 (+0.002 to +0.006) | < 0.001 |
| PleuralEffusion_Right | 0.917 (0.913–0.920) | 0.918 (0.915–0.922) | +0.001 (+0.000 to +0.003) | 0.030 |
| PleuralEffusion_Left | 0.871 (0.866–0.875) | 0.871 (0.867–0.876) | +0.001 (−0.001 to +0.002) | 0.374 |
| PulmonaryOpacities_Right | 0.862 (0.856–0.867) | 0.864 (0.859–0.869) | +0.002 (+0.000 to +0.004) | 0.006 |
| PulmonaryOpacities_Left | 0.853 (0.847–0.859) | 0.856 (0.850–0.863) | +0.003 (+0.001 to +0.005) | 0.010 |
| Atelectasis_Right | 0.831 (0.826–0.837) | 0.833 (0.827–0.839) | +0.002 (−0.000 to +0.004) | 0.070 |
| Atelectasis_Left | 0.803 (0.796–0.809) | 0.804 (0.798–0.811) | +0.002 (−0.000 to +0.004) | 0.080 |

</details>

Validation and test scores are close for both models (for example 0.850 → 0.848 for DenseNet-121), so
neither checkpoint looks overfit to the validation split.

## Reading the result

- **The direction is consistent.** EVA-X-Tiny is ahead on every one of the 13 AUROCs, and the paired
  difference in mean AUROC excludes zero.
- **The effect is small.** +0.003 AUROC is well below what would change clinical usefulness. Most
  per-finding differences are within ±0.002 and not distinguishable from zero.
- **EVA-X does this with fewer parameters** (5.5M vs 7.0M) and did not overfit. Its validation loss
  kept falling for 30 epochs, while DenseNet-121 peaked at epoch 6. It trains about 25% faster per epoch
  (≈ 2,100 vs ≈ 1,700 img/s) but used more epochs, so the whole run took longer.

## Caveats before drawing conclusions

1. **One training run per model.** The bootstrap CI covers test-sample variation only, not
   run-to-run variation from the random seed. A difference of 0.003 could be of the same size as seed
   noise. Repeating each model with 3 seeds is the most important next step.
2. **Different recipes and budgets.** Each model uses its standard recipe, and they are not tuned
   equally. DenseNet-121 might gain from regularization (it overfit early); EVA-X-Tiny might gain
   from a longer schedule (it was still improving very slowly).
3. **Provisional label and preprocessing choices.** The grade ≥ 2 threshold, per-image min-max
   normalization and 224 px input are not final team decisions. A different policy could change the
   gap.
4. **The test split has now been used.** If the team changes the setup and re-tests, report that
   the test set was seen before, or decide the final configuration on validation only first.

## Reproduce

```bash
python scripts/train.py --model densenet121      # outputs/runs/densenet121_s224_<time>/
python scripts/train.py --model eva_x_tiny       # downloads the EVA-X-Tiny weights on first use
python scripts/evaluate_test.py outputs/runs/densenet121_s224_<time> outputs/runs/eva_x_tiny_s224_<time>
```

`evaluate_test.py` writes `outputs/test_eval/results.json`, `results.md` and per-image test predictions.
It refuses to overwrite an existing result unless `--force` is given. Use `--split val` for a dry run.
