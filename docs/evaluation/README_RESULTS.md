# Results — DenseNet-121 vs EVA-X-Tiny on the TAIX-Ray Test Set

**Research question:** does chest-X-ray-specific self-supervised pretraining (EVA-X) detect abnormalities
on unseen ICU radiographs better than conventional ImageNet transfer learning (DenseNet-121)?

**Answer: no, not once both models are tuned.**

- **With default recipes** (2026-11-15), EVA-X-Tiny was slightly ahead: +0.003 mean AUROC.
- **After hyperparameter optimisation on the validation split** (2026-11-25), DenseNet-121 is slightly
  ahead: **0.852 vs 0.850 mean AUROC** (difference −0.002, 95% CI −0.002 to −0.001), and **0.858 vs 0.856
  mean AUPRC**.
- **What changed:** tuning improved DenseNet-121 by +0.004 on test but left EVA-X-Tiny unchanged, because
  the EVA-X authors' recipe was already near its optimum.

Either way, the two models are within 0.003 of each other on every summary metric. That is far too small
to matter clinically. On this dataset, chest-X-ray pretraining at the Tiny scale gives no measurable
advantage over ImageNet pretraining.

**Metrics:** mean AUROC (primary) and mean AUPRC (secondary) over the 8 side-specific labels and the
5 EDA findings. AUPRC's chance level is the share of positives (prevalence), listed in the detail tables.

## Setup

Both models use **the same data pipeline** (`scripts/common.py`), so the comparison isolates the
architecture and pretraining:

- **Targets:** 8 binary outputs, positive when grade ≥ 2 on each original ordinal column. The 5 findings
  combine left and right: the label is either side positive, and the score is the higher of the two.
- **Input:** each image is min-max scaled, zero-padded to a square and resized to 224 × 224.
- **Augmentation:** random rotation ±10°, shift ±5% and zoom 0.9–1.1. No horizontal flip, because the
  labels are left/right specific.
- **Selection:** always on validation (best epoch by mean validation AUROC).

| | DenseNet-121 | EVA-X-Tiny |
|---|---|---|
| Architecture | CNN, 6.96M parameters | ViT-Ti/16 with SwiGLU + RoPE, 5.53M parameters |
| Pretraining | ImageNet-1k, supervised (torchvision) | 520k chest X-rays, masked image modelling ([MapleF/eva_x](https://huggingface.co/MapleF/eva_x), rev `35ddcd6d`) |
| **Default recipe** | AdamW lr 1e-4, wd 1e-4, batch 128, 20 epochs | Authors' recipe: AdamW lr 5e-4, layer decay 0.55, wd 0.05, drop path 0.2, batch 512, 30 epochs |
| **Tuned recipe** ([how](../hpo/README_HPO.md)) | lr 6.7e-4, wd 8.4e-4, dropout 0.13, batch 128, 6 epochs | lr 3.1e-3, layer decay 0.76, wd 0.0027, drop path 0.35, batch 512, 6 epochs |
| Validation mAUROC, default → tuned | 0.850 → 0.854 | 0.852 → 0.852 |
| Training time, default → tuned (RTX 5090) | 19.5 → 9.3 min | 42.6 → 7.2 min |

## Test results: summary

42,928 images from 9,545 patients. 95% CIs come from 1,000 patient-level bootstrap resamples. Patients are
resampled rather than images because many patients have several radiographs. Differences are paired
(same resamples for both models) and given as EVA-X-Tiny minus DenseNet-121, with two-sided bootstrap
p-values.

| Metric | Recipes | DenseNet-121 | EVA-X-Tiny | Difference (EVA-X minus DenseNet) | p |
|---|---|---|---|---|---|
| **Mean AUROC, 8 labels** | default | 0.848 (0.845–0.850) | 0.850 (0.848–0.853) | +0.003 (+0.002 to +0.003) | < 0.001 |
| | **tuned** | **0.852** (0.849–0.854) | 0.850 (0.848–0.853) | **−0.002** (−0.002 to −0.001) | < 0.001 |
| Mean AUROC, 5 findings | default | 0.834 (0.831–0.837) | 0.837 (0.834–0.840) | +0.003 (+0.002 to +0.004) | < 0.001 |
| | **tuned** | **0.838** (0.835–0.841) | 0.837 (0.834–0.839) | −0.001 (−0.002 to −0.001) | 0.002 |
| **Mean AUPRC, 8 labels** | default | 0.853 (0.849–0.858) | 0.857 (0.852–0.861) | +0.003 (+0.002 to +0.004) | < 0.001 |
| | **tuned** | **0.858** (0.853–0.862) | 0.856 (0.851–0.860) | **−0.002** (−0.003 to −0.001) | < 0.001 |
| Mean AUPRC, 5 findings | default | 0.876 (0.871–0.880) | 0.879 (0.875–0.883) | +0.003 (+0.002 to +0.004) | < 0.001 |
| | **tuned** | **0.880** (0.875–0.884) | 0.878 (0.874–0.882) | −0.001 (−0.002 to −0.001) | < 0.001 |

## Test results per finding (tuned models)

| Finding | Prevalence | AUROC DenseNet-121 | AUROC EVA-X-Tiny | Diff. | AUPRC DenseNet-121 | AUPRC EVA-X-Tiny | Diff. |
|---|---|---|---|---|---|---|---|
| Cardiomegaly | 0.523 | 0.866 | 0.868 | +0.002 | 0.872 | 0.873 | +0.001 |
| Congestion | 0.551 | 0.788 | 0.786 | −0.002 | 0.813 | 0.810 | −0.003 |
| Pleural effusion (either side) | 0.618 | 0.887 | 0.885 | −0.002 | 0.926 | 0.925 | −0.001 |
| Pulmonary opacity (either side) | 0.470 | 0.846 | 0.843 | −0.003 | 0.843 | 0.839 | −0.004 |
| Atelectasis (either side) | 0.825 | 0.803 | 0.801 | −0.002 | 0.945 | 0.945 | −0.000 |

EVA-X-Tiny keeps a small edge only for cardiomegaly (+0.002 AUROC, CI +0.000 to +0.003). That is also the
finding where it led most clearly before tuning (+0.007). DenseNet-121 is ahead for opacity, effusion,
congestion and atelectasis.

<details>
<summary>All 8 side-specific labels (tuned models, AUROC and AUPRC with 95% CIs)</summary>

| Label | AUROC DenseNet-121 | AUROC EVA-X-Tiny | Diff. (95% CI) | p |
|---|---|---|---|---|
| HeartSize | 0.866 (0.860–0.872) | 0.868 (0.862–0.874) | +0.002 (+0.000 to +0.003) | 0.024 |
| PulmonaryCongestion | 0.788 (0.781–0.794) | 0.786 (0.780–0.793) | −0.002 (−0.003 to +0.000) | 0.092 |
| PleuralEffusion_Right | 0.920 (0.916–0.923) | 0.919 (0.915–0.922) | −0.001 (−0.002 to −0.000) | 0.030 |
| PleuralEffusion_Left | 0.874 (0.870–0.879) | 0.873 (0.868–0.877) | −0.002 (−0.003 to −0.000) | 0.006 |
| PulmonaryOpacities_Right | 0.867 (0.862–0.872) | 0.864 (0.859–0.869) | −0.003 (−0.004 to −0.001) | < 0.001 |
| PulmonaryOpacities_Left | 0.858 (0.852–0.865) | 0.855 (0.849–0.862) | −0.003 (−0.005 to −0.001) | < 0.001 |
| Atelectasis_Right | 0.835 (0.829–0.841) | 0.833 (0.827–0.838) | −0.002 (−0.004 to −0.001) | 0.002 |
| Atelectasis_Left | 0.806 (0.800–0.812) | 0.804 (0.798–0.810) | −0.002 (−0.004 to −0.000) | 0.032 |

| Label | Prevalence | AUPRC DenseNet-121 | AUPRC EVA-X-Tiny | Diff. (95% CI) | p |
|---|---|---|---|---|---|
| HeartSize | 0.523 | 0.872 (0.863–0.880) | 0.873 (0.864–0.881) | +0.001 (−0.001 to +0.002) | 0.348 |
| PulmonaryCongestion | 0.551 | 0.813 (0.804–0.822) | 0.810 (0.800–0.819) | −0.003 (−0.005 to −0.000) | 0.028 |
| PleuralEffusion_Right | 0.387 | 0.890 (0.883–0.896) | 0.889 (0.882–0.895) | −0.001 (−0.002 to +0.001) | 0.184 |
| PleuralEffusion_Left | 0.505 | 0.874 (0.867–0.880) | 0.872 (0.865–0.878) | −0.002 (−0.003 to −0.000) | 0.018 |
| PulmonaryOpacities_Right | 0.409 | 0.834 (0.824–0.844) | 0.829 (0.818–0.839) | −0.005 (−0.007 to −0.002) | < 0.001 |
| PulmonaryOpacities_Left | 0.328 | 0.777 (0.763–0.792) | 0.772 (0.758–0.787) | −0.006 (−0.008 to −0.003) | < 0.001 |
| Atelectasis_Right | 0.669 | 0.907 (0.901–0.912) | 0.905 (0.899–0.910) | −0.002 (−0.003 to −0.000) | 0.026 |
| Atelectasis_Left | 0.704 | 0.899 (0.893–0.904) | 0.898 (0.892–0.904) | −0.000 (−0.002 to +0.001) | 0.660 |

</details>

## Reading the result

- **Tuning, not pretraining, explained the earlier gap.** The default DenseNet-121 recipe (lr 1e-4,
  20 epochs) was too timid on the learning rate and too long: the model overfit after epoch 6. With a
  higher learning rate and a 6-epoch schedule, DenseNet-121 gains 0.004–0.005 on both metrics. EVA-X-Tiny gains
  nothing, because its authors had already tuned their recipe.
- **AUPRC agrees with AUROC** in every summary row, so the conclusion doesn't depend on the metric.
  AUPRC is lowest where prevalence is lowest (left-sided opacity: 0.33 positives, AUPRC 0.77), as expected.
- **Validation predicted test well.** For every model, test mAUROC is 0.002 below validation (0.0018–0.0023),
  so the shift is uniform, and the tuned ranking (DenseNet-121 ahead) is the same on both splits. So there are no signs that the
  search overfit the validation split.
- **EVA-X-Tiny is cheaper to train.** The tuned EVA-X-Tiny trains in 7 minutes with fewer parameters
  (5.5M vs 7.0M), but that is an efficiency point, not an accuracy one.

## Caveats

1. **One training run per configuration.** The bootstrap CIs cover test-sample variation only, not
   seed-to-seed variation. A gap of 0.002 may be of the same size as seed noise. Repeating each tuned
   configuration with 3 seeds is the most useful next step.
2. **Provisional label and preprocessing choices.** The grade ≥ 2 threshold, per-image min-max normalization
   and 224 px input are not final team decisions.
3. **The test split has now been used twice:** once for the default recipes (2026-11-15) and once for the
   tuned ones (2026-11-25). The hyperparameter search used validation only and never saw test results, so
   the tuned comparison is still a fair held-out evaluation. Any further change should again be decided on
   validation first.
4. **Tiny scale only.** EVA-X-Small or -Base (22M / 86M parameters) might benefit more from chest-X-ray
   pretraining than the 6M Tiny model does.

## Reproduce

```bash
# default recipes
python scripts/train.py --model densenet121
python scripts/train.py --model eva_x_tiny
python scripts/evaluate_test.py outputs/runs/densenet121_s224_<time> outputs/runs/eva_x_tiny_s224_<time>

# hyperparameter search (docs/hpo/README_HPO.md), then the tuned comparison
PY=python bash scripts/run_hpo.sh > outputs/hpo.log 2>&1
python scripts/evaluate_test.py outputs/runs/hpo_densenet121_trial016_lrx2 outputs/runs/hpo_eva_x_tiny_trial016_lrx2 \
    --out-dir outputs/test_eval_tuned
```

`evaluate_test.py` writes `results.json`, `results.md` and per-image predictions. It refuses to overwrite an
existing result unless `--force` is given. Use `--split val` for a dry run.
