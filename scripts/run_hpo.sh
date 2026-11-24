#!/usr/bin/env bash
# Full three-stage HPO for both models (about 9 h on one RTX 5090). Run from the repository root:
#   PY=/path/to/python bash scripts/run_hpo.sh > outputs/hpo.log 2>&1
# Every step can be re-run: the stage-1 study lives in SQLite (outputs/hpo/<model>/study.db) and
# stage 2 skips runs that already have a summary.json, so an interruption costs at most one trial.
set -euo pipefail
PY=${PY:-python}
export PYTHONIOENCODING=utf-8 HF_DATASETS_DISABLE_PROGRESS_BARS=1
# Stage-3 factors used on 2026-11-25. DenseNet-121 needed x4.5 and x6 to bracket its peak.
declare -A LR_FACTORS=([densenet121]="0.5,2,3,4.5,6" [eva_x_tiny]="0.5,2,3")

for model in densenet121 eva_x_tiny; do
  echo "===== $(date '+%F %T') $model: stage 1 (TPE + Hyperband, 20 trials)"
  "$PY" scripts/hpo.py search --model "$model" --n-trials 20
  echo "===== $(date '+%F %T') $model: stage 2 (top 3, 6-epoch and full schedules)"
  "$PY" scripts/hpo.py refine --model "$model" --top-k 3
  echo "===== $(date '+%F %T') $model: stage 3 (local learning-rate check)"
  "$PY" scripts/hpo.py local --model "$model" --lr-factors "${LR_FACTORS[$model]}"
  echo "===== $(date '+%F %T') $model: report"
  "$PY" scripts/hpo.py report --model "$model"
done
echo "===== $(date '+%F %T') HPO pipeline finished"
