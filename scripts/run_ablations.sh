#!/bin/bash
################################################################################
# Run Ablation Experiments for Hybrid UNet variants
# Creates `final_experiments_v1/ablations_experiments/` and runs three experiments:
#  1) hybrid_full: ASPP + HINT (baseline hybrid)
#  2) hybrid_aspp_only: ASPP only
#  3) hybrid_hint_only: HINT only
#
# Usage:
#   bash run_ablations.sh [--quick] [--batch-size N] [--num-workers W] [--epochs E]
################################################################################

set -euo pipefail

# Defaults
EPOCHS=100
BATCH_SIZE=2
NUM_WORKERS=4
LR=3e-5
DROPOUT=0.1
SEED=42
QUICK=0

while [[ $# -gt 0 ]]; do
  case $1 in
    --quick)
      QUICK=1; shift ;;
    --epochs)
      EPOCHS="$2"; shift 2 ;;
    --batch-size)
      BATCH_SIZE="$2"; shift 2 ;;
    --num-workers)
      NUM_WORKERS="$2"; shift 2 ;;
    --lr)
      LR="$2"; shift 2 ;;
    --dropout)
      DROPOUT="$2"; shift 2 ;;
    --seed)
      SEED="$2"; shift 2 ;;
    -h|--help)
      sed -n '1,180p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ "$QUICK" -eq 1 ]]; then
  echo "Quick mode enabled: overriding epochs -> 2"
  EPOCHS=2
fi

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${BASE_DIR}/../nnUNet_raw/Dataset001_AISD"
ABLATIONS_ROOT="${BASE_DIR}/../final_experiments_v1/ablations_experiments"
SCRIPTS_DIR="${BASE_DIR}/scripts"
VENV_PYTHON="${BASE_DIR}/../venv/bin/python"

mkdir -p "${ABLATIONS_ROOT}"

# Export allocator hint
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# Define experiments
declare -A exps
# We run three ablations only: baseline UNet3+ (no ASPP, no HINT), ASPP-only, HINT-only
exps[unet3p_baseline]="--no_use_aspp --no_use_hint"
exps[unet3p_aspp_only]="--use_aspp --no_use_hint"
exps[unet3p_hint_only]="--no_use_aspp --use_hint"

for name in "unet3p_baseline" "unet3p_hint_only" "unet3p_aspp_only"; do
  FLAGS=${exps[$name]}
  EXP_DIR="${ABLATIONS_ROOT}/${name}"
  mkdir -p "${EXP_DIR}"
  LOG_TRAIN="${EXP_DIR}/train.log"
  LOG_EVAL="${EXP_DIR}/eval.log"

  echo "\n=== Running ${name} ==="
  echo "Save dir: ${EXP_DIR}"
  echo "Flags: ${FLAGS}"

  # Train (force --no_compile to avoid Inductor crashes)
  "${VENV_PYTHON}" "${SCRIPTS_DIR}/train.py" \
    --model_name hybrid_unet \
    --data_dir "${DATA_DIR}" \
    --save_dir "${EXP_DIR}" \
    --lr ${LR} \
    --dropout ${DROPOUT} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    ${FLAGS} \
    --no_compile \
    --val_split 0.15 \
    --num_workers ${NUM_WORKERS} \
    --seed ${SEED} 2>&1 | tee "${LOG_TRAIN}"

  # Locate checkpoint
  if [[ -f "${EXP_DIR}/best_model.pth" ]]; then
    CHECKPOINT="${EXP_DIR}/best_model.pth"
  elif [[ -f "${EXP_DIR}/final_model.pth" ]]; then
    CHECKPOINT="${EXP_DIR}/final_model.pth"
  else
    echo "No checkpoint found for ${name}, check ${LOG_TRAIN}"; continue
  fi

  # Evaluate
  "${VENV_PYTHON}" "${SCRIPTS_DIR}/evaluate.py" \
    --checkpoint "${CHECKPOINT}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${EXP_DIR}/evaluation" \
    --model_name hybrid_unet \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --visualize_every 10 \
    2>&1 | tee "${LOG_EVAL}"

  echo "Finished ${name}. Results at ${EXP_DIR}"
done

# Summary
echo "\nAll ablation experiments finished. Results base: ${ABLATIONS_ROOT}"

exit 0
