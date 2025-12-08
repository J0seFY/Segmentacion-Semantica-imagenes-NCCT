#!/bin/bash
################################################################################
# Run Hybrid U-Net experiment (exp3_Hybrid_UnetV2)
# - Entrena y evalúa el modelo `hybrid_unet` guardando resultados en
#   final_experiments_v1/results/exp3_Hybrid_UnetV2
# - Incluye opciones para run rápido (--quick) y protecciones para OOM
#
# Usage:
#   bash run_hybrid_exp3.sh [--quick] [--epochs N] [--batch-size B] [--num-workers W]
################################################################################

set -euo pipefail

# Defaults (safe for GPU memory)
EPOCHS=100
BATCH_SIZE=2
NUM_WORKERS=4
LR=3e-5
DROPOUT=0.1
SEED=42
QUICK=0

# Parse args
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
      sed -n '1,120p' "$0"; exit 0 ;;
    *)
      echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ "$QUICK" -eq 1 ]]; then
  echo "Quick mode enabled: overriding epochs -> 2"
  EPOCHS=2
fi

# Paths
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${BASE_DIR}/../nnUNet_raw/Dataset001_AISD"
RESULTS_DIR="${BASE_DIR}/results/exp3_Hybrid_UnetV2"
SCRIPTS_DIR="${BASE_DIR}/scripts"
VENV_PYTHON="${BASE_DIR}/../venv/bin/python"

mkdir -p "${RESULTS_DIR}"
mkdir -p "${RESULTS_DIR}/evaluation"

LOG_TRAIN="${RESULTS_DIR}/hybrid_unet_training.log"
LOG_EVAL="${RESULTS_DIR}/hybrid_unet_eval.log"

# Environment hints to reduce CUDA fragmentation (PyTorch recommendation)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

echo "Starting Hybrid U-Net experiment (exp3_Hybrid_UnetV2) at $(date)"
echo "Results -> ${RESULTS_DIR}"

echo "Training parameters: epochs=${EPOCHS}, batch_size=${BATCH_SIZE}, num_workers=${NUM_WORKERS}, lr=${LR}, dropout=${DROPOUT}"

# Train
"${VENV_PYTHON}" "${SCRIPTS_DIR}/train.py" \
  --model_name hybrid_unet \
  --data_dir "${DATA_DIR}" \
  --save_dir "${RESULTS_DIR}" \
  --lr ${LR} \
  --dropout ${DROPOUT} \
  --batch_size ${BATCH_SIZE} \
  --epochs ${EPOCHS} \
  --no_compile \
  --val_split 0.15 \
  --num_workers ${NUM_WORKERS} \
  --seed ${SEED} 2>&1 | tee "${LOG_TRAIN}"

# Determine best checkpoint
if [[ -f "${RESULTS_DIR}/best_model.pth" ]]; then
  CHECKPOINT="${RESULTS_DIR}/best_model.pth"
elif [[ -f "${RESULTS_DIR}/final_model.pth" ]]; then
  CHECKPOINT="${RESULTS_DIR}/final_model.pth"
else
  echo "No checkpoint found in ${RESULTS_DIR}. Look into ${LOG_TRAIN}"; exit 1
fi

# Evaluate
"${VENV_PYTHON}" "${SCRIPTS_DIR}/evaluate.py" \
  --checkpoint "${CHECKPOINT}" \
  --data_dir "${DATA_DIR}" \
  --output_dir "${RESULTS_DIR}/evaluation" \
  --model_name hybrid_unet \
  --batch_size ${BATCH_SIZE} \
  --num_workers ${NUM_WORKERS} \
  --visualize_every 10 \
  2>&1 | tee "${LOG_EVAL}"

# Final message
echo "Experiment completed at $(date)"
echo "Logs: ${LOG_TRAIN}, ${LOG_EVAL}"
echo "Evaluation results: ${RESULTS_DIR}/evaluation"

exit 0
