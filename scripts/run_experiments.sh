#!/bin/bash
################################################################################
# Automated Experimental Pipeline for Ischemic Stroke Segmentation
# Thesis Final Experiments
#
# This script runs three sequential experiments:
#   1. U-Net (baseline)
#   2. Attention U-Net
#   3. Hybrid U-Net (SOTA: UNet3+ + ASPP + HINT)
#
# Usage: bash run_experiments.sh
################################################################################

set -e  # Exit on error

# Configuration
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$(cd "${BASE_DIR}/../nnUNet_raw/Dataset001_AISD" && pwd)"
RESULTS_DIR="${BASE_DIR}/results"
SCRIPTS_DIR="${BASE_DIR}/scripts"

# Training parameters
EPOCHS=100
BATCH_SIZE=8
NUM_WORKERS=4
VAL_SPLIT=0.15

# Create results directory
mkdir -p "${RESULTS_DIR}"

# Log file
MAIN_LOG="${RESULTS_DIR}/experiment_pipeline.log"
echo "Starting experimental pipeline at $(date)" | tee "${MAIN_LOG}"
echo "Base directory: ${BASE_DIR}" | tee -a "${MAIN_LOG}"
echo "Data directory: ${DATA_DIR}" | tee -a "${MAIN_LOG}"
echo "Results directory: ${RESULTS_DIR}" | tee -a "${MAIN_LOG}"
echo "================================================================" | tee -a "${MAIN_LOG}"

################################################################################
# Experiment 1: U-Net Baseline
################################################################################
echo "" | tee -a "${MAIN_LOG}"
echo "EXPERIMENT 1: U-Net Baseline" | tee -a "${MAIN_LOG}"
echo "----------------------------------------------------------------" | tee -a "${MAIN_LOG}"

EXP1_DIR="${RESULTS_DIR}/exp1_unet"
mkdir -p "${EXP1_DIR}"

echo "Training U-Net..." | tee -a "${MAIN_LOG}"
"${BASE_DIR}/../venv/bin/python" "${SCRIPTS_DIR}/train.py" \
    --model_name unet \
    --data_dir "${DATA_DIR}" \
    --save_dir "${EXP1_DIR}" \
    --lr 1e-4 \
    --dropout 0.0 \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --val_split ${VAL_SPLIT} \
    --num_workers ${NUM_WORKERS} \
    --seed 42 \
    2>&1 | tee "${EXP1_DIR}/train_output.log"

echo "Training completed. Evaluating..." | tee -a "${MAIN_LOG}"
"${BASE_DIR}/../venv/bin/python" "${SCRIPTS_DIR}/evaluate.py" \
    --checkpoint "${EXP1_DIR}/best_model.pth" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${EXP1_DIR}/evaluation" \
    --model_name unet \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --visualize_every 10 \
    | tee -a "${EXP1_DIR}/eval_output.log"

echo "Experiment 1 completed at $(date)" | tee -a "${MAIN_LOG}"

################################################################################
# Experiment 2: Attention U-Net
################################################################################
echo "" | tee -a "${MAIN_LOG}"
echo "EXPERIMENT 2: Attention U-Net" | tee -a "${MAIN_LOG}"
echo "----------------------------------------------------------------" | tee -a "${MAIN_LOG}"

EXP2_DIR="${RESULTS_DIR}/exp2_attention_unet"
mkdir -p "${EXP2_DIR}"

echo "Training Attention U-Net..." | tee -a "${MAIN_LOG}"
"${BASE_DIR}/../venv/bin/python" "${SCRIPTS_DIR}/train.py" \
    --model_name attention_unet \
    --data_dir "${DATA_DIR}" \
    --save_dir "${EXP2_DIR}" \
    --lr 1e-4 \
    --dropout 0.0 \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --val_split ${VAL_SPLIT} \
    --num_workers ${NUM_WORKERS} \
    --seed 42 \
    2>&1 | tee "${EXP2_DIR}/train_output.log"

echo "Training completed. Evaluating..." | tee -a "${MAIN_LOG}"
"${BASE_DIR}/../venv/bin/python" "${SCRIPTS_DIR}/evaluate.py" \
    --checkpoint "${EXP2_DIR}/best_model.pth" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${EXP2_DIR}/evaluation" \
    --model_name attention_unet \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --visualize_every 10 \
    | tee -a "${EXP2_DIR}/eval_output.log"

echo "Experiment 2 completed at $(date)" | tee -a "${MAIN_LOG}"

################################################################################
# Experiment 3: Hybrid U-Net (SOTA)
################################################################################
echo "" | tee -a "${MAIN_LOG}"
echo "EXPERIMENT 3: Hybrid U-Net (SOTA)" | tee -a "${MAIN_LOG}"
echo "----------------------------------------------------------------" | tee -a "${MAIN_LOG}"

EXP3_DIR="${RESULTS_DIR}/exp3_hybrid_unet"
mkdir -p "${EXP3_DIR}"

echo "Training Hybrid U-Net (UNet3+ + ASPP + HINT)..." | tee -a "${MAIN_LOG}"
"${BASE_DIR}/../venv/bin/python" "${SCRIPTS_DIR}/train.py" \
    --model_name hybrid_unet \
    --data_dir "${DATA_DIR}" \
    --save_dir "${EXP3_DIR}" \
    --lr 3e-5 \
    --dropout 0.1 \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --val_split ${VAL_SPLIT} \
    --num_workers ${NUM_WORKERS} \
    --seed 42 \
    2>&1 | tee "${EXP3_DIR}/train_output.log"

echo "Training completed. Evaluating..." | tee -a "${MAIN_LOG}"
"${BASE_DIR}/../venv/bin/python" "${SCRIPTS_DIR}/evaluate.py" \
    --checkpoint "${EXP3_DIR}/best_model.pth" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${EXP3_DIR}/evaluation" \
    --model_name hybrid_unet \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --visualize_every 10 \
    | tee -a "${EXP3_DIR}/eval_output.log"

echo "Experiment 3 completed at $(date)" | tee -a "${MAIN_LOG}"

################################################################################
# Summary
################################################################################
echo "" | tee -a "${MAIN_LOG}"
echo "================================================================" | tee -a "${MAIN_LOG}"
echo "ALL EXPERIMENTS COMPLETED" | tee -a "${MAIN_LOG}"
echo "================================================================" | tee -a "${MAIN_LOG}"
echo "Pipeline finished at $(date)" | tee -a "${MAIN_LOG}"
echo "" | tee -a "${MAIN_LOG}"
echo "Results summary:" | tee -a "${MAIN_LOG}"
echo "  Experiment 1 (U-Net):          ${EXP1_DIR}" | tee -a "${MAIN_LOG}"
echo "  Experiment 2 (Attention U-Net): ${EXP2_DIR}" | tee -a "${MAIN_LOG}"
echo "  Experiment 3 (Hybrid U-Net):    ${EXP3_DIR}" | tee -a "${MAIN_LOG}"
echo "" | tee -a "${MAIN_LOG}"
echo "To compare results, check:" | tee -a "${MAIN_LOG}"
echo "  - ${RESULTS_DIR}/exp*/evaluation/evaluation_results.json" | tee -a "${MAIN_LOG}"
echo "  - ${RESULTS_DIR}/exp*/evaluation/training_curves.png" | tee -a "${MAIN_LOG}"
echo "  - ${RESULTS_DIR}/exp*/evaluation/visualizations/" | tee -a "${MAIN_LOG}"

# Extract final metrics
echo "" | tee -a "${MAIN_LOG}"
echo "Quick Metrics Comparison:" | tee -a "${MAIN_LOG}"
echo "----------------------------------------------------------------" | tee -a "${MAIN_LOG}"

for exp_dir in "${RESULTS_DIR}"/exp*; do
    if [ -f "${exp_dir}/evaluation/evaluation_results.json" ]; then
        model_name=$(basename "${exp_dir}")
        dice=$("${BASE_DIR}/../venv/bin/python" -c "import json; print(f\"{json.load(open('${exp_dir}/evaluation/evaluation_results.json'))['metrics']['dice']:.4f}\")" 2>/dev/null || echo "N/A")
        hd95=$("${BASE_DIR}/../venv/bin/python" -c "import json; print(f\"{json.load(open('${exp_dir}/evaluation/evaluation_results.json'))['metrics']['hd95']:.4f}\")" 2>/dev/null || echo "N/A")
        echo "${model_name}: Dice=${dice}, HD95=${hd95}" | tee -a "${MAIN_LOG}"
    fi
done

echo "" | tee -a "${MAIN_LOG}"
echo "Full pipeline log saved to: ${MAIN_LOG}" | tee -a "${MAIN_LOG}"
