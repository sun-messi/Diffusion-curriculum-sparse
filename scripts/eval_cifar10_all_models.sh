#!/bin/bash
# Batch evaluation script for all CIFAR-10 U-ViT models
# Generates 5000 samples from the final checkpoint (200000 steps) of each model

set -e  # Exit on error

# Get current timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Configuration
NUM_GPUS=6
N_SAMPLES=5000
MINI_BATCH_SIZE=1000
SAMPLE_STEPS=50
CHECKPOINT_STEP=200000

# Define models to evaluate
declare -A MODELS
MODELS["cifar10_uvit_small"]="/home/sunj11/Documents/U-ViT-fresh/workdir/cifar10_uvit_small/default_20260110_065028"
MODELS["cifar10_uvit_small_c"]="/home/sunj11/Documents/U-ViT-fresh/workdir/cifar10_uvit_small_c/default_20260110_125959"
MODELS["cifar10_uvit_small_cs"]="/home/sunj11/Documents/U-ViT-fresh/workdir/cifar10_uvit_small_cs/default_20260110_094434"
MODELS["cifar10_uvit_small_s"]="/home/sunj11/Documents/U-ViT-fresh/workdir/cifar10_uvit_small_s/default_20260110_193200"

# Output directories
BASE_RESULTS_DIR="eval_results"
BASE_SAMPLES_DIR="eval_samples"

# Summary file
SUMMARY_FILE="${BASE_RESULTS_DIR}/cifar10_fid_summary_${TIMESTAMP}.txt"
mkdir -p "$BASE_RESULTS_DIR"

echo "CIFAR-10 FID Evaluation Summary - $(date)" > "$SUMMARY_FILE"
echo "======================================" >> "$SUMMARY_FILE"
echo "N_SAMPLES: $N_SAMPLES" >> "$SUMMARY_FILE"
echo "SAMPLE_STEPS: $SAMPLE_STEPS" >> "$SUMMARY_FILE"
echo "CHECKPOINT_STEP: $CHECKPOINT_STEP" >> "$SUMMARY_FILE"
echo "======================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Function to extract FID score from log file
extract_fid() {
    local log_file=$1
    grep "fid=" "$log_file" | tail -1 | sed 's/.*fid=\([0-9.]*\).*/\1/'
}

# Record start time
START_TIME=$(date +%s)

# Evaluate each model
for model_name in "${!MODELS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Evaluating: ${model_name}"
    echo "=========================================="

    WORKDIR="${MODELS[$model_name]}"
    CONFIG_FILE="configs/${model_name}.py"
    CKPT_PATH="${WORKDIR}/ckpts/${CHECKPOINT_STEP}.ckpt"

    # Check for EMA model
    if [ -f "${CKPT_PATH}/nnet_ema.pth" ]; then
        NNET_PATH="${CKPT_PATH}/nnet_ema.pth"
        MODEL_TYPE="ema"
        echo "Using EMA model"
    elif [ -f "${CKPT_PATH}/nnet.pth" ]; then
        NNET_PATH="${CKPT_PATH}/nnet.pth"
        MODEL_TYPE="nnet"
        echo "Using standard model"
    else
        echo "ERROR: Checkpoint not found: $CKPT_PATH"
        echo "${model_name}: NOT FOUND" >> "$SUMMARY_FILE"
        continue
    fi

    # Output paths
    RESULTS_DIR="${BASE_RESULTS_DIR}/${model_name}/${TIMESTAMP}"
    SAMPLES_DIR="${BASE_SAMPLES_DIR}/${model_name}/${TIMESTAMP}/${CHECKPOINT_STEP}_${MODEL_TYPE}"
    LOG_FILE="${RESULTS_DIR}/eval_${CHECKPOINT_STEP}_${MODEL_TYPE}.log"

    mkdir -p "$RESULTS_DIR"
    mkdir -p "$SAMPLES_DIR"

    echo "  Config: $CONFIG_FILE"
    echo "  Checkpoint: $NNET_PATH"
    echo "  Samples: $SAMPLES_DIR"
    echo "  Log: $LOG_FILE"
    echo ""

    # Run evaluation
    source ~/anaconda3/etc/profile.d/conda.sh && conda activate UVIT && \
    accelerate launch --multi_gpu --num_processes $NUM_GPUS --mixed_precision fp16 \
        eval.py \
        --config="$CONFIG_FILE" \
        --nnet_path="$NNET_PATH" \
        --config.sample.path="$SAMPLES_DIR" \
        --config.sample.n_samples=$N_SAMPLES \
        --config.sample.mini_batch_size=$MINI_BATCH_SIZE \
        --config.sample.sample_steps=$SAMPLE_STEPS \
        --output_path="$LOG_FILE" || {
            echo "ERROR: Evaluation failed for ${model_name}"
            echo "${model_name}: FAILED" >> "$SUMMARY_FILE"
            continue
        }

    # Extract and display FID score
    FID_SCORE=$(extract_fid "$LOG_FILE")
    echo ""
    echo "${model_name} - FID Score: $FID_SCORE"
    echo "${model_name}: FID = $FID_SCORE" >> "$SUMMARY_FILE"
    echo ""
done

# Calculate total time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))
SECONDS=$((TOTAL_TIME % 60))

echo ""
echo "=========================================="
echo "All evaluations complete!"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "=========================================="
echo ""
echo "Summary:"
cat "$SUMMARY_FILE"
echo ""
echo "Summary saved to: $SUMMARY_FILE"
