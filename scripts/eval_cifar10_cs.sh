#!/bin/bash
set -e
CONFIG_FILE="configs/cifar10_uvit_small_cs.py"
NUM_GPUS=6
TIMESTAMP="20260110_094434"
CHECKPOINT_DIR="/home/sunj11/Documents/U-ViT-fresh/workdir/cifar10_uvit_small_cs/default_${TIMESTAMP}/ckpts"
RESULTS_DIR="eval_results/cifar10_uvit_small_cs/${TIMESTAMP}"
SAMPLES_DIR="eval_samples/cifar10_uvit_small_cs/${TIMESTAMP}"
CHECKPOINTS=(20000 40000 60000 80000 100000 120000 140000 160000 180000 200000)

mkdir -p "$RESULTS_DIR" "$SAMPLES_DIR"
SUMMARY_FILE="$RESULTS_DIR/fid_summary.txt"
echo "FID Summary - $(date)" > "$SUMMARY_FILE"

extract_fid() { grep "fid=" "$1" | tail -1 | sed 's/.*fid=\([0-9.]*\).*/\1/'; }

for ckpt in "${CHECKPOINTS[@]}"; do
    echo "Evaluating checkpoint: ${ckpt}"
    CKPT_PATH="${CHECKPOINT_DIR}/${ckpt}.ckpt"
    if [ -f "${CKPT_PATH}/nnet_ema.pth" ]; then
        NNET_PATH="${CKPT_PATH}/nnet_ema.pth"; MODEL_TYPE="ema"
    else
        NNET_PATH="${CKPT_PATH}/nnet.pth"; MODEL_TYPE="nnet"
    fi
    SAMPLE_PATH="${SAMPLES_DIR}/${ckpt}_${MODEL_TYPE}/"
    LOG_FILE="${RESULTS_DIR}/eval_${ckpt}_${MODEL_TYPE}.log"
    mkdir -p "$SAMPLE_PATH"

    accelerate launch --multi_gpu --num_processes $NUM_GPUS --mixed_precision fp16 \
        eval.py --config="$CONFIG_FILE" --nnet_path="$NNET_PATH" \
        --config.sample.path="$SAMPLE_PATH" --config.sample.n_samples=5000 \
        --config.sample.mini_batch_size=1000 --output_path="$LOG_FILE"

    FID=$(extract_fid "$LOG_FILE")
    echo "FID: $FID"
    echo "${ckpt}_${MODEL_TYPE}: FID = $FID" >> "$SUMMARY_FILE"
done
echo "Done! Samples: $SAMPLES_DIR"
cat "$SUMMARY_FILE"
