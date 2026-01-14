#!/bin/bash
set -e
CONFIG_FILE="configs/cifar10_uvit_small.py"
NUM_GPUS=6
TIMESTAMP="20260111_002937"
CHECKPOINT_DIR="/home/sunj11/Documents/U-ViT-fresh/workdir/cifar10_uvit_small/default_${TIMESTAMP}/ckpts"
RESULTS_DIR="eval_results/cifar10_uvit_small/${TIMESTAMP}"
SAMPLES_DIR="eval_samples/cifar10_uvit_small/${TIMESTAMP}"
CHECKPOINTS=(40000 60000 200000)

mkdir -p "$RESULTS_DIR" "$SAMPLES_DIR"

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

    echo "Done: ${ckpt}"
done
echo "Done! Samples: $SAMPLES_DIR"
