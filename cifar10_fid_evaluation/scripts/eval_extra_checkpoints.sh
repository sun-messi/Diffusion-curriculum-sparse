#!/bin/bash
# Evaluate 90k, 110k, 130k, 150k checkpoints for all 4 models
set -e

source ~/anaconda3/etc/profile.d/conda.sh && conda activate UVIT

NUM_GPUS=6
CHECKPOINTS=(90000 110000 130000 150000)

declare -A TIMESTAMPS
TIMESTAMPS["cifar10_uvit_small"]="20260110_065028"
TIMESTAMPS["cifar10_uvit_small_c"]="20260110_125959"
TIMESTAMPS["cifar10_uvit_small_cs"]="20260110_094434"
TIMESTAMPS["cifar10_uvit_small_s"]="20260110_193200"

cd /home/sunj11/Documents/U-ViT-fresh

for model in cifar10_uvit_small cifar10_uvit_small_c cifar10_uvit_small_cs cifar10_uvit_small_s; do
    ts="${TIMESTAMPS[$model]}"

    for ckpt in "${CHECKPOINTS[@]}"; do
        NNET="workdir/${model}/default_${ts}/ckpts/${ckpt}.ckpt/nnet_ema.pth"
        SAMPLE="eval_samples/${model}/${ts}/${ckpt}_ema/"
        LOG="eval_results/${model}/${ts}/eval_${ckpt}_ema.log"

        mkdir -p "$SAMPLE" "$(dirname $LOG)"

        echo ""
        echo "=========================================="
        echo "Model: $model | Checkpoint: $ckpt"
        echo "=========================================="

        accelerate launch --multi_gpu --num_processes $NUM_GPUS --mixed_precision fp16 \
            eval.py \
            --config="configs/${model}.py" \
            --nnet_path="$NNET" \
            --config.sample.path="$SAMPLE" \
            --config.sample.n_samples=5000 \
            --config.sample.mini_batch_size=1000 \
            --output_path="$LOG"

        FID=$(grep "fid=" "$LOG" | tail -1 | sed 's/.*fid=\([0-9.]*\).*/\1/')
        echo "FID: $FID"
        echo "${ckpt}_ema: FID = $FID" >> "eval_results/${model}/${ts}/fid_summary.txt"
    done
done

echo ""
echo "=========================================="
echo "All evaluations complete!"
echo "=========================================="
