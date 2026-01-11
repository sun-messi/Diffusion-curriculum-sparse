#!/bin/bash
# Evaluate FID for CIFAR-10 model at multiple steps

CKPT_DIR="workdir/cifar10_uvit_small/default_20260111_002937"
CONFIG="configs/cifar10_uvit_small.py"
STEPS="50000 100000 150000 200000"

for step in $STEPS; do
    echo "=== Evaluating step $step ==="
    nnet_path="${CKPT_DIR}/ckpts/${step}.ckpt/nnet_ema.pth"

    if [ ! -f "$nnet_path" ]; then
        echo "Checkpoint not found: $nnet_path"
        continue
    fi

    python eval.py \
        --config=$CONFIG \
        --nnet_path=$nnet_path \
        --output_path="${CKPT_DIR}/eval_${step}.log"
done

echo "=== Done ==="
