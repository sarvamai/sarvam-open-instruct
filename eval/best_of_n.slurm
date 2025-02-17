#!/bin/bash

# Default values
MODEL_PATH="/home/tanay_sarvam_ai/open-instruct/checkpoints/rlvr_llamaseek_8b_indic_gsm_math_checkpoints/step_150"  # Replace with your model path
NUM_RUNS=16                   # Number of times to run the benchmark
NUM_PREDS=4                 # Number of predictions per sample
TEMPERATURE=0.9
MAX_TOKENS=4096

mkdir -p /projects/data/aashay_sarvam_ai/logs

# Submit jobs
for run in $(seq 1 $NUM_RUNS); do
    sbatch \
        --job-name="benchmark_${run}" \
        --output="/projects/data/aashay_sarvam_ai/logs/%j_%N.log" \
        --error="/projects/data/aashay_sarvam_ai/logs/%j_%N.err" \
        --ntasks-per-node=1 \
        --cpus-per-task=8 \
        --gres=gpu:1 \
        --gres-flags=enforce-binding \
        --partition=defq \
        --container-image=/projects/data/aashay_sarvam_ai/cuda+12.2-transformers+4.47.sqsh \
        --container-mounts=/projects/data/:/home:rw \
        --container-writable \
        --wrap="export HF_TOKEN='' && export HF_HOME='/home/aashay_sarvam_ai/hf_cache/'  && cd /home/aashay_sarvam_ai/sarvam-open-instruct && pip install -e . && python3 /home/aashay_sarvam_ai/sarvam-open-instruct/eval/batch-benchmark-math-gsm.py \
                --model_path ${MODEL_PATH} \
                --num_preds ${NUM_PREDS} --iter ${run}"
done

echo "Submitted ${NUM_RUNS} benchmark jobs"
