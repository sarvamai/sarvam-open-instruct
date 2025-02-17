export HF_HOME=/data/.cache/huggingface
export WANDB_CACHE_DIR=/data/.cache/wandb
export TRITON_CACHE_DIR=/data/.cache/triton
export RAY_USE_MULTIPROCESSING_CPU_COUNT=1

pip install -e .

huggingface-cli whoami

CUDA_LAUNCH_BLOCKING=1 python open_instruct/ppo_vllm_thread_ray_gtrl.py \
    --dataset_mixer '{"sarvam/RLVR-Indic-MATH-GSM": 1.0, "sarvam/RLVR-Indic-FC": 1.0}' \
    --run_name rlvr_llamaseek_1M_SFT_8b_indic_gsm_math_fc_with_think \
    --dataset_train_splits train train \
    --max_token_length 8192 \
    --max_prompt_token_length 8192 \
    --response_length 8192 \
    --model_name_or_path /projects/data/tanay_sarvam_ai/llamaseek-8b-predistilled-deepseek-1M-sft \
    --reward_model_path /projects/data/tanay_sarvam_ai/llamaseek-8b-predistilled-deepseek-1M-sft \
    --non_stop_penalty \
    --stop_token eos \
    --temperature 1.0 \
    --ground_truths_key ground_truth \
    --sft_messages_key translated_messages \
    --learning_rate 3e-7 \
    --total_episodes 200000 \
    --penalty_reward_value -10.0 \
    --deepspeed_stage 3 \
    --per_device_train_batch_size 4 \
    --local_rollout_forward_batch_size 4 \
    --local_mini_batch_size 16 \
    --local_rollout_batch_size 16 \
    --actor_num_gpus_per_node 8 8 8 6 \
    --vllm_tensor_parallel_size 2 \
    --beta 0.05 \
    --apply_verifiable_reward true \
    --output_dir /data/open-instruct/checkpoints/rlvr_llamaseek_1M_SFT_8b_indic_gsm_math_fc_with_think \
    --seed 3 \
    --num_evals 3 \
    --save_freq 25 \
    --reward_model_multiplier 0.0 \
    --gradient_checkpointing \
    --save_value_model \
    --think_mode \
    --with_tracking
