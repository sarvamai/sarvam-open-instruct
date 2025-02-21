export HF_HOME=/data/.cache/huggingface
export WANDB_CACHE_DIR=/data/.cache/wandb
export TRITON_CACHE_DIR=/data/.cache/triton
export RAY_USE_MULTIPROCESSING_CPU_COUNT=1

pip install -e .

python open_instruct/ppo_vllm_thread_ray_gtrl.py \
    --dataset_mixer '{"sarvam/RLVR-Indic-MATH-GSM-w-Prompt": 1.0}' \
    --run_name rlvr_mistral-24b-sft-phase2-equal-think_indic_gsm_math-w-prompt_no_think_output_2048 \
    --dataset_train_splits train \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --response_length 2048 \
    --model_name_or_path /projects/data/rahul_sarvam_ai/models/nemo_models/mistral-24b-sft-phase2-equal-think/hf_model \
    --reward_model_path /projects/data/rahul_sarvam_ai/models/nemo_models/mistral-24b-sft-phase2-equal-think/hf_model \
    --non_stop_penalty \
    --stop_token eos \
    --temperature 1.0 \
    --ground_truths_key ground_truth \
    --sft_messages_key translated_messages \
    --learning_rate 2e-7 \
    --total_episodes 100000 \
    --penalty_reward_value -10.0 \
    --deepspeed_stage 3 \
    --per_device_train_batch_size 6 \
    --local_rollout_forward_batch_size 6 \
    --local_mini_batch_size 24 \
    --local_rollout_batch_size 24 \
    --actor_num_gpus_per_node 8 8 8 6 \
    --vllm_tensor_parallel_size 2 \
    --beta 0.05 \
    --apply_verifiable_reward true \
    --output_dir /data/open-instruct/checkpoints/rlvr_mistral-24b-sft-phase2-equal-think_indic_gsm_math-w-prompt_no_think_output_2048 \
    --seed 3 \
    --num_evals 3 \
    --save_freq 25 \
    --reward_model_multiplier 0.0 \
    --gradient_checkpointing \
    --save_value_model \
    --with_tracking \
    # --think_mode \