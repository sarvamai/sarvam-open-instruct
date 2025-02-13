export HF_HOME=/data/.cache/huggingface
export WANDB_CACHE_DIR=/data/.cache/wandb
export TRITON_CACHE_DIR=/data/.cache/triton
export RAY_USE_MULTIPROCESSING_CPU_COUNT=1

pip install -e .

python open_instruct/grpo_vllm_thread_ray_gtrl.py \
    --output_dir /data/open-instruct/checkpoints/rlvr_llama3_8b_indic_gsm_math_grpo \
    --run_name rlvr_llama3_8b_indic_gsm_math_grpo \
    --dataset_mixer_list sarvam/RLVR-Indic-MATH-GSM 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list sarvam/RLVR-Indic-MATH-GSM 1.0 \
    --dataset_mixer_eval_list_splits test \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --response_length 2048 \
    --number_samples_per_prompt 4 \
    --model_name_or_path /data/Llama-3.1-8B-Instruct \
    --stop_strings '"</answer>"' \
    --non_stop_penalty False \
    --chat_template_name llama3.1-8b \
    --stop_token eos \
    --penalty_reward_value 0.0 \
    --temperature 0.7 \
    --ground_truths_key ground_truth \
    --sft_messages_key translated_messages \
    --learning_rate 3e-7 \
    --total_episodes 100000 \
    --deepspeed_stage 3 \
    --per_device_train_batch_size 6 \
    --local_rollout_forward_batch_size 6 \
    --local_mini_batch_size 24 \
    --local_rollout_batch_size 24 \
    --num_epochs 1 \
    --actor_num_gpus_per_node 6 \
    --vllm_tensor_parallel_size 2 \
    --beta 0.01 \
    --apply_verifiable_reward true \
    --seed 3 \
    --num_evals 100 \
    --save_freq 25 \
    --reward_model_multiplier 0.0 \
    --no_try_launch_beaker_eval_jobs \
    --gradient_checkpointing \
    # --with_tracking
