import argparse
import pandas as pd
from vllm import LLM, SamplingParams
from datasets import load_dataset
from transformers import AutoTokenizer
from open_instruct.ground_truth_utils import verify_function_sample
import uuid
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Batch benchmark for function calling tasks')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the model checkpoint')
    parser.add_argument('--num_preds', type=int, default=1,
                      help='Number of predictions per sample')
    parser.add_argument('--temperature', type=float, default=0.1,
                      help='Sampling temperature')
    parser.add_argument('--max_tokens', type=int, default=4096,
                      help='Maximum number of tokens to generate')
    parser.add_argument('--iter', type=int, default=1,
                      help='Iteration number')
    parser.add_argument('--think_mode', type=bool, default=False,
                      help='Think mode')
    return parser.parse_args()

def check_answer(text: str, expected_answer: int, dataset_name: str) -> bool:
    if dataset_name.lower() == "function_calling":
        return verify_function_sample(text, expected_answer)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def main():
    args = parse_args()

    dataset_name = "sarvam/RLVR-Indic-FC" #
    
    # Load dataset
    ds = load_dataset(dataset_name).shuffle(seed=42)["test"]
    samples_df = ds.to_pandas()
    print(f"Loaded dataset with columns: {samples_df.columns}")

    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95,
        max_model_len=8192,
        trust_remote_code=True
    )

    # Prepare prompts
    all_prompts = []
    all_answers = []
    all_clean = []
    all_dataset = []
    sys_prompt = "You are a helpful assistant. Think deeply before answering the user's question."

    for _, sample in samples_df.iterrows():
        prompt = sample["translated_messages"][0]['content']
        if args.think_mode:
            prompt_complete = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            prompt_complete = [{"role": "user", "content": prompt}]

        prompt_message = tokenizer.apply_chat_template(prompt_complete, tokenize=False)
        
        all_clean.append(prompt_complete)
        all_prompts.append(prompt_message)
        all_answers.append(sample["ground_truth"])
        all_dataset.append(sample["dataset"])

    # Replicate prompts for multiple predictions
    all_prompts = all_prompts * args.num_preds
    all_answers = all_answers * args.num_preds
    all_clean = all_clean * args.num_preds
    all_dataset = all_dataset * args.num_preds
    # Generate responses
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    print(f"Generating {len(all_prompts)} responses...")
    generations = llm.generate(all_prompts, sampling_params=sampling_params, use_tqdm=True)

    # Process results
    results_df = pd.DataFrame(all_clean)
    results_df['prompt'] = all_prompts
    results_df['generations'] = [gen.outputs[0].text for gen in generations]
    results_df['answers'] = all_answers
    results_df['dataset'] = all_dataset
    results_df['check'] = results_df.apply(
        lambda x: check_answer(x['generations'], x['answers'], x['dataset']), 
        axis=1
    )

    # Calculate accuracy
    results_df['correct'] = results_df['check'].apply(lambda x: 1 if x else 0)
    mean_accuracy = results_df.groupby('prompt')['correct'].mean().mean()
    results_df.to_csv(f'results_{args.model_path.split("/")[-1]}_{args.num_preds}_{args.iter}.csv', index=False)
    print("Model: ", args.model_path.split("/")[-1])
    print("Think mode: ", args.think_mode)
    print(f"Number of FC samples: {len(results_df)}")
    print(f"FC mean accuracy: {mean_accuracy:.4f}")
    results_df.to_csv(f'benchmark_results/results_dataset_{dataset_name.split("/")[-1]}_model_{args.model_path.split("/")[-1]}_num_preds_{args.num_preds}_think_{str(args.think_mode).lower()}.csv', index=False)

if __name__ == "__main__":
    main()


