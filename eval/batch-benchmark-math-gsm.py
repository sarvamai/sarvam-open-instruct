import argparse
import pandas as pd
from vllm import LLM, SamplingParams
from datasets import load_dataset
from transformers import AutoTokenizer
from open_instruct.ground_truth_utils import verify_gsm8k_sample, verify_math_sample
import uuid
import os

filepath = uuid.uuid4()


def parse_args():
    parser = argparse.ArgumentParser(description='Batch benchmark for math and GSM tasks')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the model checkpoint')
    parser.add_argument('--num_preds', type=int, default=8,
                      help='Number of predictions per sample')
    parser.add_argument('--temperature', type=float, default=0.9,
                      help='Sampling temperature')
    parser.add_argument('--max_tokens', type=int, default=4096,
                      help='Maximum number of tokens to generate')
    parser.add_argument('--iter', type=int, default=1,
                      help='Iteration number')
    return parser.parse_args()


def check_answer(text: str, expected_answer: int, dataset_name: str) -> bool:
    if dataset_name.lower() == "gsm8k":
        return verify_gsm8k_sample(text, expected_answer)
    elif dataset_name.lower() == "math":
        return verify_math_sample(text, expected_answer)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def main():
    args = parse_args()
    
    # Load dataset
    ds = load_dataset("sarvam/RLVR-Indic-MATH-GSM").shuffle(seed=42)["test"]
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
        prompt_complete = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]
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
    print(f"Mean accuracy: {mean_accuracy:.4f}")
    os.makedirs(f'data/results_{filepath}', exist_ok=True)
    results_df.to_csv(f'data/results_{filepath}/results_{args.model_path.split("/")[-1]}_{args.num_preds}_{args.iter}.csv', index=False)
if __name__ == "__main__":
    main()



    

