import argparse
import pandas as pd
from vllm import LLM, SamplingParams
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer
from open_instruct.ground_truth_utils import verify_gsm8k_sample, verify_math_sample
import uuid
import os
import random


def parse_args():
    parser = argparse.ArgumentParser(description='Batch benchmark for math and GSM tasks')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the model checkpoint')
    parser.add_argument('--temperature', type=float, default=0.1,
                      help='Sampling temperature')
    parser.add_argument('--max_tokens', type=int, default=8192,
                      help='Maximum number of tokens to generate')
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

    gsm_dataset_name = "sarvam/RLVR-Indic-GSM-w-Prompt" #
    math_dataset_name = "sarvam/RLVR-Indic-MATH-w-Prompt"
    
    # Load dataset
    gsm_ds = load_dataset(gsm_dataset_name).shuffle()["train"]
    math_ds = load_dataset(math_dataset_name).shuffle()["train"]
    ds = concatenate_datasets([gsm_ds, math_ds])

    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.95,
        max_model_len=4096,
        trust_remote_code=True
    )

    # Prepare prompts
    all_prompts = []
    all_samples = []
    correct_samples = []
    incorrect_samples = []

    for sample in ds:

        prompt_message = tokenizer.apply_chat_template(sample["translated_messages"], tokenize=False)
        
        all_samples.append(sample)
        all_prompts.append(prompt_message)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    print(f"Generating {len(all_prompts)} responses...")
    generations = llm.generate(all_prompts, sampling_params=sampling_params, use_tqdm=True)
    generations = [gen.outputs[0].text for gen in generations]

    for sample, generation in zip(all_samples, generations):
        verification_result = check_answer(generation, sample['ground_truth'], sample['dataset'])
        if verification_result:
            correct_samples.append(sample)
        else:
            incorrect_samples.append(sample)


    random.shuffle(correct_samples)
    random.shuffle(incorrect_samples)

    positive_samples_dataset = Dataset.from_list(correct_samples[:int(0.25*len(incorrect_samples))])
    negative_samples_dataset = Dataset.from_list(incorrect_samples)

    positive_samples_dataset.save_to_disk("/data/open-instruct/datasets/pass_samples_gsm_math_with_prompts_200225")
    negative_samples_dataset.save_to_disk("/data/open-instruct/datasets/fail_samples_gsm_math_with_prompts_200225")

    if len(positive_samples_dataset) and len(negative_samples_dataset):
        combined_data = concatenate_datasets([positive_samples_dataset, negative_samples_dataset])
        combined_data = combined_data.train_test_split(test_size=0.1)
        combined_data.push_to_hub("sarvam/RLVR-Indic-MATH-GSM-w-Prompt", token=os.environ["HF_TOKEN"])

if __name__ == "__main__":
    main()



    

