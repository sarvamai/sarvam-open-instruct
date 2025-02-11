import os
import json
import re
import random
import argparse
from typing import List, Dict, Any
import pandas as pd
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

###############################################################################
# Verification logic                                                          #
###############################################################################

def verify_function_sample(model_output: str, ground_truth: List[Dict[str, Any]]) -> bool:
    """
    Verify if model output matches ground truth function calls using multiple
    extraction methods. Similar to math verification, we try multiple approaches
    to extract function calls.
    """
    # Clean the model output of any header tags
    model_output = re.sub(r'<\|start_header_id\|>.*?<\|end_header_id\|>\n?', '', model_output)
    
    # Parse ground truth if it's a string
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except json.JSONDecodeError:
            return False
    
    all_extracted_calls = []

    # Method 1: Extract from code blocks with ```
    code_blocks = re.findall(r"```(?:json)?\n?(.*?)```", model_output, re.DOTALL)
    for block in code_blocks:
        try:
            parsed = json.loads(block.strip())
            if isinstance(parsed, list):
                all_extracted_calls.append(parsed)
        except json.JSONDecodeError:
            continue

    # Method 2: Extract from <function_calls> tags
    function_calls_blocks = re.findall(
        r"<function_calls>\s*(.*?)\s*</function_calls>",
        model_output,
        re.DOTALL
    )
    for block in function_calls_blocks:
        # Remove any embedded code blocks if present
        block = re.sub(r"```(?:json)?\n?(.*?)```", r"\1", block, flags=re.DOTALL)
        try:
            parsed = json.loads(block.strip())
            if isinstance(parsed, list):
                all_extracted_calls.append(parsed)
        except json.JSONDecodeError:
            continue

    # Method 3: Look for direct JSON array in the text
    try:
        json_arrays = re.findall(r"\[\s*{.*?}\s*\]", model_output, re.DOTALL)
        for array in json_arrays:
            try:
                parsed = json.loads(array)
                if isinstance(parsed, list):
                    all_extracted_calls.append(parsed)
            except json.JSONDecodeError:
                continue
    except Exception:
        pass

    # Method 4: Try parsing the entire output as JSON (last resort)
    try:
        parsed = json.loads(model_output.strip())
        if isinstance(parsed, list):
            all_extracted_calls.append(parsed)
    except json.JSONDecodeError:
        pass

    # Compare each extracted call list with ground truth
    for calls in all_extracted_calls:
        if is_function_calls_equivalent(calls, ground_truth):
            return True

    return False


def is_function_calls_equivalent(
    actual_calls: List[Dict[str, Any]],
    expected_calls: List[Dict[str, Any]]
) -> bool:
    """
    Check if two lists of function calls are equivalent by comparing function
    names and arguments.
    """
    if len(actual_calls) != len(expected_calls):
        return False

    for actual, expected in zip(actual_calls, expected_calls):
        if not isinstance(actual, dict) or not isinstance(expected, dict):
            return False
        if "name" not in actual or "arguments" not in actual:
            return False
        if actual["name"] != expected["name"]:
            return False

        actual_args = actual["arguments"]
        expected_args = expected["arguments"]

        if not isinstance(actual_args, dict) or not isinstance(expected_args, dict):
            return False

        # Check if all expected arguments are present with correct values
        for key, value in expected_args.items():
            if key not in actual_args or actual_args[key] != value:
                return False

    return True

###############################################################################
# Main script                                                                 #
###############################################################################

def main():
    """
    Script that:
    1) Loads the entire 'sarvam/RLVR_function_calling' dataset from Hugging Face.
    2) Uses vLLM to generate outputs (in batches, without converting entire dataset to a list).
    3) Verifies model outputs against ground-truth function calls.
    4) Appends the model output and correctness ("is_correct") to the dataset.
    5) Saves the updated dataset to disk via `save_to_disk()`.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path or ID of the model to use for vLLM.")
    parser.add_argument("--batch_size", type=int, default=8192,
                        help="Batch size (number of prompts) to send to LLM in one go.")
    parser.add_argument("--output_dir", type=str, default="vllm_inference_dataset",
                        help="Directory to save the updated dataset.")
    args = parser.parse_args()

    # 1) Load the Hugging Face dataset
    print("Loading dataset 'sarvam/RLVR_function_calling' ...")
    ds = load_dataset("sarvam/RLVR_function_calling", split="train")  # entire dataset
    ds = ds.select(range(500))
    # 2) Initialize vLLM and tokenizer
    print(f"Initializing vLLM with model {args.model_path} ...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.96,  # Adjust if needed
        max_model_len=8192,
        trust_remote_code=True
    )
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    total_samples = len(ds)
    print(f"Total samples in dataset: {total_samples}")

    # Prepare containers to store model output and correctness
    # We'll fill these in by index as we generate in batches:
    model_outputs = [None] * total_samples
    correctness_flags = [None] * total_samples

    # 3) Generation in batches
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        max_tokens=1024,
    )

    print(f"Generating with batch size {args.batch_size} ...")

    for start_idx in range(0, total_samples, args.batch_size):
        end_idx = min(start_idx + args.batch_size, total_samples)
        batch_ds = ds.select(range(start_idx, end_idx))

        # Each item in "messages" is a list of dicts suitable for vLLM
        batch_prompts = batch_ds["original_messages"]
        
        # Apply chat template to each prompt
        processed_prompts = [
            tokenizer.apply_chat_template(prompt, tokenize=False)
            for prompt in batch_prompts
        ]

        batch_generations = llm.generate(
            prompts=processed_prompts,
            sampling_params=sampling_params
        )

        for i, generation in enumerate(batch_generations):
            full_model_output = generation.outputs[0].text if generation.outputs else ""
            sample_idx = start_idx + i

            ground_truth_str = batch_ds[i].get("ground_truth", "[]")
            if isinstance(ground_truth_str, str):
                try:
                    ground_truth = json.loads(ground_truth_str)
                except:
                    ground_truth = []
            else:
                ground_truth = ground_truth_str

            is_correct = verify_function_sample(full_model_output, ground_truth)

            model_outputs[sample_idx] = full_model_output
            correctness_flags[sample_idx] = is_correct

    # 4) Add the new columns to the dataset
    print("Adding 'model_output' and 'is_correct' columns to the dataset...")
    ds = ds.add_column("model_output", model_outputs)
    ds = ds.add_column("is_correct", correctness_flags)

    # 5) Save the updated dataset to disk
    print(f"Saving updated dataset to {args.output_dir} ...")
    ds.save_to_disk(args.output_dir)

    # Final stats
    total_correct = sum(flag for flag in correctness_flags if flag)
    accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0.0
    print(f"\nDone. Accuracy: {accuracy:.2f}% [{total_correct}/{total_samples}]")
    print(f"Dataset with new columns saved at '{args.output_dir}'.")

if __name__ == "__main__":
    main()