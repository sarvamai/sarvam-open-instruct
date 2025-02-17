import random
from datasets import load_dataset, concatenate_datasets, Dataset
import copy
import httpx
import asyncio
import os
from open_instruct.ground_truth_utils import verify_gsm8k_sample, verify_math_sample


async def call_hosted_model(messages, model, model_url, llm_config):
    headers = {"Content-Type": "application/json"}
    payload = copy.deepcopy(llm_config)
    payload["stream"] = False
    payload["messages"] = messages
    payload["model"] = model
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url=model_url,
            headers=headers,
            json=payload,
            timeout=240,
        )
    try:
        model_output = response.json()["choices"][0]["message"]["content"]
    except:
        import pdb
        pdb.set_trace()

    return model_output


async def rate_limited_call_hosted_model(messages, model, model_url, llm_config, semaphore):
    async with semaphore:
        return await call_hosted_model(messages, model, model_url, llm_config)


async def main(model, dataset, dataset_name, semaphore):
    tasks = []
    
    for i, data in enumerate(dataset):

        messages = data["translated_messages"]
        task = asyncio.create_task(rate_limited_call_hosted_model(messages, model["model"], model["model_url"], llm_config, semaphore))
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    positive_samples = []
    negative_samples = []
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Error for sample {i}: {result}")
            continue

        if dataset_name == "gsm":
            verification_result = verify_gsm8k_sample(result, dataset[i]["ground_truth"])
        elif dataset_name == "math":
            verification_result = verify_math_sample(result, dataset[i]["ground_truth"])
        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}")

        if verification_result:
            positive_samples.append(dataset[i])
        else:
            negative_samples.append(dataset[i])

    return positive_samples, negative_samples
            


if __name__ == "__main__":

    llm_config = {
        "temperature": 0.1,
    }

    model = {
        "model": "meta-llama/Llama-3.1-70B-Instruct",
        "model_url": "http://10.67.27.8:8075/v1/chat/completions",
    }

    dataset = load_dataset("sarvam/RLVR-GSM-Indic").shuffle(seed=42)["train"]
    dataset_name = "gsm"

    semaphore = asyncio.Semaphore(500)

    positive_samples, negative_samples = asyncio.run(main(model, dataset, dataset_name, semaphore))

    random.shuffle(positive_samples)
    random.shuffle(negative_samples)
    positive_samples_dataset = Dataset.from_list(positive_samples[:int(0.25*len(negative_samples))])
    negative_samples_dataset = Dataset.from_list(negative_samples)
    # Save both positive and negative samples datasets

    positive_samples_dataset.save_to_disk("/data/open-instruct/datasets/pass_samples_gsm_math")
    negative_samples_dataset.save_to_disk("/data/open-instruct/datasets/fail_samples_gsm_math")

    if len(positive_samples_dataset) and len(negative_samples_dataset):
        combined_data = concatenate_datasets([positive_samples_dataset, negative_samples_dataset])
        combined_data.push_to_hub("sarvam/RLVR-Indic-MATH-GSM", token=os.environ["HF_TOKEN"])
    