from collections import defaultdict
import copy
import httpx
import asyncio
import os
from datasets import load_dataset
from open_instruct.ground_truth_utils import verify_gsm8k_sample, verify_math_sample
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


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
            timeout=600,
        )
    try:
        model_output = response.json()["choices"][0]["message"]["content"]
    except:
        import pdb
        pdb.set_trace()

    return model_output


async def call_hosted_model_rate_limited(messages, model, model_url, llm_config, semaphore):
    async with semaphore:
        return await call_hosted_model(messages, model, model_url, llm_config)
    

async def call_openai_model(messages, model):
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return completion.choices[0].message.content
    

async def call_openai_model_rate_limited(messages, model, semaphore):
    async with semaphore:
        return await call_openai_model(messages, model)


async def evaluate_model(model, dataset, semaphore):
    tasks = []
    
    for i, data in enumerate(dataset):
        messages = data["translated_messages"]
        if ("gpt" in model["model"]) or ("o3" in model["model"]):
            task = asyncio.create_task(call_openai_model_rate_limited(messages, model["model"], semaphore))
        else:
            task = asyncio.create_task(call_hosted_model_rate_limited(messages, model["model"], model["model_url"], llm_config, semaphore))
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    verification_results = {}
    invalid_results = []
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            invalid_results.append(i)
            # print(f"Error for sample {i}: {result}")
            continue

        if dataset[i]["dataset"].lower() == "gsm8k":
            verification_result = verify_gsm8k_sample(result, dataset[i]["ground_truth"])
        elif dataset[i]["dataset"].lower() == "math":
            verification_result = verify_math_sample(result, dataset[i]["ground_truth"])
        
        if dataset[i]["language"] not in verification_results:
            verification_results[dataset[i]["language"]] = []
        verification_results[dataset[i]["language"]].append(verification_result)

    final_evaluation = {}

    for language, results in verification_results.items():
        final_evaluation[language] = {
            "accuracy": sum(results)/len(results),
            "num_samples": len(results)
        }

    return final_evaluation


async def main(models, dataset):

    tasks = []

    for model in models:
        semaphore = asyncio.Semaphore(100)
        task = asyncio.create_task(evaluate_model(model, dataset, semaphore))
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Error for model: {result}")
            continue
        print(f"Model: {models[i]['model']}\nFull results: {result}") # noqa

if __name__ == "__main__":

    llm_config = {
        "temperature": 0.1,
    }

    models = [
        {
            "model": "gpt-4o-mini",
            "model_url": "https://api.openai.com/v1/chat/completions",
        },
        {
            "model": "/home/tanay_sarvam_ai/open-instruct/checkpoints/rlvr_llamaseek_8b_indic_gsm_math_checkpoints/step_75",
            "model_url": "http://10.67.27.15:8078/v1/chat/completions",
        },
        {
            "model": "/home/tanay_sarvam_ai/open-instruct/checkpoints/rlvr_llamaseek_8b_indic_gsm_math_checkpoints/step_150",
            "model_url": "http://10.67.27.1:8086/v1/chat/completions",
        },
        {
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "model_url": "http://10.67.27.1:8082/v1/chat/completions",
        },
        {
            "model": "o3-mini",
            "model_url": "https://api.openai.com/v1/chat/completions",
        },
        {
            "model": "/home/tanay_sarvam_ai/llamaseek_8b_hf_model-1M-sft",
            "model_url": "http://10.67.27.2:8077/v1/chat/completions",
        },
        {
            "model": "meta-llama/Llama-3.1-70B-Instruct",
            "model_url": "http://10.67.27.16:8083/v1/chat/completions",
        },
    ]

    dataset = load_dataset("sarvam/RLVR-Indic-MATH-GSM-Easy-w-Lang").shuffle(seed=42)["train"]
    asyncio.run(main(models, dataset))

