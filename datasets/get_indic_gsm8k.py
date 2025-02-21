import csv
import os
from time import sleep
from datasets import load_dataset, Dataset, concatenate_datasets
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from indic_translation_utils import init_model, generate_translation
# -------------------------------
# Set up your generation settings
# -------------------------------

TOTAL_SAMPLES_INDIC = 20000
TOTAL_SAMPLES_ENGLISH = 2000

all_configs = [
    {
        'lang_short': 'ta',
        'lang': 'Tamil',
        'num_samples': int(TOTAL_SAMPLES_INDIC * 0.08)
    },
    {
        'lang_short': 'hi',
        'lang': 'Hindi',
        'num_samples': int(TOTAL_SAMPLES_INDIC * 0.28)
    },
    {
        'lang_short': 'mr',
        'lang': 'Marathi',
        'num_samples': int(TOTAL_SAMPLES_INDIC * 0.08)
    },
    {
        'lang_short': 'gu',
        'lang': 'Gujarati',
        'num_samples': int(TOTAL_SAMPLES_INDIC * 0.08)
    },
    {
        'lang_short': 'bn',
        'lang': 'Bengali',
        'num_samples': int(TOTAL_SAMPLES_INDIC * 0.08)
    },
    {
        'lang_short': 'ml',
        'lang': 'Malayalam',
        'num_samples': int(TOTAL_SAMPLES_INDIC * 0.08)
    },
    {
        'lang_short': 'kn',
        'lang': 'Kannada',
        'num_samples': int(TOTAL_SAMPLES_INDIC * 0.08)
    },
    {
        'lang_short': 'or',
        'lang': 'Odia',
        'num_samples': int(TOTAL_SAMPLES_INDIC * 0.08)
    },
    {
        'lang_short': 'pa',
        'lang': 'Punjabi',
        'num_samples': int(TOTAL_SAMPLES_INDIC * 0.08)
    },
    {
        'lang_short': 'te',
        'lang': 'Telugu',
        'num_samples': int(TOTAL_SAMPLES_INDIC * 0.08)
    },
]

# -----------------------------------------------------
# Worker function: process a single row
# -----------------------------------------------------
def process_row(row, lang, lang_short):
    
    input_text = "Question: " + row["question"]
    
    if lang == "English":
        prompt = f"The response should be in English. The response should end with the final answer (strictly in Arabic numerals and not in other scripts like devanagari) separated by #### from the rest of the response. For example, if the answer is 123, the response should end with '#### 123'. "
        translated_text = prompt + input_text
    else:
        model = init_model(lang, lang_short)

        translated_input = generate_translation(model, input_text)

        translated_input = translated_input.replace("<original_text>", "").replace("</original_text>", "").strip().strip("\"")

        if "####" in translated_input:
            # Count occurrences of ####
            count = translated_input.count("####")
            if count == 1:
                translated_input = translated_input.split("####")[0].strip()
            elif count > 1:
                # Split by last occurrence of ####
                translated_input = "####".join(translated_input.split("####")[:-1]).strip()

        translated_text = translated_input

    translated_messages = [{"role": "user", "content": translated_text}]

    try:
        ground_truth = row["answer"].split("####")[1].strip()
    except:
        ground_truth = None

    return {
        "question": row["question"],
        "translated_messages": translated_messages,
        "ground_truth": ground_truth,
        "dataset": "gsm8k",
        "language": lang,
    }

# -----------------------
# Main processing script
# -----------------------
def main():

    dataset = load_dataset("openai/gsm8k", "main")
    all_results = []

    for config in all_configs:
        sampled_dataset = dataset.shuffle()["train"].select(range(config['num_samples']))
        df = sampled_dataset.to_pandas()
        
        rows_to_process = [row for _, row in df.iterrows()]
        
        total_rows = len(rows_to_process)
        results = []

        prompt = f"The response should be in {config['lang']}. The response should end with the final answer (strictly in Arabic numerals and not in other scripts like devanagari) separated by #### from the rest of the response. For example, if the answer is 123, the response should end with '#### 123'."
        model = init_model(config['lang'], config['lang_short'])
        translated_prompt = generate_translation(model, prompt)
        translated_prompt = translated_prompt.replace("<original_text>", "").replace("</original_text>", "").strip().strip("\"")

    
        # Set up a ThreadPoolExecutor. Adjust max_workers based on your needs and system.
        with ThreadPoolExecutor(max_workers=50) as executor:
            # Submit all translation tasks concurrently.
            futures = {executor.submit(process_row, row, config['lang'], config['lang_short']): row for row in rows_to_process}
            
            # Create progress bar
            with tqdm(total=total_rows, desc=f"Translating rows for {config['lang']}") as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        result["translated_messages"][0]["content"] = translated_prompt + " " + result["translated_messages"][0]["content"]
                        if result["ground_truth"] is not None:
                            results.append(result)
                            pbar.update(1)
                    except Exception as e:
                        print("Error processing a row:", e)
                        pbar.update(1)
        
        csv_file = f"local_data_new/translations_gsm8k_{config['lang']}.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["ground_truth", "question", "translated_messages", "dataset", "language"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"Translations saved to {csv_file}")

        all_results.extend(results)
        print(f"Processed {config['lang']} with {len(results)} samples")

        sleep(60)

    indic_dataset = Dataset.from_list(all_results)
    english_dataset = dataset.shuffle()["train"].select(range(TOTAL_SAMPLES_ENGLISH))
    # Add original_messages and translated_messages columns to english dataset
    english_dataset = english_dataset.map(
        lambda x: process_row(x, "English", "en")
    )
    
    columns_to_keep = ["ground_truth", "question", "translated_messages", "dataset", "language"]
    english_dataset = english_dataset.select_columns(columns_to_keep)
    
    indic_with_english = concatenate_datasets([indic_dataset, english_dataset])
    indic_with_english = indic_with_english.shuffle()
    indic_with_english.push_to_hub(
        f"sarvam/RLVR-Indic-GSM-w-Prompt",
        token=os.environ["HF_TOKEN"]
    )

if __name__ == "__main__":
    main()