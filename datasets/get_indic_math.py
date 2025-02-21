import csv
from time import sleep
from datasets import load_dataset, Dataset, concatenate_datasets
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os

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
    
    question = row["messages"][0]["content"].split("Question:")[-1].strip()
    input_text = "Question: " + question
    
    if lang == "English":
        prompt = "The response should be in English. Return your final response as 'Final Answer: \\boxed{<answer>}', where <answer> is the number or mathematical expression of the solution. "
        translated_text = prompt + input_text
    else:
        model = init_model(lang, lang_short)

        translated_input = generate_translation(model, input_text)

        translated_text = translated_input.replace("<original_text>", "").replace("</original_text>", "").strip().strip("\"")

    translated_messages = [{"role": "user", "content": translated_text}]

    return {
        "question": question,
        "translated_messages": translated_messages,
        "ground_truth": row["ground_truth"],
        "dataset": "MATH",
        "language": lang,
    }

# -----------------------
# Main processing script
# -----------------------
def main():

    dataset = load_dataset("allenai/RLVR-MATH")
    all_results = []

    for config in all_configs:
        sampled_dataset = dataset.shuffle()["train"].select(range(config['num_samples']))
        df = sampled_dataset.to_pandas()
        
        rows_to_process = [row for _, row in df.iterrows()]
        
        total_rows = len(rows_to_process)
        results = []

        prompt = f"The response should be in {config['lang']}. Return your final response as 'Final Answer: \\boxed{{<answer>}}', where <answer> is the number or mathematical expression of the solution."
        model = init_model(config['lang'], config['lang_short'])
        translated_prompt = generate_translation(model, prompt)
        translated_prompt = translated_prompt.replace("<original_text>", "").replace("</original_text>", "").strip().strip("\"")
    
        # Set up a ThreadPoolExecutor. Adjust max_workers based on your needs and system.
        with ThreadPoolExecutor(max_workers=100) as executor:
            # Submit all translation tasks concurrently.
            futures = {executor.submit(process_row, row, config['lang'], config['lang_short']): row for row in rows_to_process}
            
            # Create progress bar
            with tqdm(total=total_rows, desc=f"Translating rows for {config['lang']}") as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        result["translated_messages"][0]["content"] = translated_prompt + " " + result["translated_messages"][0]["content"]
                        results.append(result)
                        pbar.update(1)
                    except Exception as e:
                        print("Error processing a row:", e)
                        pbar.update(1)
        
        csv_file = f"local_data_new/translations_math_{config['lang']}.csv"
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
        f"sarvam/RLVR-Indic-MATH-w-Prompt",
        token=os.environ["HF_TOKEN"]
    )

if __name__ == "__main__":
    main()