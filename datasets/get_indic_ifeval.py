import copy
import csv
import json
import os
from time import sleep
from datasets import load_dataset, Dataset, concatenate_datasets

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from indic_translation_utils import init_model, generate_translation
# -------------------------------
# Set up your generation settings
# -------------------------------

TOTAL_SAMPLES_INDIC = 50
TOTAL_SAMPLES_ENGLISH = 10

all_configs = [
    {
        'lang_short': 'hi',
        'lang': 'Hindi',
        'num_samples': int(TOTAL_SAMPLES_INDIC * 0.28)
    },
    {
        'lang_short': 'ta',
        'lang': 'Tamil',
        'num_samples': int(TOTAL_SAMPLES_INDIC * 0.08)
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

def postprocess_translation(translated_text):
    if "<original_text>" in translated_text:
        start_tag = "<original_text>"
        end_tag = "</original_text>"
        start_idx = translated_text.find(start_tag) + len(start_tag)
        end_idx = translated_text.find(end_tag)
        if start_idx != -1 and end_idx != -1:
            translated_text = translated_text[start_idx:end_idx]
    return translated_text


def process_row(row, lang, lang_short, keys_to_translate):
    # If the Vertex AI client isn't thread-safe, initialize the model inside each worker.
    model = init_model(lang, lang_short)
    input_text = row["messages"][0]["content"].strip(".") + f". Your response should be in {lang}."

    if isinstance(row["ground_truth"], str):
        gt_dict = json.loads(row["ground_truth"])
        str_flag = True
    else:
        gt_dict = row["ground_truth"]
        str_flag = False

    if lang == "English":
        translated_messages = [{"role": "user", "content": input_text}]

    else:
        for key in keys_to_translate:
            if key in gt_dict and gt_dict[key] is not None:
                if isinstance(gt_dict[key], list):
                    translated_list = []
                    for item in gt_dict[key]:
                        translated_key = generate_translation(model, item)
                        translated_key = postprocess_translation(translated_key)
                        translated_list.append(translated_key)
                    gt_dict[key] = translated_list
                else:
                    translated_key = generate_translation(model, gt_dict[key])
                    gt_dict[key] = postprocess_translation(translated_key)
        
        final_translated_text = ""
        for text in input_text.split(". "):
            if text.strip() == "":
                continue
            translated_text = generate_translation(model, text + ".")
            translated_text = postprocess_translation(translated_text)
            final_translated_text += translated_text + " "

        translated_messages = [{"role": "user", "content": final_translated_text.strip()}]

    return {
        "original_messages": row["messages"],
        "translated_messages": translated_messages,
        "ground_truth": json.dumps(gt_dict, ensure_ascii=False) if str_flag else gt_dict,
        "dataset": "ifeval",
        "language": lang,
        "constraint_type": row["constraint_type"],
        "constraint": row["constraint"]
    }

# -----------------------
# Main processing script
# -----------------------
def main():

    dataset = load_dataset("allenai/RLVR-IFeval")
    all_results = []

    keys_to_translate = [
            "end_phrase",
            "forbidden_words",
            "original_prompt",
        ]

    for i, config in enumerate(all_configs[:1]):
        
        dataset = dataset.filter(lambda x: x["constraint_type"] in ["End Checker", "Two Responses", "Forbidden Words", "Repeat Prompt"])
        sampled_dataset = dataset.shuffle()["train"].select(range(config['num_samples']))

        df = sampled_dataset.to_pandas()
        
        rows_to_process = [row for _, row in df.iterrows()]
        
        total_rows = len(rows_to_process)
        print(f"Processing {config['lang']} with {total_rows} samples")
        results = []

    
        # Set up a ThreadPoolExecutor. Adjust max_workers based on your needs and system.
        with ThreadPoolExecutor(max_workers=100) as executor:
            # Submit all translation tasks concurrently.
            futures = {executor.submit(process_row, row, config['lang'], config['lang_short'], keys_to_translate): row for row in rows_to_process}
            
            # Create progress bar
            with tqdm(total=total_rows, desc=f"Translating rows for {config['lang']}") as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                        pbar.update(1)
                    except Exception as e:
                        print("Error processing a row:", e)
                        pbar.update(1)
        
        csv_file = f"local_data_200225/translations_ifeval_{config['lang']}.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["ground_truth", "original_messages", "translated_messages", "dataset", "language", "constraint_type", "constraint"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"Translations saved to {csv_file}")

        all_results.extend(results)
        print(f"Processed {config['lang']} with {len(results)} samples")

        if i != len(all_configs) - 1:
            sleep(60)

    indic_dataset = Dataset.from_list(all_results)
    english_dataset = dataset.shuffle()["train"].select(range(TOTAL_SAMPLES_ENGLISH))
    # Add original_messages and translated_messages columns to english dataset
    english_dataset = english_dataset.map(lambda x: process_row(x, "English", "en", []))
    
    columns_to_keep = ["ground_truth", "original_messages", "translated_messages", "dataset", "language", "constraint_type", "constraint"]
    english_dataset = english_dataset.select_columns(columns_to_keep)
    
    indic_with_english = concatenate_datasets([indic_dataset, english_dataset])
    indic_with_english = indic_with_english.shuffle()

    indic_with_english = indic_with_english.train_test_split(test_size=0.1)

    import pdb; pdb.set_trace()

    indic_with_english.push_to_hub(
        f"sarvam/RLVR-Indic-IF-Eval",
        token=os.environ["HF_TOKEN"]
    )

if __name__ == "__main__":
    main()