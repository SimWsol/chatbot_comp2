import os
import re
import json
from datasets import load_dataset

# Directory for saving processed datasets
output_dir = "processed_dailydialog"
os.makedirs(output_dir, exist_ok=True)

# Length constraints for filtering
MIN_LENGTH = 5
MAX_LENGTH = 5000


def preprocess_text(text):
    """Cleans and normalizes text by removing extra spaces and fixing punctuation."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s([?.!,])", r"\1", text)
    return text.strip()


def is_valid_length(input_text, response_text):
    """Checks if both input and response meet the length requirements."""
    return MIN_LENGTH <= len(input_text) <= MAX_LENGTH and MIN_LENGTH <= len(response_text) <= MAX_LENGTH


def save_pair_to_files(txt_file, jsonl_file, input_text, response_text):
    """Processes and writes valid input-response pairs to text and JSONL files."""
    input_cleaned = preprocess_text(input_text)
    response_cleaned = preprocess_text(response_text)
    if is_valid_length(input_cleaned, response_cleaned):
        txt_file.write(f"{input_cleaned}|{response_cleaned}\n")
        json.dump({"input": input_cleaned, "response": response_cleaned}, jsonl_file)
        jsonl_file.write("\n")


# Combined output files
combined_txt_path = os.path.join(output_dir, "all_input_response.txt")
combined_jsonl_path = os.path.join(output_dir, "all_input_response.jsonl")
with open(combined_txt_path, "w", encoding="utf-8") as combined_txt, \
        open(combined_jsonl_path, "w", encoding="utf-8") as combined_jsonl:
    # Process DailyDialog dataset
    print("Downloading and processing DailyDialog dataset...")
    dataset = load_dataset("daily_dialog", trust_remote_code=True)

    for split in ["train", "validation", "test"]:
        split_txt_path = os.path.join(output_dir, f"dailydialog_{split}_input_response.txt")
        split_jsonl_path = os.path.join(output_dir, f"dailydialog_{split}_input_response.jsonl")

        with open(split_txt_path, "w", encoding="utf-8") as split_txt, \
                open(split_jsonl_path, "w", encoding="utf-8") as split_jsonl:

            for dialog in dataset[split]:
                utterances = dialog["dialog"]
                for i in range(len(utterances) - 1):
                    save_pair_to_files(split_txt, split_jsonl, utterances[i], utterances[i + 1])
                    save_pair_to_files(combined_txt, combined_jsonl, utterances[i], utterances[i + 1])

print(f"Processing complete. Files saved in '{output_dir}'.")