import os
import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
from tqdm import tqdm
import argparse
import glob 
import json 

parser = argparse.ArgumentParser()

parser.add_argument("--input", dest='data_directory', help="transcribed data")
parser.add_argument("--output", dest= "output_directory", help = "output file")
parser.add_argument("--base_model", dest= "base_model", help = "what is the base model", default = "/scratch4/lhyman6/OCR/OCR/ocr_llm/work")
parser.add_argument("--prompt", dest= "prompt", help = "what is your prompt for gpt?", nargs = "*", default = "Correct this OCR:")
parser.add_argument("--checkpoint", dest= "checkpoint", help = "Which checkpoint should you use", default ="")

args = parser.parse_args()

model_base_path = ars.


csv_path = '/data/lhyman6/OCR/scripts/ocr_llm/complete_testing_csv.csv'
output_csv_path = args.output_directory
model_base_path = args.base_model 


base_prompt = args.prompt
save_interval = 100  # Save after processing every 50 rows
process_row_limit = 2000  # Number of rows to process

# Helper functions
def get_latest_checkpoint(model_dir):
    checkpoints = [d for d in os.listdir(model_dir) if d.startswith("checkpoint-")]
    return max(checkpoints, key=lambda x: int(x.split('-')[-1])) if checkpoints else None

def load_and_generate(model, tokenizer, device, prompt):
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_length=1000,  # Increased max_length
            min_length=50,  # Added min_length
            num_beams=5, 
            length_penalty=2.0,  # Added length_penalty
            early_stopping=True
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if not generated_text.strip():
        return "ERROR: Blank response generated", True
    return generated_text, False


# Main function
def main():
    print("Starting processing...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    output_dictionary = {}
    
    blank_count = 0  # Initialize counter for blank responses

    print("Loading the untrained BART model...")
    untrained_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large').to(device)

    with open(args.input, "r") as in_file:
        pytesseract_dict = json.load(in_file)


    model_dir = os.path.join(args.checkpoint)
    latest_checkpoint = get_latest_checkpoint(model_dir)
    model = BartForConditionalGeneration.from_pretrained(full_path).to(device)
    
        
    for name, item in pytesseract_dict:
        result, is_blank = load_and_generate(untrained_model, tokenizer, device, f"{base_prompt}: {item}")
            if is_blank:
                blank_count += 1
            name = name.rstrip(".txt")
            name = name.lstrip("pytesseract")
            entry = name + "corrected_by_bart{}".format(args.gpt_version)
    print(f"Untrained BART model processed successfully, {blank_count} blank responses detected.")

    # Process with trained models
    for dir_name in model_dirs:
        model_dir = os.path.join(model_base_path, dir_name)
        latest_checkpoint = get_latest_checkpoint(model_dir)
        if latest_checkpoint:
            full_path = os.path.join(model_dir, latest_checkpoint)
            model = BartForConditionalGeneration.from_pretrained(full_path).to(device)
            model_column = model_output_columns[dir_name]
            print(f"Processing with model from {full_path}...")
            for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Processing {dir_name}"):
                if pd.isna(row[model_column]):
                    result, is_blank = load_and_generate(model, tokenizer, device, f"{base_prompt}: {row['pyte_ocr']}")
                    if is_blank:
                        blank_count += 1
                    df.at[index, model_column] = result
                # Save periodically
                if (index + 1) % save_interval == 0:
                    df.to_csv(output_csv_path, index=False)
                    print(f"Progress saved at row {index + 1}")
            print(f"Completed processing for {model_column}, {blank_count} blank responses detected.")

    # Saving results back to the new CSV
    df.to_csv(output_csv_path, index=False)
    print("Results saved back to the new CSV, total blank responses: ", blank_count)

if __name__ == "__main__":
    main()
