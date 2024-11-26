import os
os.environ['TRANSFORMERS_CACHE'] = '/home/sbacker2/scr4_lhyman6/sbacker2/.cache/huggingface'
os.environ['TORCH_HOME'] = '/home/sbacker2/scr4_lhyman6/sbacker2/.cache/torch'
import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
from tqdm import tqdm
import argparse
import glob 
import json 

parser = argparse.ArgumentParser()

parser.add_argument("--input", dest='data_directory', help="transcribed data")
parser.add_argument("--output", dest= "output", help = "output file")
parser.add_argument("--base_model", dest= "base_model", help = "what is the base model", default = "/scratch4/lhyman6/OCR/OCR/ocr_llm/work")
parser.add_argument("--prompt", dest= "prompt", help = "what is your prompt for gpt?", nargs = "*", default = "Correct this OCR:")
parser.add_argument("--checkpoint", dest= "checkpoint", help = "Which checkpoint should you use")

args = parser.parse_args()

model_base_path = args.base_model


csv_path = '/data/lhyman6/OCR/scripts/ocr_llm/complete_testing_csv.csv'
output_csv_path = args.output
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
    print("model loaded")
    with open(args.data_directory, "r") as in_file:
        pytesseract_dict = json.load(in_file)

    for name, item in pytesseract_dict.items():
        result, is_blank = load_and_generate(untrained_model, tokenizer, device, f"{base_prompt}: {item}")
        print("did result")
        if is_blank:
            blank_count += 1
            print(result)
        name = name.rstrip(".txt")
        name = name.lstrip("pytesseract")
        entry = name + "corrected_by_bart"
        output_dictionary[entry] = result
    print(f"Untrained BART model processed successfully, {blank_count} blank responses detected.")

    with open(args.output, "w") as out_file:
        json.dump(output_dictionary, out_file)
    
    

if __name__ == "__main__":
    main()
