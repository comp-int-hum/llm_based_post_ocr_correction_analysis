import os
os.environ['TRANSFORMERS_CACHE'] = '/home/sbacker2/scr4_lhyman6/sbacker2/.cache/huggingface'
os.environ['TORCH_HOME'] = '/home/sbacker2/scr4_lhyman6/sbacker2/.cache/torch'
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig
import torch
import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument("--input", dest='data_directory', help="transcribed data")
parser.add_argument("--output", dest= "output_directory", help = "output file")
parser.add_argument("--prompt", dest= "prompt", help = "what is your prompt for gpt?", nargs = "*")

args = parser.parse_args()
output_dictionary = {}
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
)



model = AutoPeftModelForCausalLM.from_pretrained(
    'pykale/llama-2-13b-ocr',
    quantization_config=bnb_config,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16, token= access_token
)

tokenizer = AutoTokenizer.from_pretrained('pykale/llama-2-13b-ocr')


with open (args.data_directory, "r") as in_file:
    pytesseract_material = json.load(in_file)

print(len(pytesseract_material))    
for name, ocr in pytesseract_material.items():
    print(name)
    prompt = f"""### Instruction:
    Fix the OCR errors in the provided text.

    ### Input:
    {ocr}

    ### Response:
    """

    input_ids = tokenizer(prompt, max_length=1024, return_tensors='pt', truncation=True).input_ids.cuda()
    with torch.inference_mode():
        outputs = model.generate(input_ids=input_ids, max_new_tokens=1024, do_sample=True, temperature=0.2, top_p=0.1, top_k=40)
    pred = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):].strip()
    print("")
    print(pred)
    name = name.lstrip("pytesseract")
    name = name.rstrip(".txt")
    entry = name + "corrected_by_fine_tuned_llama"
    output_dictionary[entry] = pred

with open(args.output_directory, "w") as out_file:
    json.dump(output_dictionary, out_file)

