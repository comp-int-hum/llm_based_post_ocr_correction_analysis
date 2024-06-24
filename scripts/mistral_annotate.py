import os
os.environ['TRANSFORMERS_CACHE'] = '/home/sbacker2/scr4_lhyman6/sbacker2/.cache/huggingface'
os.environ['TORCH_HOME'] = '/home/sbacker2/scr4_lhyman6/sbacker2/.cache/torch'
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import glob
import json
from huggingface_hub import login

login(token="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")
device = "cuda" 

parser = argparse.ArgumentParser()

parser.add_argument("--input_file", dest='input_file', help="the input file ")

parser.add_argument("--output_file", dest= "output_file", help = "output location")

args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

with open ( args.input_file, "r") as in_file:
    local_dic = json.load(in_file)

annotated_dictionary = {}

    
for key, value in local_dic.items():
    name = key 
    


    messages = [{"role": "user", "content": "The text following Passage: contains a variety of historical information. Please output a json object containing the answers to the following questions. What is the sentiment of this passage? What named entities are mentioned? Who is it being written from? Who is it being written to? If you do not have an answer to these questions, please output  the word BLANK instead. Passage:{}".format(value)}]

    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

    model_inputs = encodeds.to(device)

    model.to(device)
    
    generated_ids = model.generate(model_inputs, max_new_tokens=500, do_sample=True, pad_token_id=tokenizer.eos_token_id)

    output = tokenizer.batch_decode(generated_ids)[0]

    annotated_dictionary[name] = output

with open (args.output_file, "w") as out_file:
    json.dump(annotated_dictionary, out_file)
