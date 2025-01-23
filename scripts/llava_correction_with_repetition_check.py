import os
os.environ['TRANSFORMERS_CACHE'] = '/home/sbacker2/scr4_lhyman6/sbacker2/.cache/huggingface'
os.environ['TORCH_HOME'] = '/home/sbacker2/scr4_lhyman6/sbacker2/.cache/torch'
import logging
from PIL import Image
import torch
from transformers import pipeline
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import argparse
import glob
import json
import re
import sys 

log_file = open('lava_output.log', 'w')
sys.stdout = log_file

import re

import re

def detect_repeating_sequence(text):
    # Parameters
    search_length = 25
    threshold = 3

    text_list = text.split()
    #print(text_list)
    word_len = len(text_list)
    #print(word_len)
    start_point = int(word_len // 2) 
    #print(start_point)
    text_to_check = text_list[start_point:word_len]
    #print("this is text_to_check")
    #print(text_to_check)
    combined_text_to_check = " ".join(text_to_check)
    #print(combined_text_to_check)
    #print("detection started")
    # Extract the last `check_length` characters
    #print("this is text to check")

    # Iterate over the starting positions for the search sequence
    for start in range((start_point // 2)):
        # Define the search sequence
        local_text = " ".join(text_to_check)
        search_sequence = " ".join(text_to_check[start:start + search_length])
        
     #   print("this is search sequence")
      #  print(search_sequence)
        # Construct the regex pattern to find the search sequence repeated at least `threshold` times
        pattern = f"{search_sequence}"
        
        # Search for the pattern in the rest of the text
        matches = re.findall(pattern,combined_text_to_check)
       # print("these are the matches")
       # print(matches)
        # Check if the number of matches exceeds the threshold
        if len(matches) >= threshold:
            return True  # Repeating sequence detected
    
    return False  # No repeating sequence detected



def find_and_trim_repeating_sequence(text):
    # Parameters                                                                                                                                                                                                 
    search_length = 25
    threshold = 2

    text_list = text.split()

    word_len = len(text_list)

    start_point = round(word_len * .8)

    text_to_check = text_list[start_point:word_len]

    combined_text_to_check = " ".join(text_to_check)
    combined_text = " ".join(text_list)

# Iterate over the starting positions for the search sequence
    current_length = search_length
    for start in range((start_point // 2)):
        # Define the search sequence
        break_loop = False 
        current_length = search_length 
        valid_repeating_sequence = None
        while True:
            
            local_text = " ".join(text_to_check)
            
            search_sequence = " ".join(text_to_check[start:start + current_length])
            # Construct the regex pattern to find the search sequence repeated at least `threshold` times
            print("this is the search sequence")
            print(search_sequence)
            print(current_length)
            pattern = f"{search_sequence}"
            
            # Search for the pattern in the rest of the text                                                                                                                                                        

            matches = re.findall(pattern,combined_text_to_check)

            # Check if the number of matches exceeds the threshold                                                                                                                                              
            if len(matches) >= threshold:
                valid_repeating_sequence = search_sequence
                current_length += 1
            else:
                break_loop = True 
                break
            if current_length == len(local_text):
                break_loop = True
                break
                                        
        if break_loop == True and valid_repeating_sequence != None:
            break 
    # Find the first instance of the valid repeating sequence

    if valid_repeating_sequence:
        # Find the first instance of the valid repeating sequence
        pattern = re.escape(valid_repeating_sequence)
        print("this is the pattern")
        print(valid_repeating_sequence)
        
        match = re.search(valid_repeating_sequence, combined_text)
        if match:
            print("there was a match")
            first_instance_index = match.start()
            print("First instance index:", first_instance_index)
            
            # Calculate the end position of the first instance of the repeating sequence
            end_position = first_instance_index + len(valid_repeating_sequence)
            print("End position:", end_position)

            # Trim the text after the end of the first instance of the valid repeating sequence
            trimmed_text = text[:end_position]
            print("Trimmed text:", trimmed_text)
             
            return trimmed_text
        else:
            print("no match")
        return text      
        

def verify_image_type(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verifies that it is, in fact, an image
            print(f"{file_path} is a valid image.")
            return True
    except (IOError, SyntaxError) as e:
        print(f"{file_path} is not a valid image. Error: {e}")
        return False

parser = argparse.ArgumentParser()

parser.add_argument("--image_directory", dest='image_directory', help="original image files")
parser.add_argument("--prompt", dest="prompt", help = "prompt for analysis", nargs = "+", default = ["This is a historical document. Please return a transcription of the text from this document. Do not add any additional material beyond what is present in the text."])
parser.add_argument("--output_file", dest = "output_file", help = "where the output goes", required = True)

args = parser.parse_args()
new_text = []
name_list = glob.glob(args.image_directory + "/*" )
print(name_list)
output_dictionary = {}


# Check if CUDA is available
if torch.cuda.is_available():
        print("CUDA is available.")
else:
    print("CUDA is not available.")


processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)

prompt = ' '.join(args.prompt)

processor.tokenizer.padding_side = "left"
prompt =  "[INST] USER:<image>\n {} ASSISTANT:[/INST]".format(prompt)

#fix so it won't break or don't, if you only want to run on gpu. 
model.to("cuda:0")


for x in name_list:
    try:        
        if verify_image_type(x):
            image = Image.open(x)

            print(f"Opened image: {x}, size: {image.size}, format: {image.format}")
            
            
            # Ensure the image is in RGB mode (some models require it)
            if image.mode != 'RGB':
                image = image.convert('RGB')
                print(f"Converted image {x} to RGB mode.")


        inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")
        
        print("processed inputs")
    
        # Generate output
        output = model.generate(**inputs, max_new_tokens=700)

        print("generated output")
    
        # Decode and save the text
        text = processor.decode(output[0], skip_special_tokens=True)

        print("decoded text")

        
        print(text)
        if detect_repeating_sequence(text) == True:
            print("repeating text detected")
            text = find_and_trim_repeating_sequence(text)
            
            
            text = text.split("ASSISTANT:[/INST]")
            text = text[-1]
            #print(f"Text after removing repeated fragments: {new_text}")

            name =  x.split("images/")
            name = name[-1]
            name = name.rstrip(".jpg")
            
            entry = name + "corrected_by_llava_repetition_check"

            print("this is entry" + entry)
            output_dictionary[entry] = [text]
        else:                            
            text = text.split("ASSISTANT:[/INST]")
            text = text[-1]
            name =  x.split("images/")
            name = name[-1]
            name = name.rstrip(".jpg")
            entry = name + "corrected_by_llava_repetition_check"
            print(entry)
            output_dictionary[entry] = [new_text]

        print("did an image, my guy!")

    except Exception as e:
        print(f"Error processing image {x}: {e}")


        
with open(args.output_file, "w") as out_file:
    json.dump(output_dictionary, out_file)
