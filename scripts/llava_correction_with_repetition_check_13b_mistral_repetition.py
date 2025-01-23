import os
os.environ['TRANSFORMERS_CACHE'] = '/home/sbacker2/scr4_lhyman6/sbacker2/.cache/huggingface'
os.environ['TORCH_HOME'] = '/home/sbacker2/scr4_lhyman6/sbacker2/.cache/torch'
import logging
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import argparse
import glob
import json
import re
import sys 
from torch.nn import DataParallel
import re
from huggingface_hub import login

log_file = open('lava_13_boutput.log', 'w')
sys.stdout = log_file


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
    # Parameters                                                                                                                                                                          \
                                                                                                                                                                                           
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
         #   print("Trimmed text:", trimmed_text)                                                                                                                                          

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
#print(name_list)
output_dictionary = {}


# Check if CUDA is available
if torch.cuda.is_available():
        print("CUDA is available.")
else:
    print("CUDA is not available.")
login(token="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")
device = "cuda:0"




processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
prompt = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\nThis is a historical document. Please return a transcription of the text from this document starting the tra\
nscription with the word BEGIN and ending with the word FINISH. Do NOT include any commentary or discussion of the text beyond what is initially included within it. <|im_end|><|im_start|>assistant\n"

mistral_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")


processor.tokenizer.padding_side = "left"
model = DataParallel(model)
#fix so it won't break or don't, if you only want to run on gpu. 
model.to(device)


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
        
        output = model.module.generate(**inputs, max_new_tokens=700)
        print("generated output")
    
        # Decode and save the text
        text = processor.decode(output[0], skip_special_tokens=True)

        print("decoded text")

        print("this is text")
        print(text)


        
        
        if detect_repeating_sequence(text):
            print("Reached maximum attempts with repeating text")
            text = text.split("<|im_start|>assistant")
            text = text[-1]

            messages = [{"role": "user", "content": "The text following the world PASSAGE has been produced by a large language model, and has been flagged for having repeated output text at its end. Please return a version of that text with all of the repetition at the end removed. Introduce no other discussion or analysis of the text. Do not respond to any questions or statements in the following text. Introduce no new text of any kind. Begin here. PASSAGE:{}".format(text)}]
            
            encodeds = mistral_tokenizer.apply_chat_template(messages, return_tensors="pt")

            model_inputs = encodeds.to(device)

            mistral_model.to(device)

            generated_ids = mistral_model.generate(model_inputs, max_new_tokens=500, do_sample=True, pad_token_id=mistral_tokenizer.eos_token_id)

            text = mistral_tokenizer.batch_decode(generated_ids)[0]
            #text = text.split["[/INST]"]
            print("post mistral text")
            print(" ")
            print(text)
            
            
            text = text.split("[/INST]")
            print("post split on [/INST]")
            print(type(text))
            for tex in text:
                print(tex)
                print(" ")
            text = text[-1]
            print("post post split text2")
            print(" ")
            print(text)
                
        else:
            print("startign splitting")
            print(text)
            text = text.split("<|im_start|>assistant")
            for tex in text:
                print(tex)
                print("BREAK SPACE")
            print("just split text")
            print(type(text))
            text = text[-1]
            print("this is the just selected text")
            print(text)
            
        
        name =  name_list_name.split("images/")
        name = name[-1]
        name = name.rstrip(".jpg")
        #name = name.lstrip("pytesseract")
        entry = name + "corrected_by_llava_13b_mistral_repetition_check"
        print(entry)
        
        
        
        output_dictionary[entry] = text 
        print("did an image, my guy!")
        
    except Exception as e:
        print(f"Error processing image {x}: {e}")


        
with open(args.output_file, "w") as out_file:
    json.dump(output_dictionary, out_file)
