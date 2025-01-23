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
from torch.nn import DataParallel




def is_repeating_text(text, n_sentences=5, threshold=3):
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    if len(sentences) < n_sentences:
        return False
    last_sentences = sentences[-n_sentences:]
    repetition_count = sum([last_sentences.count(sentence) > 1 for sentence in set(last_sentences)])
    return repetition_count >= threshold


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
parser.add_argument("--prompt", dest="prompt", help = "prompt for analysis", nargs = "+", default = ["This is a historical document. Please return a transcription of the text from this document"])
parser.add_argument("--output_file", dest = "output_file", help = "where the output goes", required = True)

args = parser.parse_args()

name_list = glob.glob(args.image_directory + "/*" )
print(name_list)
output_dictionary = {}





# Check if CUDA is available
if torch.cuda.is_available():
        print("CUDA is available.")
else:
    print("CUDA is not available.")



#device = 0 if torch.cuda.is_available() else -1

#if device == 0:
 #   print("cuda available")
#else:
 #   print("using cpus")


processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-34b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-34b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
prompt = ' '.join(args.prompt)
print(prompt)
processor.tokenizer.padding_side = "left"
#prompt =  "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n{}{}<|im_end|><|im_start|>assistant\n".format(image,prompt)

#fix so it won't break or don't, if you only want to run on gpu. 

model = DataParallel(model)
model.to("cuda:0")


    

#print("created the pipeline")



for x in name_list:
    print(x)
    try:        
        if verify_image_type(x):
            image = Image.open(x)

            print(f"Opened image: {x}, size: {image.size}, format: {image.format}")
            
            
            # Ensure the image is in RGB mode (some models require it)
            if image.mode != 'RGB':
                image = image.convert('RGB')
                print(f"Converted image {x} to RGB mode.")

#        prompt =  "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n{}{}<|im_end|><|im_start|>assistant\n".format(image,prompt)
        prompt = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\nThis is a historical document. Please return a transcription of the text from this document starting the transcription with the word BEGIN and ending with the word FINISH. Do NOT include any commentary or discussion of the text beyond what is initially included within it. <|im_end|><|im_start|>assistant\n"
        inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")
        
        print("processed inputs")
    
        # Generate output
    #        output = model.generate(**inputs, max_new_tokens=500)
        output = model.module.generate(**inputs, max_new_tokens=700)
        print("generated output")
    
        # Decode and save the text
        text = processor.decode(output[0], skip_special_tokens=True)

        print("decoded text")
        print(text)

        name =  x.split("images/")
        name = name[-1]
        name = name.rstrip(".jpg")
        #name = name.lstrip("pytesseract")
        entry = name + "corrected_by_llava"
        print(entry)
        text = text.split("<|im_start|> assistant")
        text = text[-1]

        

        output_dictionary[entry] = text 

        print("did an image, my guy!")

    except Exception as e:
        print(f"Error processing image {x}: {e}")

print("change_to_trigger")
        
with open(args.output_file, "w") as out_file:
    json.dump(output_dictionary, out_file)
