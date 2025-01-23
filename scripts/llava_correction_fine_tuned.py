import os
os.environ['TRANSFORMERS_CACHE'] = '/home/sbacker2/scr4_lhyman6/sbacker2/.cache/huggingface'
os.environ['TORCH_HOME'] = '/home/sbacker2/scr4_lhyman6/sbacker2/.cache/torch'
import logging
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import argparse
import glob
import json
from peft import PeftModel



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
parser.add_argument("--prompt", dest="prompt", help = "prompt for analysis", nargs = "*", default = ["This is a historical document. Please return a transcription of the text from this document"] )
parser.add_argument("--output_file", dest = "output_file", help = "where the output goes", required = True)
parser.add_argument("--checkpoint", dest = "checkpoint", help = "what checkpoint do you want to use?")


args = parser.parse_args()

name_list = glob.glob(args.image_directory + "/*" )
print(name_list)
output_dictionary = {}


# Check if CUDA is available                                                                                                                                                                                          
if torch.cuda.is_available():
        print("CUDA is available.")
else:
    print("CUDA is not available.")

checkpoint_name = args.checkpoint
checkpoint_name = checkpoint_name.split("/")
checkpoint_name = checkpoint_name[8]



#processor = AutoProcessor.from_pretrained(args.model)

processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

base_model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")

peft_model_id = args.checkpoint

model = PeftModel.from_pretrained(base_model, peft_model_id)

model.merge_adapter()

# If you are using GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

prompt = ' '.join(args.prompt)
print(prompt)
processor.tokenizer.padding_side = "left"


prompt =  "USER:<image>\n {}\nASSISTANT:".format(prompt)




#fix so it won't break or don't, if you only want to run on gpu.                                                                                                                                                      
#model.to("cuda:0")


for x in name_list:
    print(x)
    try:
        if verify_image_type(x):
            image = Image.open(x)

            print(f"Opened image: {x}, size: {image.size}, format: {image.format}")


            # Ensure the image is in RGB mode (some models require it)                                                                                                                                                
            #if image.mode != 'RGB':
            #    image = image.convert('RGB')
             #   print(f"Converted image {x} to RGB mode.")


        inputs = processor(prompt, image, return_tensors="pt").to(device)

        print("processed inputs")

        # Generate output                                                                                                                                                                                             
        output = model.generate(**inputs, max_new_tokens=500)

        print("generated output")

        # Decode and save the text                                                                                                                                                                                    
        #text = processor.decode(output[0], skip_special_tokens=True)
        text = processor.batch_decode(output, skip_special_tokens=True)[0] 
        print("decoded text")
        print(text)
        name =  x.split("images/")
        name = name[-1]
        name = name.rstrip(".jpg")
        #name = name.lstrip("pytesserac                                                                                                                                                                            
        entry = name + "corrected_by_llava" + checkpoint_name
        print(entry)
        print("this is text")
        print(text)
        text = text.split("\nASSISTANT")
        text = text[-1]

        #print("this is text")
        #print(text)
        output_dictionary[entry] = text

        print("did an image, my guy!")

    except Exception as e:
        print(f"Error processing image {x}: {e}")



with open(args.output_file, "w") as out_file:
    json.dump(output_dictionary, out_file)
