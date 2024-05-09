import argparse
import os 
import jiwer
from jiwer import wer, cer 
import json

parser = argparse.ArgumentParser()

parser.add_argument("--control_directory", dest='control_directory', help="ground truth file_folder")

parser.add_argument("--test_directory", dest= "test_directory", help = "test file folder")

parser.add_argument("--output_file", dest = "output_folder", help = "where the output goes", required = True)
args = parser.parse_args()
control_files = []
test_files = []
output_dictionary = {}



with open(args.control_directory, "r") as the_file:  
    control_files = json.load(the_file)
    print(len(control_files))

    
# new test files should always be in a specific json format



with open(args.test_directory, "r") as in_file:
    local_files = json.load(in_file)
    print(len(local_files))

        #for key, item in local_file.items():
         #   print(key)
        

output_dictionary = {} 

#for element in local_files:
 #   print(element)




for key_1, gpt_item in local_files.items():
    print(key_1)
    #print(gpt_item)
    entry = key_1.split("corrected")
    print(entry)
   
    entry_name = entry[0]
    print(entry_name)
    for key, item in control_files.items():
        new_x = key.rstrip(".txt")
        print(new_x)
        if entry_name == new_x:
            print("matched")
            if gpt_item.get("choices"):
                test_text = gpt_item["choices"][0]["message"]["content"]
            
                control_text = item["control_text"]
            
            
                error = cer(control_text, test_text)
                print(error)
                model_name = gpt_item["model"]
                new_key = key_1.split("pytesseract")
                print(new_key)
                new_key = new_key[-1]
                output_dictionary[new_key] = {"model_name" : model_name, "text" : test_text, "CER" : error}
            
            #local_dict = {name : { "text" : test_text, "CER" : error}} 

            #control_files[key].update(local_dict)

            

                
                
with open (args.output_folder, "w") as out_file:
    json.dump(output_dictionary, out_file, indent=4)

