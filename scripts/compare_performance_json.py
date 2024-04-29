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

with open(args.control_directory, "r") as the_file:  
    control_files = json.load(the_file)


for x in control_files:
    print(x)

    
# new test files should always be in a specific json format



with open(args.test_directory, "r") as in_file:
    local_files = json.load(in_file)


        #for key, item in local_file.items():
         #   print(key)
        

output_dictionary = {} 

#for element in local_files:
 #   print(element)




for key, gpt_item in local_files.items():
    print(key)
    print(gpt_item)
    entry = key.split("/work/")
   # print(entry)
    entry = entry[1]
    entry = entry.split("corrected")
    entry_name = entry[0]

    #print(type(entry_name))
    entry_name = entry_name.split("pytesseract")
    #print(entry_name)
    entry_name = entry_name[-1]                    
    #print(entry_name)
    for key, item in control_files.items():
        new_x = key.rstrip(".txt")
        print(new_x)
        if entry_name == new_x:
        
            test_text = gpt_item["choices"][0]["message"]["content"]
            control_text = item["control_text"]
            
            
            error = cer(control_text, test_text)
            name = key["model"]    
            local_dict = {name : { "text" : test_text, "CER" : error}} 

            control_files[key].update(local_dict)

            

                
                
with open (args.output_folder, "w") as out_file:
    json.dump(control_files, out_file, indent=4)

