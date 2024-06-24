import argparse
import os 
import jiwer
from jiwer import wer, cer 
import json
import numpy as np

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
    test_files = json.load(in_file)
    print(len(test_files))

        #for key, item in local_file.items():
         #   print(key)
        

output_dictionary = {} 

#for element in local_files:
 #   print(element)



for key_1, gpt_item in test_files.items():
    print(key_1)
    #print(gpt_item)
    entry = key_1.split("corrected_by_")
    print(entry)
   
    entry_name = entry[0]
    print(entry_name)
    for key, item in control_files.items():
        new_x = key.rstrip(".txt")
        print(new_x)
        if entry_name == new_x:
            print("matched")
            
            test_text = gpt_item
            
            control_text = item["control_text"]
            
            for prompt, response in gpt_item.items():
            
                error = cer(control_text, response)
                print(error)
                
                new_key = key_1.split("pytesseract")
                print(new_key)
                new_key = new_key[-1]

                if output_dictionary.get(new_key) != None:
                    output_dictionary[new_key].update({prompt: error})
                else:
                    output_dictionary[new_key] = {prompt:error}
            
            
# Initialize a dictionary to store CER values for each prompt
prompt_cer_values = {}

# Collect CER values for each prompt across all entries
for entry in output_dictionary.values():
    for prompt, cer in entry.items():
        if prompt not in prompt_cer_values:
            prompt_cer_values[prompt] = []
        prompt_cer_values[prompt].append(cer)

# Initialize dictionaries to store the statistics for each prompt
average_cer_dict = {}
median_cer_dict = {}
std_dev_cer_dict = {}

# Calculate and store the average, median, and standard deviation for each prompt
for prompt, cer_list in prompt_cer_values.items():
    cer_array = np.array(cer_list)
    average_cer_dict[prompt] = np.mean(cer_array)
    median_cer_dict[prompt] = np.median(cer_array)
    std_dev_cer_dict[prompt] = np.std(cer_array)

# Combine the statistics into a single dictionary
statistics_dict = {}
for prompt in prompt_cer_values.keys():
    statistics_dict[prompt] = {
        "average": average_cer_dict[prompt],
        "median": median_cer_dict[prompt],
        "std_dev": std_dev_cer_dict[prompt]
    }    
            
final_output = [statistics_dict, output_dictionary]
                
                
with open (args.output_folder, "w") as out_file:
    json.dump(final_output, out_file, indent=4)

