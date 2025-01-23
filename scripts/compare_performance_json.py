import argparse
import os 
import jiwer as tr 
from jiwer import wer, cer, wil 
import json
import re
from Levenshtein import distance as levenshtein_distance
from collections import Counter
from math import sqrt

def remove_special_characters(text):
    # Define the regex pattern to match newline and non-printable ASCII characters
    pattern = r'[\n\x00-\x1F\x7F.,:]'  # \x00-\x1F covers non-printable ASCII, \x7F is DEL
    # Use re.sub to replace special characters with an empty string
    cleaned_text = re.sub(pattern, ' ', text)
    return cleaned_text

def jaccard_similarity(list1, list2):
    set1, set2 = set(list1), set(list2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / set2

def levenshtein_similarity(list1, list2):
    distance = levenshtein_distance(''.join(list1), ''.join(list2))
    max_len = max(len(list1), len(list2))
    return 1 - distance / max_len

def calculate_cer(reference_text, recognized_text):
    # Calculate the Levenshtein distance
    distance = levenshtein_distance(reference_text, recognized_text)
    
    # Calculate CER
    cer = distance / len(reference_text)
    
    return cer

parser = argparse.ArgumentParser()

parser.add_argument("--control_directory", dest='control_directory', help="ground truth file_folder")

parser.add_argument("--test_directory", dest= "test_directory", help = "test file folder")

parser.add_argument("--output_file", dest = "output_folder", help = "where the output goes", required = True)
args = parser.parse_args()
control_files = []
test_files = []
output_dictionary = {}

wer_standardize_contiguous = tr.Compose(
    [
        tr.Strip(),
        tr.ToLowerCase(),
        tr.RemoveWhiteSpace(replace_by_space=True),
        tr.ReduceToListOfListOfWords(),
    ]
)

cer_contiguous = tr.Compose(
    [
        tr.Strip(),
        tr.ToLowerCase(),
        tr.RemoveWhiteSpace(replace_by_space=True),
        tr.ReduceToListOfListOfChars(),
    ]
)

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
    #print(key_1)
    #print(gpt_item)
    new_key = key_1.replace(".jpg", "")
    entry = new_key.split("corrected_by_")
    #print(entry)
   
    entry_name = entry[0]
    #print(entry_name)
    for key, item in control_files.items():
        new_x = key.rstrip(".txt")
        #print(new_x)
        if entry_name == new_x:
            print("matched")
            test_text = remove_special_characters(gpt_item)
            #print(test_text)
            control_text = item["control_text"]
            control_text = remove_special_characters(control_text)
           
           
            change = tr.ToLowerCase()(control_text)
            #print(change)
            change = tr.RemoveWhiteSpace(replace_by_space=True)(change)
            change1 = tr.ReduceToListOfListOfWords()(change)
            #print(change1)
            double_change = tr.ReduceToListOfListOfChars()(change)
            print(double_change)
            triple_change = [[]]
            for char in double_change[0]:
                if char != " ":
                    
                    triple_change[0].append(char)
            #print(triple_change)

            test_change = tr.ToLowerCase()(test_text)
            test_change = tr.RemoveWhiteSpace(replace_by_space=True)(test_change)
            test_change_1 = tr.ReduceToListOfListOfWords()(test_change)
            double_test_change = tr.ReduceToListOfListOfChars()(test_change)
            print(double_test_change)
            triple_test_change = [[]]
            for char in double_test_change[0]:
                if char != " ":

                    triple_test_change[0].append(char)


            #print("next text")
            #print(test_change_1)
            #print(triple_test_change)
            
            character_error = cer(control_text, test_text, truth_transform = cer_contiguous, hypothesis_transform = cer_contiguous)
            print("this is CER" + str(character_error))
            print("this is the local lev" + str(levenshtein_similarity(triple_change[0], triple_test_change[0])))
            local_cer = calculate_cer(triple_change[0], triple_test_change[0])
            print("this is the local cer" + str(calculate_cer(triple_change[0], triple_test_change[0])))
            word_error = wer(control_text, test_text, truth_transform = wer_standardize_contiguous, hypothesis_transform = wer_standardize_contiguous)
            print("this is WER" + str(word_error))



            word_information_lost = wil(control_text, test_text,truth_transform=wer_standardize_contiguous, hypothesis_transform=wer_standardize_contiguous)
            model_name = entry[1]
            new_key = key_1.split("pytesseract")
            print(new_key)
            new_key = new_key[-1]
            jaccard = jaccard_similarity(double_change[0], double_test_change[0])
            output_dictionary[new_key] = {"model_name" : model_name, "text" : test_text,  "CER" : character_error, "NEW_CER" : local_cer, "WER" : word_error, "WORD_JACCARD": jaccard, "WIL": word_information_lost}
            
            #local_dict = {name : { "text" : test_text, "CER" : error}} 

            #control_files[key].update(local_dict)

            

                
                
with open (args.output_folder, "w") as out_file:
    json.dump(output_dictionary, out_file, indent=4)

