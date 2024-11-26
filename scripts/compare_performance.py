import argparse
import os 
import jiwer
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

def calculate_cer(reference_text, recognized_text):
    # Calculate the Levenshtein distance                                                                                                                                    
    distance = levenshtein_distance(reference_text, recognized_text)

    # Calculate CER                                                                                                                                                         
    cer = distance / len(reference_text)

    return cer

def levenshtein_similarity(list1, list2):
    distance = levenshtein_distance(''.join(list1), ''.join(list2))
    max_len = max(len(list1), len(list2))
    return 1 - distance / max_len

def jaccard_similarity(list1, list2):
    set1, set2 = set(list1), set(list2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

parser = argparse.ArgumentParser()

parser.add_argument("--control_directory", dest='control_directory', help="ground truth file_folder")
parser.add_argument("--test_directory", dest= "test_directory", help = "test file folder")
parser.add_argument("--output_file", dest = "output_folder", help = "where the output goes", required = True)

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


args = parser.parse_args()
control_files = []
test_files = []

for filename in os.listdir(args.control_directory):
    if filename.endswith('.txt'):
        control_files.append(filename)        


#for filename in os.listdir(args.test_directory):
 #   if filename.endswith('.txt'):
  #      test_files.append(filename)


  
with open(args.test_directory, "r") as in_file:
    test_dictionary = json.load(in_file)

        
output_dictionary = {}

for key, item  in test_dictionary.items():
    entry_name = key.lstrip("pytesseract")
    print(entry_name)
    for x in control_files:
        print(x)
        if entry_name == x:
            with open (os.path.join(args.control_directory,x), "r") as control_file:
                test_text = remove_special_characters(item)
                control_text = control_file.read()
                print(control_text)
                control_text = remove_special_characters(control_text)
                #output = jiwer.process_words(control_text, test_text)
                #error = output.cer

                change = tr.ToLowerCase()(control_text)
                #print(change)                                                                                                                                                  
                change = tr.RemoveWhiteSpace(replace_by_space=True)(change)
                change1 = tr.ReduceToListOfListOfWords()(change)
                #print(change1)                                                                                                                                                 
                double_change = tr.ReduceToListOfListOfChars()(change)
                #print(double_change)                                                                                                                                           
                triple_change = [[]]
                for char in double_change[0]:
                    if char != " ":

                        triple_change[0].append(char)
                test_change = tr.ToLowerCase()(test_text)
                test_change = tr.RemoveWhiteSpace(replace_by_space=True)(test_change)
                test_change_1 = tr.ReduceToListOfListOfWords()(test_change)
                double_test_change = tr.ReduceToListOfListOfChars()(test_change)
                triple_test_change = [[]]
                for char in double_test_change[0]:
                    if char != " ":

                        triple_test_change[0].append(char)

                local_cer = calculate_cer(triple_change[0], triple_test_change[0])
                character_error = cer(control_text, test_text, truth_transform=cer_contiguous, hypothesis_transform=cer_contiguous)
                word_error = wer(control_text, test_text,truth_transform=wer_standardize_contiguous, hypothesis_transform=wer_standardize_contiguous )
                jaccard = jaccard_similarity(double_change[0], double_test_change[0])



                word_information_lost = wil(control_text, test_text,truth_transform=wer_standardize_contiguous, hypothesis_transform=wer_standardize_contiguous )
                local_dict = {"control_text" : control_text, "pytesseract_text" : { "text" : test_text, "CER" : character_error, "NEW_CER" : local_cer, "WER" : word_error,
                "WORD_JACCARD": jaccard, "WIL": word_information_lost}}
                print(local_dict)
                output_dictionary[x] = local_dict 

                        
with open (args.output_folder, "w") as out_file:
    json.dump(output_dictionary, out_file, indent = 4)

