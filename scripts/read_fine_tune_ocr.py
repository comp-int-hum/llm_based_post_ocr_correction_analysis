import json
import csv
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--input_csv", dest='input_csv', help="the input csv file")
parser.add_argument("--input_json", dest='input_json', help="the input json file")

parser.add_argument("--output_file", dest= "output_file", help = "output location")

args = parser.parse_args()



def csv_to_dict_of_dicts(csv_file_path):
    data_dict = {}
    
    # Read the CSV file and use the first column as the key for the dictionary
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            key = row[csv_reader.fieldnames[0]]  # Use the first column as the key
            data_dict[key] = {header: value for header, value in row.items() if header != csv_reader.fieldnames[0]}
    
    return data_dict

def scores(test_text, control_text):
    character_error = cer(control_text, test_text)
    word_error = wer(control_text, test_text)
    word_information_lost = (control_text, test_text)
    return {"CER" : character_error, "WER" : word_error,"WIL":  word_information_list}


fine_tune_dictionary = csv_to_dict_of_dicts(args.input_csv)

output_dictionary = {}

for key, items in fine_tune_dictionary:
    control_dictionary = items["GROUND_TRUTH"]
    bert_1 = items["BERT1"]
    bert2 = items["BERT2"]
    bert3 = items["BERT3"]
    llava1= items["LAVA1"]
    llava2 = items["LAVA2"]
    llava3 = itmes["LAVA3"]

    
    
    output_dictionary[key] = 
    
