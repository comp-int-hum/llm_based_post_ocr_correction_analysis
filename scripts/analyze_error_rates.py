import json
import numpy as np
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()

parser.add_argument("--input", dest='data_directory', help="image location")
parser.add_argument("--output", dest= "output_directory", help = "output file")

args = parser.parse_args()
# Function to load data from a JSON file
def load_data(filename):
    with open(filename, 'r') as file:
        return json.load(file)



overall_dict = load_data(args.data_directory)


# Initialize dictionaries to collect CER, WER, WIL values for each model
metrics = defaultdict(lambda: defaultdict(list))

# Iterate through each dictionary in overall_dict
for key, entry in overall_dict.items():
    
    #print(key)
    #print(entry.keys())
    for sub_key, sub_entry in entry.items():
        print(sub_key)
        if sub_key == "control_text":
            continue  # Skip control text as it doesn't have metrics
        
        # Determine the model name
        if sub_key == "pytesseract_text":
            model_name = "pytesseract"
        else:
            #print(sub_entry)
            model_name = sub_entry.get("model_name", "unknown_model")
        
        # Collect the CER, WER, WIL values for this model
        metrics[model_name]['CER'].append(sub_entry["CER"])
        metrics[model_name]["NEW_CER"].append(sub_entry["NEW_CER"])
        metrics[model_name]['WER'].append(sub_entry["WER"])
        metrics[model_name]["WORD_JACCARD"].append(sub_entry["WORD_JACCARD"])
        metrics[model_name]['WIL'].append(sub_entry["WIL"])

# Function to calculate statistics
def calculate_statistics(values):
    return {
        'average': round(np.mean(values),3),
        'median': round(np.median(values),3),
        'std_dev': round(np.std(values), 3)
    }

# Calculate statistics for each model and each metric
results = defaultdict(lambda: defaultdict(dict))

for model_name, model_metrics in metrics.items():
    for metric_name, values in model_metrics.items():
        results[metric_name][model_name] = calculate_statistics(values)

# Convert the results to a dictionary and write to a JSON file

with open(args.output_directory, 'w') as f:
    json.dump(results, f, indent=4)




