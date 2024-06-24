import json
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--input", dest='data_directory', help="image location")
parser.add_argument("--output", dest= "output_directory", help = "output file")

args = parser.parse_args()
# Function to load data from a JSON file
def load_data(filename):
    with open(filename, 'r') as file:
        return json.load(file)

# Function to calculate statistics for each OCR correction type and return them as a dictionary
def calculate_statistics(data):
    # Initialize a dictionary to hold lists of error rates for each correction type
    error_rates = {}

    # Populate the dictionary with error rates
    for document in data.values():
        for method, rate in document.items():
            if method not in error_rates:
                error_rates[method] = []
            error_rates[method].append(rate)

    # Dictionary to hold statistics
    stats = {}

    # Calculate statistics for each correction type
    for method, rates in error_rates.items():
        rates_array = np.array(rates)
        stats[method] = {
            'Average': np.mean(rates_array),
            'Median': np.median(rates_array),
            'Standard Deviation': np.std(rates_array)
        }

    return stats

# Load data from JSON file
data = load_data(args.data_directory)

# Calculate statistics
statistics = calculate_statistics(data)

# Output statistics to a JSON file
with open(args.output_directory, 'w') as outfile:
    json.dump(statistics, outfile, indent=4)

print("Statistics have been saved to 'statistics_output.json'")
