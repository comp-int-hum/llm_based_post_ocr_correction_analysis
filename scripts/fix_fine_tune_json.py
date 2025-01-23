import argparse
import json
import re

def remove_prefix_and_trim_text(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    updated_data = {}
    for key, value in data.items():
        new_key = key.replace(".txt", "")
        new_key = new_key.lstrip("pytesseract")
        updated_data[new_key] = value

    with open(output_file, 'w') as f:
        json.dump(updated_data, f, indent=4, sort_keys=True)

def main():
    parser = argparse.ArgumentParser(description="Remove 'data/images/' prefix from dictionary keys in a JSON file and save to a new JSON file. Trim text before 'ASSISTANT:[/INST]' in dictionary values.")
    parser.add_argument('--input_file', dest='input_file', required=True, help='Path to the input JSON file containing the dictionary.')
    parser.add_argument('--output_file', dest='output_file', required=True, help='Path to the output JSON file to save the updated dictionary.')

    args = parser.parse_args()
    remove_prefix_and_trim_text(args.input_file, args.output_file)

if __name__ == "__main__":
    main()

