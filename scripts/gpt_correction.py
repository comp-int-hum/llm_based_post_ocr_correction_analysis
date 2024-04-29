import requests
import argparse
import glob
import json 

parser = argparse.ArgumentParser()

parser.add_argument("--input", dest='data_directory', help="transcribed data")
parser.add_argument("--output", dest= "output_directory", help = "output file")
parser.add_argument("--gpt_version", dest= "gpt_version", help = "what version of gpt", default = "gpt-3.5-turbo")
parser.add_argument("--gpt_content", dest= "gpt_content", help = "instructions for gpt", default = "you are a helpful assistant, carefully fixing errors in documents")
parser.add_argument("--prompt", dest= "prompt", help = "what is your prompt for gpt?", default = "this is a historical text from a digitized archive. it has been created using optical character recognition, introducing numerous errors to a text that initially had none. without adding any new material, please correct the text by fixing the errors created by OCR")
parser.add_argument("--api_key", dest= "api_key", help = "where is your api key")
args = parser.parse_args()



with open(args.api_key, "r") as in_file:
    
    API_KEY = in_file.read() 

API_URL = 'https://api.openai.com/v1/chat/completions'

output_dictionary = {}

def chat_with_gpt(prompt, model= args.gpt_version):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }
    data = {
        "model": args.gpt_version,
        "messages": [
            {"role": "system", "content": args.gpt_content},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(API_URL, headers=headers, json=data)
    return response.json()


list_of_files = glob.glob("/home/sbacker2/projects/post_ocr_correction/work/pytesseract_output/*.txt")

for entry in list_of_files:
    with open(entry, "r") as in_file:
        text = in_file.read()
        new_prompt = args.prompt + text 
        response_file = chat_with_gpt(text)
        print(response_file)
        entry = entry.rstrip(".txt")
        entry = entry.lstrip("pytesseract")
        entry = entry + "corrected_by_{}".format(args.gpt_version)
        output_dictionary[entry] = response_file 

with open(args.output_directory, "w") as out_file:
    json.dump(output_dictionary, out_file)
