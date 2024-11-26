import requests
import argparse
import glob
import json 

parser = argparse.ArgumentParser()

parser.add_argument("--input", dest='data_directory', help="transcribed data")
parser.add_argument("--output", dest= "output_directory", help = "output file")
parser.add_argument("--gpt_version", dest= "gpt_version", help = "what version of gpt", default = "gpt-3.5-turbo")
parser.add_argument("--gpt_content", dest= "gpt_content", help = "instructions for gpt", default = "you are a helpful assistant, carefully fixing errors in documents")
parser.add_argument("--prompt", dest= "prompt", help = "what is your prompt for gpt?", nargs = "*")
parser.add_argument("--api_key", dest= "api_key", help = "where is your api key")

args = parser.parse_args()


print(args.api_key)
print("that was the key")

with open(args.api_key, "r") as in_file:
    API_KEY = in_file.read() 
    print(API_KEY)
    print("THAT WAS THE KEY")
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
    result = response.json()
    result['used_prompt'] = prompt 
    return result


print("this is the data directory")
print(args.data_directory)
with open (args.data_directory, "r") as in_file:
    pytesseract_material = json.load(in_file)

for name, text in pytesseract_material.items():
        prompt = " ".join(args.prompt)
        prompt = prompt + text 
        response_file = chat_with_gpt(prompt)
        print(response_file)
        if "choices" in response_file:
            text_file = response_file["choices"][0]["message"]["content"]

            name = name.rstrip(".txt")
            name = name.lstrip("pytesseract")
            entry = name + "corrected_by_{}".format(args.gpt_version)
            output_dictionary[entry] = text_file 

with open(args.output_directory, "w") as out_file:
    json.dump(output_dictionary, out_file)
