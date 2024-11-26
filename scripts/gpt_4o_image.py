import base64
import requests
import argparse
import glob
import json

parser = argparse.ArgumentParser()

parser.add_argument("--input_directory", dest='data_directory', help="image location")
parser.add_argument("--output", dest= "output_directory", help = "output file")
parser.add_argument("--gpt_content", dest= "gpt_content", help = "instructions for gpt", default = "you are a helpful assistant, carefully fixing errors in documents")
parser.add_argument("--prompt", dest= "prompt", help = "what is your prompt for gpt?", nargs = "*")
parser.add_argument("--api_key", dest= "api_key", help = "where is your api key")
args = parser.parse_args()



with open(args.api_key, "r") as in_file:
    api_key = in_file.read().strip()  # Strip any surrounding whitespace or newlines


prompt = " ".join(args.prompt)


# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
#image_path = "path_to_your_image.jpg"

name_list = glob.glob(args.data_directory + "/*" )
output_dictionary = {}
for x in name_list:

    print(x)

    # Getting the base64 string
    base64_image = encode_image(x)
    print("got base image")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt 
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1000
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    response_file = response.json()
    print(response_file)
    if "choices" in response_file:
            print("got chosen")
            text_file = response_file["choices"][0]["message"]["content"]
            print(text_file)
            name= x.split("images/")
            name = name[-1]
            name = name.rstrip(".jpg")
            entry = name + "corrected_by_gpt_4o_image_to_text"
            print(entry)
            print(response.json())

            output_dictionary[entry] = text_file

            
with open (args.output_directory, "w") as out_file:
    json.dump(output_dictionary, out_file)
