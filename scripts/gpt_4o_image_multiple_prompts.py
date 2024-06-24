import base64
import requests
import argparse
import glob
import json

parser = argparse.ArgumentParser()

parser.add_argument("--input_directory", dest='data_directory', help="image location")
parser.add_argument("--output", dest= "output_directory", help = "output file")
#parser.add_argument("--gpt_content", dest= "gpt_content", help = "instructions for gpt", default = "you are a helpful assistant, carefully fixing errors in documents")
parser.add_argument("--prompt", dest= "prompt", help = "what is your prompt for gpt?", nargs = "*")
parser.add_argument("--api_key", dest= "api_key", help = "where is your api key")
args = parser.parse_args()



with open(args.api_key, "r") as in_file:

    api_key = in_file.read()

#prompt = " ".join(args.prompt)


# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
#image_path = "path_to_your_image.jpg"

prompt_list = ["Please return all of the text contained in this  document. DO NOT include any commentary, discussion, or description of the image besides whatever text is included in it. DO NOT add any words or phrases not present in the document. Take your time.", "This is a historical document that we are translating into text. Please return all of the text from this document, ommitting nothing. Please do NOT include any of your own descriptions of the document or  discussion and analysis of its contents." "You are a careful, precise reader. Please provide a transcription of all of the words in this historical document, ommitting nothing. Do not add any words or discussion besides what is in the document", "please transcribe all the text from this historical document. DO NOT include any commentary, discussion, or description. Only return the exact text found in the document, without adding or omitting any words.", "This is a historical document. Transcribe all the text exactly as it appears in the document. Do not include any of your own words, descriptions, or analyses. Only the text in the document should be provided.", "As a careful and precise reader, transcribe every word in this historical document. Do not add any commentary, descriptions, or analysis. Only the text in the document should be included in your response.", "Transcribe the entire text of this historical document. Ensure no additional commentary, descriptions, or extra words are included. Provide only the exact text from the document."]
 


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
    for prompt in prompt_list: 
        print(prompt)
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
        #print(response_file)
        if "choices" in response_file:
            print("got chosen")
            text_file = response_file["choices"][0]["message"]["content"]
            local_dict = {prompt: text_file}   
            print(text_file)
            name= x.split("images/")
            name = name[-1]
            entry = name + "corrected_by_gpt_4o_image_to_text"
            print(entry)
            print(response.json())

            if output_dictionary.get(entry) != None:
                output_dictionary[entry].update(local_dict)

            else:
                output_dictionary[entry] = local_dict
with open (args.output_directory, "w") as out_file:
    json.dump(output_dictionary, out_file)
