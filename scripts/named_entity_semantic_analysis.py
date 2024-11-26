import json
import argparse
import os
os.environ['TRANSFORMERS_CACHE'] = '/home/sbacker2/scr4_lhyman6/sbacker2/.cache/huggingface'
os.environ['TORCH_HOME'] = '/home/sbacker2/scr4_lhyman6/sbacker2/.cache/torch'
import torch
from transformers import BertTokenizerFast, BertModel
import logging
import numpy as np 
logging.basicConfig(filename='script.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s', filemode = "w")

logging.info("Script started")

parser = argparse.ArgumentParser()

parser.add_argument("--input_file", dest='input_file', help="ner data")

parser.add_argument("--output_file", dest= "output_file", help = "output")

args = parser.parse_args()

logging.info("Loading BERT model and tokenizer")
tokenizer = BertTokenizerFast.from_pretrained('bert-large-uncased')
model = BertModel.from_pretrained('bert-large-uncased')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

model = model.float()

with open(args.input_file, "r") as in_file:
    local_file = json.load(in_file)

    
# Function to get contextual embeddings for each entity
def get_entity_embedding(text, entity_span):
    # Tokenize the full text with BERT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logging.debug(f"Running model on entity with span: {entity_span}")

    # Log the actual entity text based on the span
    entity_text = text[entity_span[0]:entity_span[1]]
    logging.debug(f"Entity text from span [{entity_span[0]}:{entity_span[1]}]: {entity_text}")
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, return_offsets_mapping=True)
    inputs.to(device)
    
    # Extract the token offsets from the tokenized text
    offset_mapping = inputs['offset_mapping'][0].cpu().numpy()

    inputs = {key: val for key, val in inputs.items() if key != 'offset_mapping'}
    
    # Initialize variables for start and end token indices
    start_token = None
    end_token = None

    # Find the token indices corresponding to the entity's character span
    for i, (start, end) in enumerate(offset_mapping):
        if start <= entity_span[0] < end:  # Start of entity is within this token
            start_token = i
        if start < entity_span[1] <= end:  # End of entity is within this token
            end_token = i
            break  # Once we find the end token, we can stop

    # If valid token spans are found, proceed with embedding extraction
    if start_token is not None and end_token is not None:
        logging.debug(f"Entity tokens: {inputs['input_ids'][0][start_token:end_token+1]}")



        num_tokens = len(inputs['input_ids'][0])
        logging.debug(f"Tokenized text: {num_tokens}")
        print(f"Tokenized text: {num_tokens}")

        #logging.debug(f"Entity span: {entity_span}")
        #print(f"Entity span: {entity_span}")

        #logging.debug(f"Length of tokenized text: {len(inputs)}")

        #print(f"Number of tokens: {num_tokens}")


    
        # Pass the tokenized text through the model
        with torch.no_grad():
            outputs = model(**inputs)
    
            # Get the hidden states (last hidden layer)
            last_hidden_state = outputs.last_hidden_state
    
            # Extract embeddings for the tokens corresponding to the entity span
            entity_embedding = last_hidden_state[:, start_token:end_token+1].mean(dim=1).cpu().numpy().tolist() 
            
            return entity_embedding
    else:
        logging.warning(f"Invalid entity span: {entity_span} could not be mapped to tokens")
        return None

def add_embeddings_to_ner(ner_data):
    text = ner_data["text"]
    if ner_data.get("NER_List") and len(ner_data["NER_List"]) > 0:  
        ner_list = ner_data["NER_List"]
        
        # Iterate through each entity in the NER_List and add embeddings
        for entity in ner_list:
            #print(entity)
            for ner in entity:
                #print(ner)
                entity_text = ner["entity"]
                entity_span = ner["span"]
                logging.debug("test")
                logging.debug(f"Processing entity: {ner['entity']} with span: {entity_span}")
                # Get the embedding for the entity
                embedding = get_entity_embedding(text, entity_span)

                # Check if embedding is None
                if embedding is None:
                    logging.warning(f"Failed to get embedding for entity: {entity_text} with span: {entity_span}")
                    ner["embedding"] = None  # Handle the case where embedding couldn't be retrieved
                    continue  # Skip this entity and move to the next one

                new_embedding = []

                
                for x in embedding:
                    # Check if x is a string representation of "nan"
                    if isinstance(x, str) and x.strip().lower() == "nan":
                        x = None  # Replace the string "nan" with None
                    # Check if x is a numeric NaN value
                    elif isinstance(x, float) and np.isnan(x):
                        x = None  # Replace numeric NaN with None
                    new_embedding.append(x)
                logging.debug(f"Processed embedding: {new_embedding}")    
                # Add the embedding to the entity dictionary
                ner["embedding"] = new_embedding

                #print(ner)
        return ner_list
        #print("returned")
    else:
        blank = []
        return blank



for doc_name, doc_contents in local_file.items():
    logging.info(f"Processing document: {doc_name}")
    for style_name, style_content in doc_contents.items():
        logging.info(f"Processing style: {style_name}")
        
        style_content["NER_List"] = add_embeddings_to_ner(style_content)
        #print(style_content)


with open (args.output_file, "w") as out_file:
    json.dump(local_file, out_file, indent = 4)

    
