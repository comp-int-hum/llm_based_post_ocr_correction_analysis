import spacy
import argparse 
import json

# Load spaCy's English NER model
nlp = spacy.load("en_core_web_sm")

parser = argparse.ArgumentParser()

parser.add_argument("--input_file", dest='input_file', help="the input file of ocr_text")

parser.add_argument("--output_file", dest= "output_file", help = "output location")

args = parser.parse_args()

nlp = spacy.load("en_core_web_lg")

if 'parser' not in nlp.pipe_names:
    print("Parser not found, adding a parser")
    nlp.add_pipe("parser", last=True)  # Add parser if it's not present

local_dictionary = {}

with open(args.input_file, "r") as in_file:
    local_dictionary = json.load(in_file)

print(nlp.pipe_names)
    
    
for document_name, document in local_dictionary.items():
    for version_name, version in document.items():
        if version_name == "control_text": 
            sentence_list = []
            print(type(version))
            doc = version
            document[version_name] = {}
            document[version_name]["text"] = doc
            sent_list = []
            doc_sent = nlp(doc)
            #for sent in doc_sent.sents:
             #   sent_list.append(sent.text)

            #document[version_name]["sentences"] = sent_list
            ner_list = []
            #for x in doc_sent.sents:
            local_list = [] 
            for token in doc_sent.ents:
                local_list.append({"entity" : token.text, "span" : [token.start_char, token.end_char], "label" : token.label_})
            ner_list.append(local_list)    
            document[version_name]["NER_List"] = ner_list
        else:
            sent_list = []
            doc = version["text"]
            doc_sent = nlp(doc)
            #for sent in doc_sent.sents:
             #   print(sent.text)
              #  sent_list.append(sent.text)
            #version["sentences"] = sent_list
            ner_list = []
            for token in doc_sent.ents:
                local_list = []
                
                local_list.append({"entity" : token.text, "span" : [token.start_char, token.end_char], "label" : token.label_})
                ner_list.append(local_list)
            version["NER_List"] = ner_list
with open(args.output_file, "w") as out_file:
    json.dump(local_dictionary, out_file, indent=4)
