import json
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--input_docs", dest="input_docs", nargs ="*", help="input documents")
parser.add_argument("--control_file", dest="control_file", help="Ground truth file")
parser.add_argument("--output_file", dest = "output_folder", help = "where the output goes", required = True)
parser.add_argument("--condensed_output", dest ="condensed_output", help = "condensed output location")

args = parser.parse_args()



with open (args.control_file, "r") as in_file:
    control_dictionary = json.load(in_file)
    print(len(control_dictionary))

    
for element in control_dictionary:
    print(element)

    
counter = 0


for x in args.input_docs:
    with open (x, "r") as local_file:
        local_test_dict = json.load(local_file)
        print(len(local_test_dict))
        for key, item in local_test_dict.items():
            edited_key = key.split("corrected")
            edited_key = edited_key[0]
            print(edited_key)
            print("that was edited key")
            for control_key, control_items in control_dictionary.items():
                new_control_key = control_key.rstrip(".txt")
                print(new_control_key)
              #  print("that was control key")
                if new_control_key == edited_key:
               #     print("Matched!")
                    counter = counter + 1
                    
                    control_dictionary[control_key][key]=item
                    print(counter)


condensed_dictionary = {}
for text, contents  in control_dictionary.items():
     #for keys, items in text.items():
         
      #   print(keys)
       #  print(items)
    print(text)
    print("")
    print(contents)
    print(contents["pytesseract_text"])
    condensed_local_dictionary = {}
    condensed_local_dictionary["pytesseract_wer"] = contents["pytesseract_text"]["CER"]
    for item in contents:
        if "corrected_by" in item:
            new_name = item.split("corrected_by")
            new_name = new_name[-1]
            condensed_local_dictionary[new_name] = contents[item]["CER"]
    condensed_dictionary[text] = condensed_local_dictionary
    
with open(args.condensed_output, "w") as out_file:
    json.dump(condensed_dictionary, out_file)

with open (args.output_folder, "w") as out_file:
    json.dump(control_dictionary, out_file)
        
