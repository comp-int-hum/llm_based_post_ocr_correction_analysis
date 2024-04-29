import json
import requests
import time
import os 

url = "https://www.loc.gov/item/mss154040167/"
basepath = "/home/sbacker2/projects/post_ocr_correction/data/"
total_page_image = []

params = {"fo" : "json"}
#full_url = f"{url}?sp={params['sp']}&st={params['st']}&fo={params['fo']}"
#print("Requesting URL:", full_url)

#this is the code to generate the initial list                                                                                                                                        
a_response = requests.get(url, params = params)
initial_data = a_response.json()
dict_of_images = {}
dict_of_texts = {}
initial_content = initial_data['resources'][0]["files"]
for x in initial_content:
    good_jpg = x[2]["url"]
    good_text = x[-1]["fulltext"]
    #print(good_text)
    name = good_jpg.split("iiif")
    name = name[1]
    name = name.split("mss")
    name = name[-1]
    name = name.split("/full")
    name = name[0]
    print(name)
    dict_of_images[name] = good_jpg
    name = name + ".txt"
    dict_of_texts[name] = good_text

for item, key in dict_of_images.items():

    r = requests.get(key)
    time.sleep(1)
    if r.status_code == 200:
        with open ("/home/sbacker2/projects/post_ocr_correction/data/{}".format(item), "wb") as the_file:
            the_file.write(r.content)
            print("wrote {}".format(item))

for item, key in dict_of_texts.items():
     with open ("/home/sbacker2/projects/post_ocr_correction/data/texts/{}".format(item), "w") as the_file:  
            the_file.write(key)                                              
