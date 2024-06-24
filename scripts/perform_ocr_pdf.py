from PIL import Image
import fitz
import glob
import pytesseract
import json
import argparse
import os
import io 

parser = argparse.ArgumentParser()

parser.add_argument("--input_file", dest='input_file', help="input_file_folder")
parser.add_argument("--output_file", dest= "output_file", help = "output_ocr_text")
args = parser.parse_args()

pdf_path = args.input_file
output_file_path = args.output_file



pytesseract.pytesseract.tesseract_cmd = r'/data/lhyman6/programs/tesseract/tesseract/bin/tesseract'
os.environ['TESSDATA_PREFIX'] = '/data/lhyman6/programs/tesseract/tesseract/tessdata'

#tessdata_prefix = '/home/sbacker2/data_lhyman6/tesseract/share'
#os.environ['TESSDATA_PREFIX'] = tessdata_prefix
#pytesseract.pytesseract.tesseract_cmd = os.path.expanduser('~/local/bin/tesseract')

#images = convert_from_path(pdf_path)

pdf_document = fitz.open(pdf_path)

output_dictionary = {}


# Perform OCR on each image


for page_num in range(len(pdf_document)):
    page = pdf_document.load_page(page_num)
    pix = page.get_pixmap()
    print("got a new page")
    # Convert Pixmap to PIL Image
    image = Image.open(io.BytesIO(pix.tobytes()))

    # Perform OCR on the image
    text = pytesseract.image_to_string(image)

    # Store the OCR result in the dictionary
    output_dictionary[f'page_{page_num + 1}'] = text



with open(args.output_file, "w") as out_file:
            json.dump(output_dictionary,out_file, indent=4)
            
