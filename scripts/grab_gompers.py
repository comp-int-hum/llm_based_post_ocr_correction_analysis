import argparse
import os
import pandas as pd
import requests

def download_image(url, path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(path, 'wb') as f:
            f.write(response.content)
    else:
        print(f"Failed to download image from {url}")

        
def save_text(text, path):
    with open(path, 'w') as f:
        f.write(text)

def process_csv(file_path, num_rows, images_dir, text_dir):
    # Create directories if they don't exist
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)
    
    # Read CSV file
    df = pd.read_csv(file_path)
    
    # Process each row up to the specified number
    for index, row in df.iterrows():
        if index >= num_rows:
            break
        
        # Generate file names
        file_name = row['DownloadUrl']
        file_name = file_name.split("mss")
        file_name = file_name[-1]
        file_name = file_name.split("/")
        file_name = file_name[0]
        print(file_name)
        image_path = os.path.join(images_dir, f"{file_name}.jpg")
        print(image_path)
        text_path = os.path.join(text_dir, f"{file_name}.txt")
        print(text_path)
        # Download and save image
        download_image(row['DownloadUrl'], image_path)
        
        # Save transcription text
        save_text(row['Transcription'], text_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process CSV file and save images and texts.')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the CSV file')
    parser.add_argument('--num_rows', type=int, required=True, help='Number of rows to process')
    parser.add_argument('--images_dir', type=str, required=True, help='Directory to save images')
    parser.add_argument('--text_dir', type=str, required=True, help='Directory to save texts')
    
    
    args = parser.parse_args()
    
    process_csv(args.csv_file, args.num_rows, args.images_dir, args.text_dir)
