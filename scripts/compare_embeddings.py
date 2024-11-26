import argparse
import json
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from fuzzywuzzy import fuzz
import logging

# Configure logging
logging.basicConfig(filename='embedding_errors.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

SIMILARITY_THRESHOLD = 80

def jaccard_similarity(set1, set2):
    """Compute Jaccard Similarity, considering only entities in the OCR that are also in the control set."""
    set1 = set(set1)  # Convert to set for easier comparison
    set2 = set(set2)

    #print("this is the union")
    
    # Calculate Jaccard Similarity
    intersection = set1.intersection(set2)
    union = set2
    #print(union)
    #print(len(union))
    if len(union) == 0:
        return 0  # Avoid division by zero
    return len(intersection) / len(union)

# Function to calculate Cosine similarity between two embeddings
def cosine_sim2(embedding1, embedding2):
    #print(embedding2)
    """Compute Cosine Similarity between two embeddings, handling NaN values."""
    # Ensure embeddings are in numpy array format
    embedding1 = np.array(embedding1).reshape(1, -1)
    embedding2 = np.array(embedding2).reshape(1, -1)
    # Check if either embedding contains NaN values
    if np.isnan(embedding1).any() or np.isnan(embedding2).any():
        return None  # Return None or a default value if NaNs are present

    return cosine_similarity(embedding1, embedding2)[0][0]

def cosine_sim(embedding1, embedding2):
    """Compute Cosine Similarity between two embeddings, handling NaN values and identifying non-numeric values."""
    #print("Checking embedding2:", embedding2)

    # Try converting embeddings to numpy arrays with float type
    try:
        embedding1 = np.array(embedding1, dtype=float).reshape(1, -1)
        embedding2 = np.array(embedding2, dtype=float).reshape(1, -1)
    except ValueError as e:
        # If there's a ValueError, log the non-numeric values
        #print("Non-numeric values found in embeddings:")
        #for value in embedding1:
         #   if not isinstance(value, (int, float)):
                #print("Non-numeric in embedding1:", value)
        #for value in embedding2:
         #   if not isinstance(value, (int, float)):
                #print("Non-numeric in embedding2:", value)

        return None

    # Check for NaN values in the numeric embeddings
    if np.isnan(embedding1).any() or np.isnan(embedding2).any():
        print("NaN values found in embeddings.")
        return None

    # Calculate cosine similarity
    return cosine_similarity(embedding1, embedding2)[0][0]


def compare_entities_and_embeddings(ocr_data, control_data):
    """Compare the named entities and their embeddings between OCR and control texts."""
    print("started comparing")
    # Extract NER List for both OCR-corrected text and control text
    ocr_entities = ocr_data["NER_List"]
    control_entities = control_data["NER_List"]
    #print(len(control_entities))
    #print(control_entities[0])
    ocr_dict = {}
    control_dict = {}

    matched_entity_list = []

    for entity in ocr_entities:
        entity = entity[0]
        ocr_dict[entity["entity"]] = entity.get("embedding")
    for thing in control_entities[0]:
        #print("this is thing")
        #print(thing)
        #print(" ")
        #thing = thing[0]
        #print("this is thing of zero")
        #print(thing)
        control_dict[thing["entity"]] = thing.get("embedding")
        
    # Create dictionaries to map entity names to their embeddings
    #ocr_dict = {entity["entity"]: entity.get("embedding") for entity in ocr_entities}
    #control_dict = {entity["entity"]: entity.get("embedding") for entity in control_entities}

    # Jaccard Similarity for Named Entities (set comparison of keys)
    jaccard_score_entities = jaccard_similarity(ocr_dict.keys(), control_dict.keys())

    # Cosine Similarity for Embeddings (only for matching entities)
    error_list = []
    cosine_scores = []
    for entity_name in ocr_dict:
        best_match = None
        best_score = 0

        for control_name in control_dict:
            similarity_score = fuzz.ratio(entity_name, control_name)
            if similarity_score > best_score and similarity_score >= SIMILARITY_THRESHOLD:
                best_match = control_name
                best_score = similarity_score

                
        if best_match:
           ocr_embedding = ocr_dict[entity_name]
           control_embedding = control_dict[best_match]

        
        
           sim_score = cosine_sim(ocr_embedding, control_embedding)
           if sim_score is not None:  # Filter out None values
               cosine_scores.append(sim_score)
           else:
               local_error = [[entity_name, ocr_embedding],[best_match,control_embedding]]
               error_list.append(local_error)
           local_list = [entity_name, best_match]
           matched_entity_list.append(local_list)
    # Calculate average cosine similarity only for valid scores
    avg_cosine_similarity = np.mean(cosine_scores) if cosine_scores else 0

    return {
        "jaccard_similarity": jaccard_score_entities,
        "avg_cosine_similarity": avg_cosine_similarity,
        "num_entities": len(ocr_dict),
        "matched_entities" : matched_entity_list,# Optional: Track number of entities in OCR data
        "match_percentage": len(matched_entity_list) / len(control_dict),
        "cosine_scores_count": len(cosine_scores),  # Optional: Track how many embeddings were successfully compared
        "ocr_named_entities": list(ocr_dict.keys()),
        "control_named_entities": list(control_dict.keys()),
        "error_list": error_list
    }

# Main function to process inputs and output results
def main(args):
    # Load the input JSON file containing all OCR-corrected and control texts
    with open(args.input_file, 'r') as f:
        data = json.load(f)

    # Dictionary to store results for each document and method
    results = {}

    # Loop through each document in the data
    for document_id, document_data in data.items():
        control_data = document_data.get("control_text", {})
        #print("grabbed control text")
        # Dictionary to hold results for different OCR methods for the current document

        document_results = {}
        
        # Compare each OCR method with control text
        for method, ocr_data in document_data.items():
            if method != "control_text":  # Skip control text comparison with itself
         #       print("this is the method")
                #print(method)
                scores = compare_entities_and_embeddings(ocr_data, control_data)
                document_results[method] = scores
        
        results[document_id] = document_results

    # Write results to output JSON file
    with open(args.output_file, 'w') as f_out:
        json.dump(results, f_out, indent=4)


# Argument parser setup
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Named Entities and Embeddings between OCR and Control Texts.")
    parser.add_argument("--input_file", dest='input_file', type=str, help="Path to input JSON file containing OCR-corrected and control texts.")
    parser.add_argument("--output_file", dest = "output_file", type=str, help="Path to output JSON file to save the similarity scores.")
    
    args = parser.parse_args()
    main(args)
