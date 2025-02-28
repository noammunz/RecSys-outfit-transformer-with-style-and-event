import json
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import random
from itertools import combinations

from src.models.Classifier import ImageEmbeddingClassifier

def load_dataset(dataset_path):
    """
    Load a Polyvore dataset from a JSON file
    
    Args:
        dataset_path: Path to the JSON dataset file
        
    Returns:
        The loaded dataset as a Python object
    """
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    return dataset

def extract_items(dataset):
    """
    Extract all unique items from the dataset with their category IDs and image paths
    
    Args:
        dataset: Loaded Polyvore dataset
        
    Returns:
        A dictionary mapping category IDs to lists of item tuples (item_id, categoryid, image_path)
    """
    items_by_category = {}
    item_set = set()  # To track unique items
    
    for outfit in dataset:
        for item in outfit["items"]:
            # Create a unique ID for the item using the image path
            item_id = item["image"].split("tid=")[-1]
            category_id = item["categoryid"]
            image_path = item["image"]
            
            # Skip if we've already seen this item
            if item_id in item_set:
                continue
                
            item_set.add(item_id)
            
            # Add the item to its category list
            if category_id not in items_by_category:
                items_by_category[category_id] = []
            
            items_by_category[category_id].append((item_id, category_id, image_path))
    
    return items_by_category

class ImageDataset(Dataset):
    """
    Dataset class for loading images from paths
    """
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            # Download the image from URL or load from disk
            if image_path.startswith('http'):
                import requests
                from io import BytesIO
                response = requests.get(image_path)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                image = Image.open(image_path).convert('RGB')
                
            # Apply transformations
            image_tensor = self.transform(image)
            return image_tensor, idx
            
        except Exception as e:
            # Return a blank image if loading fails
            print(f"Error loading image {image_path}: {e}")
            blank_image = torch.zeros(3, 224, 224)
            return blank_image, idx
    
def get_style_embeddings(items_by_category):
    """
    Get style embeddings for all items using the dataset class
    
    Args:
        items_by_category: Dictionary mapping category IDs to lists of item tuples
        
    Returns:
        Updated dictionary with item tuples including style embeddings
    """
    # Flatten all items into a single list for batch processing
    all_items = []
    for category_id, items in items_by_category.items():
        all_items.extend(items)
    
    # Extract all image paths
    image_paths = [item[2] for item in all_items]
    
    # Create dataset and get embeddings
    dataset = ImageDataset(image_paths)
    all_embeddings = get_style_embeddings_with_model(dataset)
    
    # Combine items with their embeddings
    items_with_embeddings = []
    for i, item in enumerate(all_items):
        item_id, category_id, image_path = item
        embedding = all_embeddings[i]
        items_with_embeddings.append((item_id, category_id, image_path, embedding))
    
    # Reorganize back into the category dictionary
    items_by_category_with_embeddings = {}
    for item in items_with_embeddings:
        item_id, category_id, image_path, embedding = item
        if category_id not in items_by_category_with_embeddings:
            items_by_category_with_embeddings[category_id] = []
        items_by_category_with_embeddings[category_id].append((item_id, category_id, image_path, embedding))
    
    return items_by_category_with_embeddings

def get_style_embeddings_with_model(dataset):
    """
    Function for getting style embeddings for a dataset of images
    
    Args:
        dataset: An ImageDataset instance
        
    Returns:
        List of style embeddings (numpy arrays)
    """
    # Configuration
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the style classifier model
    style_model = load_style_model()
    style_model.to(device)
    style_model.eval()
    
    # Create DataLoader for batching
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize embeddings list with the correct size
    all_embeddings = [None] * len(dataset)
    
    # Process in batches
    with torch.no_grad():
        for batch_images, batch_indices in tqdm(dataloader, desc="Computing style embeddings"):
            # Move batch to device
            batch_images = batch_images.to(device)
            
            # Get embeddings from model
            batch_embeddings = style_model.extract_features(batch_images)
            
            # Convert to numpy and store in the correct position
            batch_embeddings_np = batch_embeddings.cpu().numpy()
            
            for i, idx in enumerate(batch_indices):
                all_embeddings[idx] = batch_embeddings_np[i]
    
    return all_embeddings

def load_style_model(path="style_classifier.pth"):
    """
    Function for loading a pre-trained style classification model
    
    Returns:
        A PyTorch model for style classification
    """
    # Load the model from disk
    model = ImageEmbeddingClassifier()
    model.load_state_dict(torch.load(path))
    return model

import numpy as np
import random
from itertools import combinations

def compute_pairwise_distance(embedding1, embedding2):
    """
    Compute the Euclidean distance between two style embeddings
    
    Args:
        embedding1: First style embedding (numpy array)
        embedding2: Second style embedding (numpy array)
        
    Returns:
        Euclidean distance between the embeddings
    """
    # Calculate Euclidean distance between embeddings
    distance = np.linalg.norm(embedding1 - embedding2)
    return distance

def compute_outfit_coherence(outfit_items):
    """
    Calculate the total coherence score for an outfit based on pairwise distances
    
    Args:
        outfit_items: List of item tuples, each containing (item_id, category_id, image_path, embedding)
        
    Returns:
        Total coherence score (lower is more coherent)
    """
    # Extract embeddings from outfit items
    embeddings = [item[3] for item in outfit_items]
    
    # If outfit has fewer than 2 items, it has no pairwise distances
    if len(embeddings) < 2:
        return 0.0
    
    # Calculate pairwise distances between all items in the outfit
    total_distance = 0.0
    pair_count = 0
    
    # Iterate through all pairs
    for i, j in combinations(range(len(embeddings)), 2):
        pair_distance = compute_pairwise_distance(embeddings[i], embeddings[j])
        total_distance += pair_distance
        pair_count += 1
    
    # Return the sum of all pairwise distances
    return total_distance

def generate_random_outfit(items_by_category, min_size=4, max_size=7):
    """
    Generate a random outfit with no duplicate categories
    
    Args:
        items_by_category: Dictionary mapping category IDs to lists of item tuples
        min_size: Minimum number of items in the outfit
        max_size: Maximum number of items in the outfit
        
    Returns:
        List of item tuples forming an outfit
    """
    # Filter out categories that don't have any items
    valid_categories = [cat_id for cat_id, items in items_by_category.items() if items]
    
    # If we don't have enough valid categories, return None
    if len(valid_categories) < min_size:
        return None
    
    # Decide how many items to include in this outfit
    outfit_size = random.randint(min_size, min(max_size, len(valid_categories)))
    
    # Randomly select categories to include
    selected_categories = random.sample(valid_categories, outfit_size)
    
    # For each selected category, randomly choose one item
    outfit = []
    for category_id in selected_categories:
        item = random.choice(items_by_category[category_id])
        outfit.append(item)
    
    return outfit

def create_style_aware_dataset(items_by_category, num_outfits=1000000, min_size=4, max_size=7):
    """
    Main function to create the style-aware dataset by generating random outfits
    
    Args:
        items_by_category: Dictionary mapping category IDs to lists of item tuples
        num_outfits: Total number of outfit candidates to generate
        min_size: Minimum number of items in an outfit
        max_size: Maximum number of items in an outfit
        
    Returns:
        List of tuples (outfit, coherence_score) ranked by coherence
    """
    outfit_scores = []
    
    # Generate the specified number of random outfits
    with tqdm(total=num_outfits, desc="Generating outfit candidates") as pbar:
        for _ in range(num_outfits):
            # Generate a random outfit
            outfit = generate_random_outfit(items_by_category, min_size, max_size)
            
            # Skip if outfit generation failed
            if outfit is None:
                continue
            
            # Compute the coherence score
            coherence_score = compute_outfit_coherence(outfit)
            
            # Store the outfit and its score
            outfit_scores.append((outfit, coherence_score))
            
            pbar.update(1)
    
    return outfit_scores

def rank_and_select_outfits(outfit_scores, num_to_select=3000):
    """
    Sort outfits by coherence score and select the top N most coherent ones
    
    Args:
        outfit_scores: List of tuples (outfit, coherence_score)
        num_to_select: Number of top outfits to select
        
    Returns:
        List of the top N most coherent outfits
    """
    # Sort outfits by coherence score (lower is better)
    sorted_outfits = sorted(outfit_scores, key=lambda x: x[1])
    
    # Select the top N outfits
    top_outfits = sorted_outfits[:num_to_select]
    
    print(f"Selected top {num_to_select} outfits out of {len(outfit_scores)} candidates")
    print(f"Coherence score range: {top_outfits[0][1]:.4f} to {top_outfits[-1][1]:.4f}")
    
    # Return just the outfits without scores
    return [outfit for outfit, _ in top_outfits]

def save_dataset(outfits, output_path):
    """
    Save the final dataset to disk in JSON format
    
    Args:
        outfits: List of outfit items
        output_path: Path to save the dataset
    """
    # Convert outfits to a serializable format without embeddings
    final_outfits = []
    
    for outfit in outfits:
        final_items = []
        for item in outfit:
            item_id, category_id, image_path, _ = item  # Ignore the embedding
            
            final_items.append({
                'item_id': item_id,
                'category_id': category_id,
                'image_path': image_path
            })
        
        final_outfits.append({
            'items': final_items
        })
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the dataset
    with open(output_path, 'w') as f:
        json.dump(final_outfits, f, indent=2)
    
    print(f"Saved {len(outfits)} outfits to {output_path}")

def main():
    """
    Main function for generating style-aware outfit dataset
    """
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate a style-aware outfit dataset')
    
    # Required arguments
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to the input Polyvore dataset JSON file')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save the output dataset JSON file')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the pre-trained style classifier model')
    
    # Optional arguments with defaults
    parser.add_argument('--num_candidates', type=int, default=1000000,
                        help='Number of outfit candidates to generate (default: 1,000,000)')
    parser.add_argument('--num_outfits', type=int, default=3000,
                        help='Number of top outfits to select (default: 3,000)')
    parser.add_argument('--min_size', type=int, default=4,
                        help='Minimum number of items in an outfit (default: 4)')
    parser.add_argument('--max_size', type=int, default=7,
                        help='Maximum number of items in an outfit (default: 7)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    print(f"Loading dataset from {args.input_path}")
    # Load the dataset
    dataset = load_dataset(args.input_path)
    print(f"Loaded dataset with {len(dataset)} outfits")
    
    print("Extracting unique items and organizing by category...")
    # Extract items from the dataset
    items_by_category = extract_items(dataset)
    
    # Print category statistics
    total_items = sum(len(items) for items in items_by_category.values())
    print(f"Extracted {total_items} unique items across {len(items_by_category)} categories")
    
    print("Computing style embeddings for all items...")
    # Get style embeddings for all items
    items_by_category_with_embeddings = get_style_embeddings(items_by_category)
    
    print(f"Generating {args.num_candidates} outfit candidates...")
    # Generate random outfits and compute coherence scores
    outfit_scores = create_style_aware_dataset(
        items_by_category_with_embeddings, 
        num_outfits=args.num_candidates,
        min_size=args.min_size,
        max_size=args.max_size
    )
    
    print(f"Selecting top {args.num_outfits} most style-coherent outfits...")
    # Rank and select the top outfits
    selected_outfits = rank_and_select_outfits(outfit_scores, num_to_select=args.num_outfits)
    
    print(f"Saving final dataset to {args.output_path}...")
    # Save the final dataset
    save_dataset(selected_outfits, args.output_path)
    
    print("Dataset generation complete!")

if __name__ == "__main__":
    main()
