import json
import os
import random
import argparse
from tqdm import tqdm

def load_dataset(dataset_path):
    """
    Load a dataset from a JSON file
    
    Args:
        dataset_path: Path to the JSON dataset file
        
    Returns:
        The loaded dataset as a Python object
    """
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    return dataset

def extract_style_info(style_dataset):
    """
    Extract style information from the style dataset
    
    Args:
        style_dataset: The loaded style dataset
        
    Returns:
        A dictionary mapping category IDs to item lists with style info
    """
    style_items_by_category = {}
    
    for outfit in style_dataset:
        for item in outfit["items"]:
            category_id = item["category_id"]
            
            if category_id not in style_items_by_category:
                style_items_by_category[category_id] = []
            
            style_items_by_category[category_id].append(item)
    
    return style_items_by_category

def extract_event_info(event_dataset):
    """
    Extract event information from the event dataset
    
    Args:
        event_dataset: The loaded event dataset
        
    Returns:
        A dictionary mapping event IDs to outfits
    """
    event_outfits = {}
    
    for outfit in event_dataset:
        event_id = outfit["event_id"]
        
        if event_id not in event_outfits:
            event_outfits[event_id] = []
        
        event_outfits[event_id].append(outfit)
    
    return event_outfits

def merge_datasets(style_items_by_category, event_outfits, num_outfits=300):
    """
    Merge style and event datasets to create a combined dataset
    
    Args:
        style_items_by_category: Dictionary mapping category IDs to style items
        event_outfits: Dictionary mapping event IDs to event outfits
        num_outfits: Number of merged outfits to create
        
    Returns:
        List of merged outfits with both style and event information
    """
    merged_outfits = []
    all_events = list(event_outfits.keys())
    
    # Create a progress bar
    with tqdm(total=num_outfits, desc="Creating merged outfits") as pbar:
        # Generate the specified number of merged outfits
        while len(merged_outfits) < num_outfits:
            # Select a random event
            if not all_events:
                print("Warning: Ran out of events to sample from")
                break
                
            event_id = random.choice(all_events)
            
            # Get a random outfit for this event
            if not event_outfits[event_id]:
                all_events.remove(event_id)
                continue
                
            event_outfit = event_outfits[event_id].pop()
            
            # Get the categories present in this event outfit
            event_categories = [item["category_id"] for item in event_outfit["items"]]
            
            # Find matching style items for each category
            style_items = []
            valid_outfit = True
            
            for category_id in event_categories:
                # If we don't have style items for this category, skip this outfit
                if category_id not in style_items_by_category or not style_items_by_category[category_id]:
                    valid_outfit = False
                    break
                
                # Get a random style item for this category
                style_item = random.choice(style_items_by_category[category_id])
                style_items_by_category[category_id].remove(style_item)
                style_items.append(style_item)
            
            # If we couldn't find style items for all categories, skip this outfit
            if not valid_outfit:
                continue
            
            # Create a merged outfit with both style and event information
            merged_outfit = {
                "event_id": event_id,
                "items": []
            }
            
            # Combine style and event item information
            for i, event_item in enumerate(event_outfit["items"]):
                style_item = style_items[i]
                
                merged_item = {
                    "item_id": style_item["item_id"],
                    "category_id": event_item["category_id"],
                    "event_id": event_id,
                    "image_path": style_item["image_path"],
                    "event_image_path": event_item["image_path"]
                }
                
                merged_outfit["items"].append(merged_item)
            
            merged_outfits.append(merged_outfit)
            pbar.update(1)
            
            # If we've used all event outfits, break
            if all(len(outfits) == 0 for outfits in event_outfits.values()):
                print(f"Used all event outfits. Generated {len(merged_outfits)} merged outfits.")
                break
    
    return merged_outfits

def save_dataset(outfits, output_path):
    """
    Save the merged dataset to disk in JSON format
    
    Args:
        outfits: List of merged outfits
        output_path: Path to save the dataset
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the dataset
    with open(output_path, 'w') as f:
        json.dump(outfits, f, indent=2)
    
    print(f"Saved {len(outfits)} merged outfits to {output_path}")

def main():
    """
    Main function for merging style and event datasets
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Merge style and event datasets')
    
    # Required arguments
    parser.add_argument('--style_path', type=str, required=True,
                        help='Path to the style dataset JSON file')
    parser.add_argument('--event_path', type=str, required=True,
                        help='Path to the event dataset JSON file')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save the output merged dataset JSON file')
    
    # Optional arguments
    parser.add_argument('--num_outfits', type=int, default=5000,
                        help='Number of merged outfits to create (default: 5000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    print(f"Loading style dataset from {args.style_path}")
    style_dataset = load_dataset(args.style_path)
    print(f"Loaded style dataset with {len(style_dataset)} outfits")
    
    print(f"Loading event dataset from {args.event_path}")
    event_dataset = load_dataset(args.event_path)
    print(f"Loaded event dataset with {len(event_dataset)} outfits")
    
    # Extract style and event information
    print("Extracting style information...")
    style_items_by_category = extract_style_info(style_dataset)
    
    print("Extracting event information...")
    event_outfits = extract_event_info(event_dataset)
    
    # Print statistics
    style_categories = len(style_items_by_category)
    style_items = sum(len(items) for items in style_items_by_category.values())
    event_types = len(event_outfits)
    event_outfits_count = sum(len(outfits) for outfits in event_outfits.values())
    
    print(f"Style dataset: {style_items} items across {style_categories} categories")
    print(f"Event dataset: {event_outfits_count} outfits across {event_types} event types")
    
    # Merge the datasets
    print(f"Merging datasets to create {args.num_outfits} outfits...")
    merged_outfits = merge_datasets(
        style_items_by_category,
        event_outfits,
        num_outfits=args.num_outfits
    )
    
    # Save the merged dataset
    print(f"Saving merged dataset to {args.output_path}...")
    save_dataset(merged_outfits, args.output_path)
    
    print("Dataset merging complete!")

if __name__ == "__main__":
    main()