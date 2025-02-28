import json
import os
import random
from tqdm import tqdm
from itertools import combinations

def load_dataset(dataset_path):
    """
    Load the Fashion4Events dataset from a JSON file
    
    Args:
        dataset_path: Path to the JSON dataset file
        
    Returns:
        The loaded dataset as a Python object
    """
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    return dataset

def extract_items_by_event(dataset):
    """
    Extract all items organized by event type and clothing category
    
    Args:
        dataset: Loaded Fashion4Events dataset
        
    Returns:
        A nested dictionary mapping event IDs to category IDs to lists of items
    """
    items_by_event_and_category = {}
    
    for item in dataset:
        event_id = item["eventid"]
        category_id = item["categoryid"]
        image_path = item["path"]
        item_id = os.path.basename(image_path)
        
        # Initialize event dictionary if needed
        if event_id not in items_by_event_and_category:
            items_by_event_and_category[event_id] = {}
        
        # Initialize category list if needed
        if category_id not in items_by_event_and_category[event_id]:
            items_by_event_and_category[event_id][category_id] = []
        
        # Add the item to its event and category
        items_by_event_and_category[event_id][category_id].append({
            "item_id": item_id,
            "event_id": event_id,
            "category_id": category_id,
            "image_path": image_path
        })
    
    return items_by_event_and_category

def generate_event_outfits(items_by_event_and_category, outfit_size_range=(2, 3)):
    """
    Create outfits for each event with different category items
    
    Args:
        items_by_event_and_category: Nested dictionary of items by event and category
        outfit_size_range: Tuple defining the min and max number of items in an outfit
        
    Returns:
        List of outfits, each containing 2-3 items from different categories
    """
    outfits = []
    
    # Process each event
    for event_id, categories in tqdm(items_by_event_and_category.items(), desc="Generating event outfits"):
        # Need at least 2 categories to form an outfit
        if len(categories) < outfit_size_range[0]:
            continue
        
        # Decide max outfit size for this event (limited by available categories)
        max_size = min(outfit_size_range[1], len(categories))
        
        # Get all category IDs for this event
        category_ids = list(categories.keys())
        
        # Create outfits until we exhaust the items
        while True:
            # Decide outfit size for this iteration
            outfit_size = random.randint(outfit_size_range[0], max_size)
            
            # Select random categories for this outfit
            selected_categories = random.sample(category_ids, outfit_size)
            
            # Check if we have any items left in the selected categories
            if any(len(categories[cat_id]) == 0 for cat_id in selected_categories):
                # If any category is empty, we can't create more outfits with these categories
                # Try different categories or break if we've exhausted all combinations
                continue
            
            # Create the outfit
            outfit_items = []
            for cat_id in selected_categories:
                # Get an item from this category and remove it from available items
                if categories[cat_id]:
                    item = categories[cat_id].pop()
                    outfit_items.append(item)
            
            # Only add valid outfits (with at least 2 items)
            if len(outfit_items) >= outfit_size_range[0]:
                outfits.append(outfit_items)
            
            # If we've used up too many categories, break
            if sum(len(items) for items in categories.values()) < outfit_size_range[0]:
                break
    
    return outfits

def save_dataset(outfits, output_path):
    """
    Save the final dataset to disk in JSON format
    
    Args:
        outfits: List of outfit items
        output_path: Path to save the dataset
    """
    # Convert outfits to the desired format
    final_outfits = []
    
    for outfit in outfits:
        # Get the event ID from the first item (all items in an outfit have the same event ID)
        event_id = outfit[0]["event_id"]
        
        outfit_dict = {
            "event_id": event_id,
            "items": outfit
        }
        
        final_outfits.append(outfit_dict)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the dataset
    with open(output_path, 'w') as f:
        json.dump(final_outfits, f, indent=2)
    
    print(f"Saved {len(outfits)} outfits to {output_path}")

def main():
    """
    Main function for generating event-aware outfit dataset
    """
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate an event-aware outfit dataset')
    
    # Required arguments
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to the input Fashion4Events dataset JSON file')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save the output dataset JSON file')
    
    # Optional arguments with defaults
    parser.add_argument('--min_size', type=int, default=2,
                        help='Minimum number of items in an outfit (default: 2)')
    parser.add_argument('--max_size', type=int, default=3,
                        help='Maximum number of items in an outfit (default: 3)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    print(f"Loading dataset from {args.input_path}")
    # Load the dataset
    dataset = load_dataset(args.input_path)
    print(f"Loaded dataset with {len(dataset)} items")
    
    print("Organizing items by event and category...")
    # Extract items organized by event and category
    items_by_event_and_category = extract_items_by_event(dataset)
    
    # Print statistics
    event_count = len(items_by_event_and_category)
    total_items = sum(sum(len(items) for items in event_cats.values()) 
                      for event_cats in items_by_event_and_category.values())
    print(f"Organized {total_items} items across {event_count} events")
    
    print("Generating event-aware outfits...")
    # Generate outfits for each event
    outfits = generate_event_outfits(
        items_by_event_and_category,
        outfit_size_range=(args.min_size, args.max_size)
    )
    
    print(f"Generated {len(outfits)} event-aware outfits")
    
    print(f"Saving dataset to {args.output_path}...")
    # Save the dataset
    save_dataset(outfits, args.output_path)
    
    print("Dataset generation complete!")

if __name__ == "__main__":
    main()