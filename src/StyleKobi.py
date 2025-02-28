import cv2
import math
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans
from src.data.datasets import polyvore

def extract_dominant_colors(image_path, n_colors: int = 3):
    """
    Extracts the dominant colors from an image using KMeans clustering.
    
    Args:
        image (np.ndarray): Input image (H, W, 3) in RGB.
        n_colors (int): Number of clusters to form (default is 3).
    
    Returns:
        np.ndarray: Array of shape (3, 3) with the dominant RGB colors.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(pixels)
    dominant_colors = kmeans.cluster_centers_
    return tuple(map(tuple, dominant_colors))

def match_triplet_to_style(target_triplet):
    """
    Given an extracted triplet from an outfit, find the closest predefined
    Kobayashi style triplet using Euclidean distance.

    Args:
        target_triplet (list of tuple): A list of three (R, G, B) colors extracted from an image.

    Returns:
        str: The closest matching style name.
    """
    best_style = None
    best_distance = float("inf")
    style_to_triplet = {
        1:    [(255, 182, 193), (255, 105, 180), (255, 160, 122)],  #"PRETTY" Light Pink, Barbie Pink, Light Coral
        0:    [(216, 191, 216), (230, 230, 250), (255, 228, 225)],  #"ROMANTIC" Thistle, Lavender, Misty Rose
        4:    [(210, 180, 140), (222, 184, 135), (244, 164, 96)],   #"NATURAL" Tan, Burlywood, Sandy Brown
        6:    [(245, 222, 179), (255, 248, 220), (255, 239, 213)],  #"ELEGANT" Wheat, Cornsilk, Papaya Whip
        2:    [(240, 255, 255), (224, 255, 255), (255, 250, 250)],  #"CLEAR" Azure, Light Cyan, Snow
        3:    [(176, 224, 230), (135, 206, 235), (173, 216, 230)],  #"COOL-CASUAL" Powder Blue, Sky Blue, Light Blue
        11:   [(255, 69, 0), (255, 0, 0), (139, 0, 0)],             #"DYNAMIC" Red-Orange, Red, Dark Red
        13:   [(255, 20, 147), (255, 105, 180), (255, 0, 255)],     #"GORGEOUS" Deep Pink, Barbie Pink, Magenta
        12:   [(128, 0, 0), (139, 69, 19), (34, 139, 34)],          #"ETHNIC" Maroon, Saddle Brown, Forest Green
        8:    [(189, 183, 107), (188, 143, 143), (205, 133, 63)],   #"CHIC" Dark Khaki, Rosy Brown, Peru
        7:    [(255, 255, 0), (255, 215, 0), (173, 216, 230)],      #"MODERN" Yellow, Gold, Light Blue
        10:   [(0, 0, 0), (105, 105, 105), (169, 169, 169)],        #"CLASSIC" Black, Dim Gray, Dark Gray
        9:    [(85, 107, 47), (47, 79, 79), (139, 69, 19)],         #"DANDY" Dark Olive Green, Dark Slate Gray, Saddle Brown
        5:    [(47, 79, 79), (0, 0, 139), (0, 0, 0)],               #"FORMAL" Dark Slate Gray, Dark Blue, Black
    }

    for style, triplet in style_to_triplet.items():
        # Test all permutations of the target triplet to allow any order
        for perm in itertools.permutations(target_triplet, 3):
            distance = sum(
                math.sqrt(
                    (perm[i][0] - triplet[i][0]) ** 2 +
                    (perm[i][1] - triplet[i][1]) ** 2 +
                    (perm[i][2] - triplet[i][2]) ** 2
                ) for i in range(3)
            )
            if distance < best_distance:
                best_distance = distance
                best_style = style

    return best_style

def get_style(v):
    domain_color = extract_dominant_colors(v)  # Extract colors
    best_style = match_triplet_to_style(domain_color)  # Get style
    return best_style


polyvore_dir = '/home/nogaschw/outfit-transformer-main/Fashion Rec -Style and Event - Data/polyvore_images.csv'

# Read the CSV file
polyvore_df = pd.read_csv(polyvore_dir)

# Apply the get_style function to the image_path column
polyvore_df['style'] = polyvore_df['image_path'].apply(get_style)
polyvore_df.to_csv("style.csv")