import sys
import os
from pathlib import Path
import torch
from tqdm import tqdm
import json

# Add src to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.preprocessing.image_processor import ImageProcessor
from src.feature_extraction.clip_extractor import CLIPExtractor
from src.search.faiss_index import FAISSSearcher
from src.utils.image_utils import get_image_files

def build_and_save_index(data_dir="data/images", output_dir="data/index", batch_size=32):
    """Build FAISS index and save it to disk."""
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    preprocessor = ImageProcessor()
    feature_extractor = CLIPExtractor()
    searcher = FAISSSearcher()
    
    # Get all product images
    print("Getting image paths...")
    image_paths = get_image_files(data_dir)
    
    if not image_paths:
        raise ValueError(f"No images found in {data_dir}")
    
    print(f"Found {len(image_paths)} images")
    
    # Process images in batches
    print("Processing images and building index...")
    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i:i + batch_size]
        # Process images
        images = torch.cat([preprocessor.process_image(path) for path in batch_paths])
        # Extract features
        features = feature_extractor.extract_features(images)
        # Add to index
        searcher.add_images(features, batch_paths)
    
    # Save index and paths
    index_path = Path(output_dir) / "product_index.faiss"
    paths_path = Path(output_dir) / "image_paths.json"
    
    print(f"Saving index to {index_path}")
    searcher.save_index(str(index_path))
    
    print(f"Saving image paths to {paths_path}")
    with open(paths_path, 'w') as f:
        json.dump(searcher.image_paths, f)
    
    print("Done!")

if __name__ == "__main__":
    build_and_save_index() 