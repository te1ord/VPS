import sys
import os
from pathlib import Path
import torch
from tqdm import tqdm
import json
from PIL import Image
import numpy as np
import argparse

# Add src to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.preprocessing.image_processor import ImageProcessor
from src.feature_extraction.clip_extractor import CLIPExtractor
from src.search.faiss_index import FAISSSearcher, IndexItem
from src.preprocessing.object_detector import ObjectDetector
from src.utils.image_utils import get_image_files

def build_and_save_index(
    data_dir: str = "data/images",
    output_dir: str = "data/index",
    batch_size: int = 32,
    use_detection: bool = True,
    debug_limit: int = None
):
    """
    Build FAISS index and save it to disk.
    
    Args:
        data_dir: Directory containing product images
        output_dir: Directory to save the index
        batch_size: Batch size for processing (used in non-detection mode)
        use_detection: Whether to use object detection
        debug_limit: If set, process only this many images (for debugging)
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    preprocessor = ImageProcessor()
    feature_extractor = CLIPExtractor()
    searcher = FAISSSearcher()
    
    if use_detection:
        detector = ObjectDetector()
    
    # Get all product images
    print("Getting image paths...")
    image_paths = get_image_files(data_dir)
    
    if not image_paths:
        raise ValueError(f"No images found in {data_dir}")
    
    if debug_limit:
        image_paths = image_paths[:debug_limit]
        print(f"Debug mode: processing only {debug_limit} images")
    
    print(f"Found {len(image_paths)} images")
    
    # Statistics for logging
    total_objects = 0
    total_images = len(image_paths)
    
    if use_detection:
        # Process images with object detection
        print("Processing images with object detection...")
        i = 0
        for image_path in tqdm(image_paths):
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            features_list = []
            items_list = []
            
            # First, process and add the full image
            full_image_tensor = preprocessor.preprocess(image).unsqueeze(0)
            full_image_features = feature_extractor.extract_features(full_image_tensor)
            
            # Add full image to index
            w, h = image.size
            full_image_item = IndexItem(
                original_image_path=str(image_path),
                bbox=(float(0), float(0), float(w), float(h)),
                class_name="full_image",
                confidence=1.0,
                is_full_image=True
            )
            
            features_list.append(full_image_features)
            items_list.append(full_image_item)
            
            # Then detect and process objects
            detections = detector.detect_objects(image)
            
            # Process each detection
            for detection in detections:
                # Skip if it's a fallback whole image detection
                if detection.class_name == "whole_image":
                    continue
                
                # Process cropped image
                image_tensor = preprocessor.preprocess(detection.cropped_image).unsqueeze(0)
                
                # Extract features
                features = feature_extractor.extract_features(image_tensor)
                
                # Convert bbox to Python floats
                bbox = tuple(float(x) for x in detection.bbox)
                
                # Create index item
                item = IndexItem(
                    original_image_path=str(image_path),
                    bbox=bbox,
                    class_name=str(detection.class_name),
                    confidence=float(detection.confidence),
                    is_full_image=False
                )
                
                features_list.append(features)
                items_list.append(item)
                total_objects += 1
            
            # Concatenate features and add to index
            if features_list:
                all_features = torch.cat(features_list, dim=0)
                searcher.add_items(all_features, items_list)

            i += 1
            # if i > 10:
            #     break
    else:
        # Process images in batches without detection
        print("Processing images in batches (without detection)...")
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i:i + batch_size]
            # Process images
            images = torch.cat([preprocessor.process_image(path) for path in batch_paths])
            # Extract features
            features = feature_extractor.extract_features(images)
            # Add to index
            searcher.add_images(features, batch_paths)
    
    # Save index
    index_path = Path(output_dir) / "product_index.faiss"
    print(f"Saving index to {index_path}")
    searcher.save_index(str(index_path))
    
    # Print statistics
    print(f"\nIndexing Statistics:")
    print(f"Total images processed: {total_images}")
    if use_detection:
        print(f"Total objects detected: {total_objects}")
        print(f"Average objects per image: {total_objects/total_images:.2f}")
    print("Done!")

def main():
    parser = argparse.ArgumentParser(description="Build FAISS index for visual product search")
    parser.add_argument("--data-dir", default="data/images", help="Directory containing product images")
    parser.add_argument("--output-dir", default="data/index", help="Directory to save the index")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--no-detection", action="store_true", help="Disable object detection")
    parser.add_argument("--debug-limit", type=int, help="Process only N images (for debugging)")
    
    args = parser.parse_args()
    
    build_and_save_index(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        use_detection=not args.no_detection,
        debug_limit=args.debug_limit
    )

if __name__ == "__main__":
    main() 