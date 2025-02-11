import torch
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Optional
from PIL import Image
import time

from src.preprocessing.image_processor import ImageProcessor
from src.feature_extraction.clip_extractor import CLIPExtractor
from src.search.faiss_index import FAISSSearcher, IndexItem
from src.preprocessing.object_detector import ObjectDetector

class SearchEngine:
    def __init__(self, preprocessor: ImageProcessor, feature_extractor: CLIPExtractor, 
                 searcher: FAISSSearcher, detector: ObjectDetector, min_region_size: float = 0.1):
        self.preprocessor = preprocessor
        self.feature_extractor = feature_extractor
        self.searcher = searcher
        self.detector = detector
        self.min_region_size = min_region_size
    
    def process_search_results(self, distances: np.ndarray, indices: np.ndarray) -> List[Tuple[str, List[Tuple[IndexItem, float]]]]:
        """
        Process search results: group by original image, filter regions, and sort by best score.
        
        Args:
            distances: Similarity scores from FAISS search
            indices: Indices from FAISS search
        
        Returns:
            List of (image_path, [(item, score)]) tuples, sorted by best score
        """
        # Define classes to filter out
        FILTERED_CLASSES = {'man', 'woman', 'person', 'boy', 'girl'}
        
        # Get metadata for all results
        metadata_items = self.searcher.get_metadata(indices)
        
        # Group results by original image
        image_results = defaultdict(list)
        for item, score in zip(metadata_items, distances[0]):
            image_results[item.original_image_path].append((item, score))
        
        # Process each image's results
        final_results = []
        for img_path, items in image_results.items():
            # Load image once for area calculations
            image = Image.open(img_path).convert('RGB')
            w, h = image.size
            image_area = w * h
            
            # Filter and process items for this image
            valid_items = []
            for item, score in items:
                if item.is_full_image:
                    # Always keep full images
                    valid_items.append((item, score))
                else:
                    # Skip human-related detections
                    if item.class_name.lower() in FILTERED_CLASSES:
                        continue
                        
                    # Calculate region area percentage
                    bbox = item.bbox
                    region_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    area_ratio = region_area / image_area
                    
                    # Only keep regions larger than minimum size
                    if area_ratio >= self.min_region_size:
                        valid_items.append((item, score))
            
            if valid_items:  # Only add images that have valid results
                final_results.append((img_path, valid_items))
        
        # Sort by best score of any valid item for each image
        final_results.sort(key=lambda x: max(score for _, score in x[1]), reverse=True)
        return final_results
    
    def search(self, image: Optional[Image.Image] = None, text_query: Optional[str] = None,
              selected_region: Optional[tuple] = None, use_auto_tagging: bool = False,
              visual_weight: float = 0.5, merge_strategy: str = "embedding", k: int = 50) -> Tuple[List[Tuple[str, List[Tuple[IndexItem, float]]]], float]:
        """
        Perform search using image and/or text query.
        Returns processed results and search time.
        """
        start_time = time.time()
        
        # Extract features based on inputs
        image_features = None
        text_features = None
        
        if image is not None:
            # Process image region if selected
            if selected_region:
                region = image.crop(selected_region)
                image_tensor = self.preprocessor.preprocess(region).unsqueeze(0)
            else:
                image_tensor = self.preprocessor.preprocess(image).unsqueeze(0)
            
            image_features = self.feature_extractor.extract_features(image_tensor)
            
            # Handle auto-tagging
            if use_auto_tagging and not text_query:
                auto_text_query, text_features = self.feature_extractor.generate_text_query(image_tensor)
                return_tags = True
            else:
                auto_text_query = None
                return_tags = False
        
        if text_query:
            text_features = self.feature_extractor.extract_text_features(text_query)
        
        # Perform search
        if image_features is not None:
            if text_features is not None:
                # Hybrid search
                distances, indices = self.searcher.search(
                    image_features,
                    k=k,
                    text_features=text_features,
                    visual_weight=visual_weight,
                    merge_strategy=merge_strategy
                )
            else:
                # Pure visual search
                distances, indices = self.searcher.search(image_features, k=k)
        else:
            # Pure text search
            distances, indices = self.searcher.search(
                text_features,
                k=k,
                text_features=None
            )
        
        # Process results
        processed_results = self.process_search_results(distances, indices)
        search_time = time.time() - start_time
        
        return processed_results, search_time 