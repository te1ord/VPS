import faiss
import numpy as np
import torch
import yaml
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass

@dataclass
class IndexItem:
    """Class to store index item metadata."""
    original_image_path: str  # Path to the original image
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 (None for full image)
    class_name: str  # Detected class name or "full_image"
    confidence: float  # Detection confidence or 1.0 for full image
    is_full_image: bool  # Whether this is a full image embedding

class FAISSSearcher:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.config = config['search']
        self.index = self._create_index()
        self.metadata: List[IndexItem] = []
        self.image_paths: List[str] = []  # Keep for backward compatibility
    
    def _create_index(self) -> faiss.Index:
        """Create a FAISS index based on configuration."""
        dimension = self.config['dimension']
        
        if self.config['index_type'] == 'IndexFlatIP':
            return faiss.IndexFlatIP(dimension)
        else:
            raise ValueError(f"Unsupported index type: {self.config['index_type']}")
    
    def add_items(self, features: torch.Tensor, items: List[IndexItem]):
        """
        Add features and their metadata to the index.
        
        Args:
            features: Normalized feature tensor of shape (N, dimension)
            items: List of IndexItem objects with metadata
        """
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        
        if features.shape[0] != len(items):
            raise ValueError("Number of features and items must match")
        
        self.index.add(features)
        self.metadata.extend(items)
        # Update image_paths for backward compatibility
        self.image_paths.extend([item.original_image_path for item in items])
    
    def add_images(self, features: torch.Tensor, image_paths: List[str]):
        """
        Legacy method for adding image features without detection metadata.
        Converts to new format treating each image as a full-image item.
        
        Args:
            features: Normalized feature tensor of shape (N, dimension)
            image_paths: List of corresponding image paths
        """
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        
        if features.shape[0] != len(image_paths):
            raise ValueError("Number of features and image paths must match")
        
        # Convert to IndexItems
        items = [
            IndexItem(
                original_image_path=path,
                bbox=(0, 0, 1, 1),  # Normalized coordinates for full image
                class_name="full_image",
                confidence=1.0,
                is_full_image=True
            )
            for path in image_paths
        ]
        
        self.add_items(features, items)
    
    def search(self, query_features: torch.Tensor, k: int = None, text_features: Optional[torch.Tensor] = None, 
             visual_weight: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar items using visual and/or text features.
        
        Args:
            query_features: Visual query feature tensor of shape (1, dimension)
            k: Number of results to return (default: from config)
            text_features: Optional text query feature tensor of shape (1, dimension)
            visual_weight: Weight for visual features (1 - visual_weight will be used for text)
        
        Returns:
            distances: Array of similarity scores
            indices: Array of indices in the index
        """
        if k is None:
            k = self.config.get('max_results', 5)
        
        if isinstance(query_features, torch.Tensor):
            query_features = query_features.cpu().numpy()
        
        # Ensure query features are 2D
        if len(query_features.shape) == 1:
            query_features = query_features.reshape(1, -1)
            
        # If text features are provided, combine them with visual features
        if text_features is not None:
            if isinstance(text_features, torch.Tensor):
                text_features = text_features.cpu().numpy()
            if len(text_features.shape) == 1:
                text_features = text_features.reshape(1, -1)
                
            # Combine features with weighting
            combined_features = visual_weight * query_features + (1 - visual_weight) * text_features
            # Normalize combined features
            combined_features = combined_features / np.linalg.norm(combined_features, axis=1, keepdims=True)
            query_features = combined_features
        
        distances, indices = self.index.search(query_features, k)
        return distances, indices
    
    def get_metadata(self, indices: np.ndarray) -> List[IndexItem]:
        """Get metadata for given indices."""
        return [self.metadata[idx] for idx in indices[0]]
    
    def get_image_paths(self, indices: np.ndarray) -> List[str]:
        """Legacy method to get image paths for given indices."""
        if self.metadata:
            return [item.original_image_path for item in self.get_metadata(indices)]
        return [self.image_paths[idx] for idx in indices[0]]
    
    def _convert_to_json_serializable(self, item: IndexItem) -> dict:
        """Convert IndexItem to JSON serializable dictionary."""
        return {
            'original_image_path': str(item.original_image_path),
            'bbox': tuple(float(x) for x in item.bbox),
            'class_name': str(item.class_name),
            'confidence': float(item.confidence),
            'is_full_image': bool(item.is_full_image)
        }
    
    def save_index(self, path: str):
        """Save the FAISS index and metadata to disk."""
        # Save FAISS index
        faiss.write_index(self.index, path)
        
        # Save metadata if available
        if self.metadata:
            metadata_path = Path(path).with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump([self._convert_to_json_serializable(item) for item in self.metadata], f)
        # Save legacy image paths if no metadata
        elif self.image_paths:
            paths_path = Path(path).parent / "image_paths.json"
            with open(paths_path, 'w') as f:
                json.dump(self.image_paths, f)
    
    def load_index(self, path: str):
        """Load the FAISS index and metadata from disk."""
        # Load FAISS index
        self.index = faiss.read_index(path)
        
        # Try to load metadata first
        metadata_path = Path(path).with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                data = json.load(f)
                self.metadata = [IndexItem(**item) for item in data]
                # Update image_paths for backward compatibility
                self.image_paths = [item.original_image_path for item in self.metadata]
        else:
            # Try legacy image paths
            paths_path = Path(path).parent / "image_paths.json"
            if paths_path.exists():
                with open(paths_path, 'r') as f:
                    self.image_paths = json.load(f)
                    # Create basic metadata for backward compatibility
                    self.metadata = [
                        IndexItem(
                            original_image_path=path,
                            bbox=(0, 0, 1, 1),
                            class_name="full_image",
                            confidence=1.0,
                            is_full_image=True
                        )
                        for path in self.image_paths
                    ] 