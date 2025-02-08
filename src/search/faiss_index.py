import faiss
import numpy as np
import torch
import yaml
from pathlib import Path
from typing import List, Tuple

class FAISSSearcher:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.config = config['search']
        self.index = self._create_index()
        self.image_paths = []
    
    def _create_index(self) -> faiss.Index:
        """Create a FAISS index based on configuration."""
        dimension = self.config['dimension']
        
        if self.config['index_type'] == 'IndexFlatIP':
            return faiss.IndexFlatIP(dimension)
        else:
            raise ValueError(f"Unsupported index type: {self.config['index_type']}")
    
    def add_images(self, features: torch.Tensor, image_paths: List[str]):
        """
        Add image features and their paths to the index.
        
        Args:
            features: Normalized feature tensor of shape (N, dimension)
            image_paths: List of corresponding image paths
        """
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        
        if features.shape[0] != len(image_paths):
            raise ValueError("Number of features and image paths must match")
        
        self.index.add(features)
        self.image_paths.extend(image_paths)
    
    def search(self, query_features: torch.Tensor, k: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar images.
        
        Args:
            query_features: Query feature tensor of shape (1, dimension)
            k: Number of results to return (default: from config)
        
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
        
        distances, indices = self.index.search(query_features, k)
        return distances, indices
    
    def get_image_paths(self, indices: np.ndarray) -> List[str]:
        """Get image paths for given indices."""
        return [self.image_paths[idx] for idx in indices[0]]
    
    def save_index(self, path: str):
        """Save the FAISS index to disk."""
        faiss.write_index(self.index, path)
    
    def load_index(self, path: str):
        """Load the FAISS index from disk."""
        self.index = faiss.read_index(path) 