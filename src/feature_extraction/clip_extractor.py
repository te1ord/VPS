import torch
import clip
import yaml
from pathlib import Path
from typing import Union, List

class CLIPExtractor:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.device = torch.device(config['model']['device'])
        self.model_name = config['model']['clip_model']
        
        # Load the model
        self.model, _ = clip.load(self.model_name, device=self.device)
        self.model.eval()  # Set to evaluation mode
    
    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features from preprocessed images using CLIP.
        
        Args:
            images: Preprocessed image tensor of shape (B, C, H, W)
        
        Returns:
            features: Normalized feature tensor of shape (B, D) where D is embedding dimension
        """
        images = images.to(self.device)
        features = self.model.encode_image(images)
        
        # Normalize features
        features = features / features.norm(dim=-1, keepdim=True)
        
        return features.cpu().float()
    
    def extract_features_from_paths(self, image_paths: Union[str, List[str]], preprocessor) -> torch.Tensor:
        """
        Extract features from image paths using provided preprocessor.
        
        Args:
            image_paths: Single image path or list of image paths
            preprocessor: ImageProcessor instance for preprocessing
        
        Returns:
            features: Normalized feature tensor
        """
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        
        # Preprocess images
        images = preprocessor.process_batch(image_paths)
        
        # Extract features
        return self.extract_features(images) 