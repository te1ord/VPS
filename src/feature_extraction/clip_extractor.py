import torch
import clip
import yaml
from pathlib import Path
from typing import Union, List, Tuple

class CLIPExtractor:
    def __init__(self, config_path: str = "config/config.yaml"):
        # Load main config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.device = config['model']['device']
        self.model_name = config['model']['clip_model']
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)
        
        # Load product attributes from categories file
        categories_path = Path(config_path).parent / "tagging_categories.yaml"
        try:
            with open(categories_path, 'r') as f:
                categories_config = yaml.safe_load(f)
            self.product_attributes = categories_config['product_attributes']
            print(f"Loaded {len(self.product_attributes)} product categories for tagging")
        except Exception as e:
            print(f"Warning: Could not load categories from {categories_path}. Using default categories.")
            # Fallback to default categories if file loading fails
            self.product_attributes = [
                "dress", "t-shirt", "jeans", "shoes", "sneakers", "boots",
                "jacket", "coat", "sweater", "skirt", "pants", "shorts",
                "formal", "casual", "sporty", "elegant", "vintage", "modern",
                "black", "white", "red", "blue", "green", "yellow", "pink", "purple",
                "leather", "cotton", "denim", "silk", "wool", "synthetic",
                "floral", "striped", "plain", "patterned", "checkered",
                "summer", "winter", "spring", "autumn",
                "men's", "women's", "unisex",
                "accessories", "bag", "watch", "jewelry", "sunglasses",
                "athletic", "business", "party", "everyday"
            ]
    
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from images using CLIP."""
        with torch.no_grad():
            images = images.to(self.device)
            image_features = self.model.encode_image(images)
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu()
    
    def extract_text_features(self, text: str) -> torch.Tensor:
        """Extract features from text using CLIP."""
        with torch.no_grad():
            text = clip.tokenize([text]).to(self.device)
            text_features = self.model.encode_text(text)
            # Normalize features
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features.cpu()
    
    def get_image_tags(self, image_tensor: torch.Tensor, top_k: int = 5, threshold: float = 0.1) -> Tuple[List[str], List[float]]:
        """
        Get top-k tags for an image using zero-shot classification.
        Returns only tags with non-zero confidence scores.
        
        Args:
            image_tensor: Preprocessed image tensor
            top_k: Maximum number of top tags to return
            
        Returns:
            tags: List of tags with non-zero confidence (may be less than top_k)
            scores: List of confidence scores for each tag
        """
        with torch.no_grad():
            # Prepare text tokens for all attributes
            text_tokens = clip.tokenize([f"a photo of {attr}" for attr in self.product_attributes]).to(self.device)
            
            # Get image and text features
            image_features = self.model.encode_image(image_tensor.to(self.device))
            text_features = self.model.encode_text(text_tokens)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity scores
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            # Get top-k tags and scores
            values, indices = similarity[0].topk(top_k)
            
            # Filter out zero confidence scores
            non_zero_mask = values >= threshold
            values = values[non_zero_mask]
            indices = indices[non_zero_mask]
            
            # Convert to lists
            top_tags = [self.product_attributes[idx] for idx in indices]
            scores = values.cpu().numpy().tolist()
            
            return top_tags, scores
    
    def generate_text_query(self, image_tensor: torch.Tensor) -> str:
        """
        Generate a descriptive text query from image tags.
        
        Args:
            image_tensor: Preprocessed image tensor
            
        Returns:
            query: Generated text query
        """
        tags, scores = self.get_image_tags(image_tensor, top_k=5)
        
        # Handle cases with different numbers of tags
        if not tags:
            return "a product"  # fallback if no confident tags
        elif len(tags) == 1:
            return f"a {tags[0]} product"
        else:
            return f"a {', '.join(tags[:-1])} and {tags[-1]} product"
    
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