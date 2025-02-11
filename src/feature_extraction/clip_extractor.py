import torch
import clip
import yaml
from pathlib import Path
from typing import Union, List, Tuple, Dict

class CLIPExtractor:
    def __init__(self, config_path: str = "src/config/config.yaml"):
        # Load main config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.device = config['model']['device']
        self.model_name = config['model']['clip_model']
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)
        
        # Load product attributes and pre-compute text embeddings
        categories_path = Path(config_path).parent / "tagging_categories.yaml"
        try:
            with open(categories_path, 'r') as f:
                categories_config = yaml.safe_load(f)
            self.product_attributes = categories_config['product_attributes']
            print(f"Loaded {len(self.product_attributes)} product categories for tagging")
            
            # Pre-compute embeddings for individual tags
            self.tag_embeddings = self._precompute_tag_embeddings()
            print("Pre-computed tag embeddings")
            
        except Exception as e:
            print(f"Warning: Could not load categories from {categories_path}. Using default categories.")
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
            self.tag_embeddings = self._precompute_tag_embeddings()
    
    def _precompute_tag_embeddings(self) -> Dict[str, torch.Tensor]:
        """Pre-compute text embeddings for individual tags."""
        tag_embeddings = {}
        with torch.no_grad():
            # Process tags in batches to avoid memory issues with large category lists
            batch_size = 32
            for i in range(0, len(self.product_attributes), batch_size):
                batch_tags = self.product_attributes[i:i + batch_size]
                text_tokens = clip.tokenize([f"a photo of {attr}" for attr in batch_tags]).to(self.device)
                text_features = self.model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Store individual tag embeddings
                for idx, tag in enumerate(batch_tags):
                    tag_embeddings[tag] = text_features[idx].cpu()
        
        return tag_embeddings
    
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
        Get top-k tags for an image using zero-shot classification with pre-computed embeddings.
        Returns only tags with confidence scores above threshold.
        """
        with torch.no_grad():
            # Get image features
            image_features = self.model.encode_image(image_tensor.to(self.device))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Stack pre-computed tag embeddings
            tag_embeddings = torch.stack(list(self.tag_embeddings.values())).to(self.device)
            
            # Calculate similarity scores
            similarity = (100.0 * image_features @ tag_embeddings.T).softmax(dim=-1)
            
            # Get top-k tags and scores
            values, indices = similarity[0].topk(top_k)
            
            # Filter out scores below threshold
            non_zero_mask = values >= threshold
            values = values[non_zero_mask]
            indices = indices[non_zero_mask]
            
            # Convert to lists
            top_tags = [self.product_attributes[idx] for idx in indices]
            scores = values.cpu().numpy().tolist()
            
            return top_tags, scores
    
    def generate_text_query(self, image_tensor: torch.Tensor) -> Tuple[str, torch.Tensor]:
        """
        Generate a descriptive text query from image tags and compute its embedding.
        
        Returns:
            query: Generated text query string
            query_embedding: CLIP embedding for the complete query
        """
        tags, scores = self.get_image_tags(image_tensor, top_k=5)
        
        # Generate query text
        if not tags:
            query = "a product"
        elif len(tags) == 1:
            query = f"a {tags[0]} product"
        else:
            query = f"a {', '.join(tags[:-1])} and {tags[-1]} product"
        
        # Get embedding for the complete query
        with torch.no_grad():
            text_tokens = clip.tokenize([query]).to(self.device)
            query_embedding = self.model.encode_text(text_tokens)
            query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
        
        return query, query_embedding.cpu()
    
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