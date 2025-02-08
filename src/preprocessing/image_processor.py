import torch
from torchvision import transforms
from PIL import Image
import yaml
from pathlib import Path

class ImageProcessor:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.preprocess = self._build_transforms(config['preprocessing'])
    
    def _build_transforms(self, config):
        """Build the transformation pipeline."""
        return transforms.Compose([
            transforms.Resize(config['image_size']),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config['mean'],
                std=config['std']
            ) if config['normalize'] else transforms.Lambda(lambda x: x)
        ])
    
    def process_image(self, image_path: str) -> torch.Tensor:
        """Process a single image."""
        if isinstance(image_path, str):
            image_path = Path(image_path)
        
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.preprocess(image)
            return image_tensor.unsqueeze(0)  
        except Exception as e:
            raise ValueError(f"Error processing image {image_path}: {str(e)}")
    
    def process_batch(self, image_paths: list) -> torch.Tensor:
        """Process a batch of images."""
        tensors = []
        for img_path in image_paths:
            tensors.append(self.process_image(img_path))
        return torch.cat(tensors, dim=0) 