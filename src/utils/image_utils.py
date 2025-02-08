from pathlib import Path
from typing import List, Union
import os
from PIL import Image
import torch
from torchvision.transforms import functional as F

def get_image_files(directory: Union[str, Path]) -> List[str]:
    """
    Get all image files from a directory recursively.
    
    Args:
        directory: Path to the directory
    
    Returns:
        List of image file paths
    """
    if isinstance(directory, str):
        directory = Path(directory)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    image_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(str(Path(root) / file))
    
    return sorted(image_files)
