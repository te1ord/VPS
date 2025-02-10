from dataclasses import dataclass
from pathlib import Path
from typing import List, Union
import os
from PIL import Image
import torch
from typing import List, Tuple, Optional, Set
from torchvision.transforms import functional as F

@dataclass
class Detection:
    """Class to store detection results."""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_name: str
    cropped_image: Image.Image
    

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

def crop_image(image: Image.Image, bbox: tuple, preserve_aspect_ratio: bool = False) -> Image.Image:
    """
    Crop image to bounding box.
    
    Args:
        image: Input image
        bbox: Bounding box coordinates (x1, y1, x2, y2)
        preserve_aspect_ratio: If True, pad the crop to square for feature extraction
    
    Returns:
        Cropped image (padded to square if preserve_aspect_ratio is True)
    """
    x1, y1, x2, y2 = map(int, bbox)
    cropped = image.crop((x1, y1, x2, y2))
    
    if preserve_aspect_ratio:
        # Pad to square while preserving aspect ratio
        w, h = cropped.size
        if w > h:
            new_img = Image.new('RGB', (w, w), (0, 0, 0))
            new_img.paste(cropped, (0, (w - h) // 2))
        else:
            new_img = Image.new('RGB', (h, h), (0, 0, 0))
            new_img.paste(cropped, ((h - w) // 2, 0))
        return new_img
    
    return cropped

def compute_iou(box1: Tuple[float, float, float, float], 
                box2: Tuple[float, float, float, float]) -> float:
    """Compute IoU between two boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Compute intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Compute areas
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Compute IoU
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0

def non_max_suppression(detections: List[Detection], iou_threshold: float = 0.5, cross_class_suppression: bool = False) -> List[Detection]:
    """Apply non-max suppression to remove overlapping detections.
    
    Args:
        cross_class_suppression: If True, suppress overlapping detections
            regardless of class. Default: False (keep different classes)
    """
    if not detections:
        return []
    
    # Sort by confidence
    detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
    
    # Initialize list of kept detections
    kept_detections = []
    
    while detections:
        # Take the detection with highest confidence
        current = detections.pop(0)
        kept_detections.append(current)
        
        # Filter out detections that overlap too much with current
        filtered_detections = []
        for detection in detections:
            # Only skip different classes when cross_class_suppression is disabled
            if not cross_class_suppression and detection.class_name != current.class_name:
                filtered_detections.append(detection)
                continue
                
            iou = compute_iou(current.bbox, detection.bbox)
            if iou <= iou_threshold:
                filtered_detections.append(detection)
        
        detections = filtered_detections
    
    return kept_detections