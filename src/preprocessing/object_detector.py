from ultralytics import YOLO
import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Set
import yaml
import logging
from src.utils.image_utils import crop_image, Detection, non_max_suppression
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ObjectDetector:
    def __init__(self, config_path: str = "src/config/config.yaml", confidence_threshold: float = None):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.confidence_threshold = confidence_threshold if confidence_threshold is not None else config['detection']['confidence_threshold']
        self.iou_threshold = config['detection'].get('nms_iou_threshold', 0.5)
        self.model = YOLO(config['detection']['model_path'])
        self.device = torch.device(config['model']['device'])
        self.model.to(self.device)
        
        # Set up class filtering
        self.include_classes = set(config['detection'].get('include_classes', []))
        self.exclude_classes = set(config['detection'].get('exclude_classes', []))
        
        # Log configuration
        logger.info(f"Loaded model: {config['detection']['model_path']}")
        logger.info(f"Confidence threshold: {self.confidence_threshold}")
        logger.info(f"NMS IoU threshold: {self.iou_threshold}")
        logger.info(f"Excluded classes: {sorted(self.exclude_classes)}")
        if self.include_classes:
            logger.info(f"Included classes: {sorted(self.include_classes)}")
    
    def _should_include_class(self, class_name: str) -> bool:
        """
        Determine if a class should be included based on filtering rules.
        
        Args:
            class_name: Name of the class to check
            
        Returns:
            bool: True if class should be included, False otherwise
        """
        # Always exclude classes in exclude list
        if class_name in self.exclude_classes:
            logger.debug(f"Excluding class {class_name} (in exclude list)")
            return False
        
        # If include list is empty, include all classes except excluded ones
        if not self.include_classes:
            return True
        
        # If include list is not empty, only include classes from that list
        return class_name in self.include_classes
    
    def detect_objects(self, image: Image.Image) -> List[Detection]:
        """
        Detect objects in image and return their cropped regions.
        
        Args:
            image: PIL Image to process
        
        Returns:
            List of Detection objects containing bboxes, confidences, and cropped images
        """
        # Convert PIL to format expected by YOLO
        results = self.model(image, conf=self.confidence_threshold)
        
        detections = []
        excluded_count = 0
        for r in results[0].boxes:
            class_name = self.model.names[int(r.cls[0])]
            conf = float(r.conf[0])
            
            # Skip if class should not be included
            if not self._should_include_class(class_name):
                excluded_count += 1
                logger.debug(f"Excluding detection: {class_name} (conf: {conf:.2f})")
                continue
            
            bbox = r.xyxy[0].cpu().numpy()
            
            # Crop and pad the detected region
            cropped_img = crop_image(image, bbox)
            
            detections.append(Detection(
                bbox=tuple(bbox),
                confidence=conf,
                class_name=class_name,
                cropped_image=cropped_img
            ))
        
        logger.debug(f"Found {len(detections)} valid detections, excluded {excluded_count} detections")
        
        # Apply NMS to remove overlapping detections
        original_count = len(detections)
        detections = non_max_suppression(detections, self.iou_threshold, cross_class_suppression=False)
        logger.debug(f"After NMS: kept {len(detections)} detections, removed {original_count - len(detections)}")
        
        # If no detections (or all filtered out), use whole image as fallback
        if not detections:
            logger.debug("No valid detections, using whole image as fallback")
            w, h = image.size
            detections.append(Detection(
                bbox=(0, 0, w, h),
                confidence=1.0,
                class_name="whole_image",
                cropped_image=crop_image(image, (0, 0, w, h))
            ))
        
        return detections 