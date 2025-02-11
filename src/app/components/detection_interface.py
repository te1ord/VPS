import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Optional, Tuple
from src.preprocessing.object_detector import ObjectDetector
from src.utils.image_utils import crop_image, non_max_suppression
import yaml

def display_image_with_detections(image: Image.Image, detector: ObjectDetector) -> Optional[Tuple[float, float, float, float]]:
    """
    Display image with detected regions and allow selection.
    Returns the selected region's bbox or None if whole image is selected.
    """
    # Load config
    with open("src/config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Detect objects
    detections = detector.detect_objects(image)
    
    # Filter out human-related classes and small regions
    FILTERED_CLASSES = {'man', 'woman', 'person', 'boy', 'girl'}
    w, h = image.size
    image_area = w * h
    valid_detections = []
    
    for det in detections:
        if det.class_name.lower() in FILTERED_CLASSES:
            continue
        
        # Calculate region area percentage
        bbox = det.bbox
        region_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        area_ratio = region_area / image_area
        
        if area_ratio >= config['app']['min_region_size']:
            valid_detections.append(det)
    
    # Apply NMS across all classes
    valid_detections = non_max_suppression(
        valid_detections, 
        iou_threshold=config['app']['nms_iou_threshold_ui'],
        cross_class_suppression=True
    )
    
    # Create figure and axis with controlled size
    # Resize image for display if too large
    max_size = 600
    aspect_ratio = w / h
    if w > max_size or h > max_size:
        if aspect_ratio > 1:
            new_size = (max_size, int(max_size / aspect_ratio))
        else:
            new_size = (int(max_size * aspect_ratio), max_size)
        display_image = image.resize(new_size, Image.Resampling.LANCZOS)
    else:
        display_image = image
        new_size = (w, h)
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    if not valid_detections:
        # If no valid detections, show original image and message
        with col1:
            st.image(display_image, use_container_width=True)
        with col2:
            st.warning("No confident regions detected. Using the whole image for search.")
        return None
    
    # Create figure for displaying detections
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(display_image)
    
    # Scale factor for bounding boxes
    scale_x = new_size[0] / w
    scale_y = new_size[1] / h
    
    # Draw rectangles with different colors
    colors = plt.cm.rainbow(np.linspace(0, 1, len(valid_detections)))
    for det, color in zip(valid_detections, colors):
        x1, y1, x2, y2 = det.bbox
        # Scale coordinates
        x1, x2 = x1 * scale_x, x2 * scale_x
        y1, y2 = y1 * scale_y, y2 * scale_y
        
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label above the box
        plt.text(x1, y1-5, det.class_name, color=color, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Remove axes and set tight layout
    ax.axis('off')
    plt.tight_layout()
    
    # Create options list
    options = ["Whole Image"] + [f"{det.class_name} (Confidence: {det.confidence:.2f})" for det in valid_detections]
    
    with col1:
        st.pyplot(fig)
    
    with col2:
        st.markdown("### 🎯 Select Region")
        selected_idx = st.radio(
            "Choose a region to search for:",
            options,
            help="Select a specific region or use the whole image"
        )
        
        if selected_idx != "Whole Image":
            # Find the selected detection
            selected_detection = valid_detections[options.index(selected_idx) - 1]
            # Show the selected region in original scale
            region = crop_image(image, selected_detection.bbox, preserve_aspect_ratio=False)
            st.markdown("### Selected Region")
            st.image(region, use_container_width=True)
            return selected_detection.bbox
    
    return None 