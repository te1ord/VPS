import streamlit as st
import torch
import yaml
from pathlib import Path
from PIL import Image
import sys
import os
import json
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Optional

# Add src to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.preprocessing.image_processor import ImageProcessor
from src.feature_extraction.clip_extractor import CLIPExtractor
from src.search.faiss_index import FAISSSearcher, IndexItem
from src.preprocessing.object_detector import ObjectDetector
from src.utils.image_utils import compute_iou, crop_image, non_max_suppression

# Load configuration
with open("config/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

@st.cache_resource
def load_models():
    """Load and cache models and pre-computed index."""
    preprocessor = ImageProcessor()
    feature_extractor = CLIPExtractor()
    detector = ObjectDetector()
    searcher = FAISSSearcher()
    
    # Load pre-computed index
    index_dir = Path("data/index_w_det")
    index_path = index_dir / "product_index.faiss"
    
    if not index_path.exists():
        st.error("Index not found. Please run build_index.py first!")
        st.stop()
    
    # Load index
    searcher.load_index(str(index_path))
    
    return preprocessor, feature_extractor, detector, searcher

def load_and_process_image(image_data):
    """Load and process uploaded image."""
    if image_data is not None:
        return Image.open(image_data).convert('RGB')
    return None


def process_search_results(distances: np.ndarray, indices: np.ndarray, searcher: FAISSSearcher) -> List[Tuple[str, List[Tuple[IndexItem, float]]]]:
    """
    Process search results: group by original image, filter regions, and sort by best score.
    
    Args:
        distances: Similarity scores from FAISS search
        indices: Indices from FAISS search
        searcher: FAISSSearcher instance
    
    Returns:
        List of (image_path, [(item, score)]) tuples, sorted by best score
    """
    # Define classes to filter out
    FILTERED_CLASSES = {'man', 'woman', 'person', 'boy', 'girl'}
    
    # Get metadata for all results
    metadata_items = searcher.get_metadata(indices)
    
    # Group results by original image
    image_results = defaultdict(list)
    for item, score in zip(metadata_items, distances[0]):
        image_results[item.original_image_path].append((item, score))
    
    # Process each image's results
    final_results = []
    for img_path, items in image_results.items():
        # Load image once for area calculations
        image = Image.open(img_path).convert('RGB')
        w, h = image.size
        image_area = w * h
        
        # Filter and process items for this image
        valid_items = []
        for item, score in items:
            if item.is_full_image:
                # Always keep full images
                valid_items.append((item, score))
            else:
                # Skip human-related detections
                if item.class_name.lower() in FILTERED_CLASSES:
                    continue
                    
                # Calculate region area percentage
                bbox = item.bbox
                region_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                area_ratio = region_area / image_area
                
                # Only keep regions larger than 50% of the image
                if area_ratio >= config['detection']['min_region_size']:
                    valid_items.append((item, score))
        
        if valid_items:  # Only add images that have valid results
            final_results.append((img_path, valid_items))
    
    # Sort by best score of any valid item for each image
    final_results.sort(key=lambda x: max(score for _, score in x[1]), reverse=True)
    return final_results

def display_results(processed_results: List[Tuple[str, List[Tuple[IndexItem, float]]]], num_results: int):
    """Display search results grouped by original images."""
    # Take only the requested number of results
    results_to_show = processed_results[:num_results]
    
    # Display results in a grid
    cols = st.columns(3)
    
    for idx, (img_path, items) in enumerate(results_to_show):
        # Sort items by score
        items = sorted(items, key=lambda x: x[1], reverse=True)
        best_item = items[0]
        
        with cols[idx % 3]:
            # Load and display original image
            image = Image.open(img_path).convert('RGB')
            st.image(image, caption=f"Match Score: {best_item[1]:.3f}", use_container_width=True)
            
            # Add expander for detected regions if there are any non-full-image detections
            region_items = [(item, score) for item, score in items if not item.is_full_image]
            if region_items:
                with st.expander("View Detected Regions"):
                    for item, score in region_items:
                        st.markdown(f"##### {item.class_name} (Score: {score:.3f})")
                        region = crop_image(image, item.bbox, preserve_aspect_ratio=False)
                        st.image(region, use_container_width=True)

def display_image_with_detections(image: Image.Image, detector: ObjectDetector) -> Optional[tuple]:
    """
    Display image with detected regions and allow selection.
    Returns the selected region's bbox or None if whole image is selected.
    """
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
        
        if area_ratio >= config['detection']['min_region_size']:
            valid_detections.append(det)
    
    # Apply NMS across all classes
    valid_detections = non_max_suppression(valid_detections, iou_threshold=config['detection']['nms_iou_threshold_ui'], cross_class_suppression=True)
    
    if not valid_detections:
        return None
    
    # Create options list
    options = ["Whole Image"] + [f"{det.class_name} (Confidence: {det.confidence:.2f})" for det in valid_detections]
    
    # Create figure and axis with controlled size
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    # Resize image for display if too large
    max_size = 600  # Reduced maximum size
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
    
    # Create figure
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
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
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
            # Show the selected region in original scale (without padding for display)
            region = crop_image(image, selected_detection.bbox, preserve_aspect_ratio=False)
            st.markdown("### Selected Region")
            st.image(region, use_container_width=True)
            return selected_detection.bbox
    
    return None

def main():
    st.title(config['app']['title'])
    st.write(config['app']['description'])
    
    # Load models and pre-computed index
    with st.spinner('Loading models...'):
        preprocessor, feature_extractor, detector, searcher = load_models()
    
    # Create two columns for search inputs
    col1, col2 = st.columns(2)
    
    with col2:
        # Text search input
        st.markdown("### 💬 Text Search")
        text_query = st.text_input("", placeholder="Describe what you're looking for...", label_visibility="collapsed")
    
    with col1:
        # File uploader with custom styling
        st.markdown("### 🖼️ Upload an Image")
        uploaded_file = st.file_uploader("", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")
        
        # Add auto-tagging toggle if image is uploaded and no text query
        use_auto_tagging = False
        if uploaded_file is not None:
            if text_query:
                st.info("Auto-tagging is disabled when text query is provided")
            else:
                use_auto_tagging = st.checkbox("Use automatic image tagging", value=False,
                                             help="Enable to automatically detect and use image tags for hybrid search")
    
    # Show weighting slider for hybrid search
    show_weight_slider = (uploaded_file is not None and 
                         (text_query or (use_auto_tagging and not text_query)))
    
    if show_weight_slider:
        col1, col2 = st.columns(2)
        with col1:
            visual_weight = st.slider(
                "Visual-Text Weight Balance",
                min_value=0.0,
                max_value=1.0,
                value=0.7 if use_auto_tagging else 0.5,
                help="0 = Text only, 1 = Image only"
            )
        with col2:
            merge_strategy = st.selectbox(
                "Merging Strategy",
                options=["embedding", "score"],
                help=("'embedding': Combine features before search\n"
                     "'score': Separate searches with score combination")
            )
    else:
        visual_weight = 0.7 if use_auto_tagging else 0.5  # Default weights
        merge_strategy = "embedding"  # Default strategy
    
    # Add control for number of similar products
    num_results = st.slider("Number of similar products to show", min_value=1, max_value=20, value=6)
    
    # Process search
    if uploaded_file is not None or text_query:
        if uploaded_file is not None:
            # Load and display input image
            image = load_and_process_image(uploaded_file)
            
            # Display input image with controlled size
            st.markdown("### 📤 Input Image")
            
            # Optional region selection
            use_detection = st.checkbox("Detect and select regions of interest", value=False,
                                      help="Enable to detect and select specific regions in the image")
            
            # Create placeholder for image display
            image_container = st.empty()
            
            # Resize image for display if too large
            w, h = image.size
            max_size = 600
            if w > max_size or h > max_size:
                aspect_ratio = w / h
                if aspect_ratio > 1:
                    new_size = (max_size, int(max_size / aspect_ratio))
                else:
                    new_size = (int(max_size * aspect_ratio), max_size)
                display_image = image.resize(new_size, Image.Resampling.LANCZOS)
            else:
                display_image = image
            
            selected_region = None
            if use_detection:
                # Clear the image container before showing detections
                image_container.empty()
                selected_region = display_image_with_detections(image, detector)
            else:
                # Center the image using columns within the container
                with image_container:
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.image(display_image, use_container_width=True)
            
            # Process image and search
            with st.spinner('🔍 Finding similar products...'):
                # Extract features based on selection
                if selected_region:
                    # Crop and process selected region (with padding for feature extraction)
                    region = crop_image(image, selected_region, preserve_aspect_ratio=True)
                    image_tensor = preprocessor.preprocess(region).unsqueeze(0)
                else:
                    # Process full image
                    image_tensor = preprocessor.preprocess(image).unsqueeze(0)
                
                image_features = feature_extractor.extract_features(image_tensor)
                
                # Display search mode
                search_mode = (
                    "🔄 Hybrid Search (Image + Text)" if text_query or use_auto_tagging
                    else "👁️ Visual Search"
                )
                st.markdown(f"### {search_mode}")
                
                if text_query or use_auto_tagging:
                    st.markdown(f"*Using {visual_weight:.1%} visual and {(1-visual_weight):.1%} textual features*")
                
                # Handle text features (either from user query or auto-tagging)
                text_features = None
                if use_auto_tagging and not text_query:
                    auto_text_query, query_embedding = feature_extractor.generate_text_query(image_tensor)
                    tags, scores = feature_extractor.get_image_tags(image_tensor)
                    
                    # Display detected tags with confidence scores
                    if tags:  # Only show tags section if we have non-zero confidence tags
                        st.markdown("### 🏷️ Detected Tags")
                        for tag, score in zip(tags, scores):
                            st.write(f"- {tag}: {score:.1f}%")
                        
                        st.markdown("### 🤖 Generated Query")
                        st.info(f'"{auto_text_query}"')
                        
                        # Use the pre-computed query embedding
                        text_features = query_embedding
                    else:
                        st.warning("No confident tags detected. Using pure visual search.")
                
                elif text_query:
                    text_features = feature_extractor.extract_text_features(text_query)
                
                # Search similar items
                if text_features is not None:
                    # Hybrid search with visual and text features
                    distances, indices = searcher.search(
                        image_features,
                        k=50,
                        text_features=text_features,
                        visual_weight=visual_weight,
                        merge_strategy=merge_strategy
                    )
                else:
                    # Pure visual search
                    distances, indices = searcher.search(image_features, k=50)
                
                # Process and display results
                processed_results = process_search_results(distances, indices, searcher)
                
                # Display results
                st.markdown("### 🛍️ Similar Products")
                display_results(processed_results, num_results)
        
        else:  # Pure text search
            with st.spinner('🔍 Finding similar products...'):
                # Display search mode
                st.markdown("### 📝 Text Search")
                
                # Extract text features
                text_features = feature_extractor.extract_text_features(text_query)
                
                # Search using text features only
                distances, indices = searcher.search(
                    text_features,
                    k=50,
                    text_features=None  # No need for hybrid search
                )
                
                # Process and display results
                processed_results = process_search_results(distances, indices, searcher)
                
                # Display results
                st.markdown("### 🛍️ Similar Products")
                display_results(processed_results, num_results)

if __name__ == "__main__":
    main() 