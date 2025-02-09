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

# Add src to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.preprocessing.image_processor import ImageProcessor
from src.feature_extraction.clip_extractor import CLIPExtractor
from src.search.faiss_index import FAISSSearcher
from src.preprocessing.object_detector import ObjectDetector

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

def crop_image(image: Image.Image, bbox: tuple) -> Image.Image:
    """Crop image to bounding box."""
    x1, y1, x2, y2 = map(int, bbox)
    return image.crop((x1, y1, x2, y2))

def process_regions(image, preprocessor, feature_extractor, detector):
    """Process image and extract features for full image and detected regions."""
    features_list = []
    detections_list = []
    
    # First, process full image
    full_image_tensor = preprocessor.preprocess(image).unsqueeze(0)
    full_image_features = feature_extractor.extract_features(full_image_tensor)
    
    # Add full image
    w, h = image.size
    features_list.append(full_image_features)
    detections_list.append({
        'bbox': (0, 0, w, h),
        'class_name': 'full_image',
        'confidence': 1.0,
        'is_full_image': True
    })
    
    # Detect and process objects
    object_detections = detector.detect_objects(image)
    
    # Process each detection
    for detection in object_detections:
        if detection.class_name == "whole_image":
            continue
        
        # Process cropped image
        image_tensor = preprocessor.preprocess(detection.cropped_image).unsqueeze(0)
        features = feature_extractor.extract_features(image_tensor)
        
        features_list.append(features)
        detections_list.append({
            'bbox': detection.bbox,
            'class_name': detection.class_name,
            'confidence': detection.confidence,
            'is_full_image': False
        })
    
    return torch.cat(features_list, dim=0), detections_list

def display_results(results, num_results):
    """Display search results grouped by original images."""
    # Group results by original image
    image_results = defaultdict(list)
    for item, score in results:
        image_results[item.original_image_path].append((item, score))
    
    # Sort images by their best matching region's score
    sorted_images = sorted(
        image_results.items(),
        key=lambda x: max(score for _, score in x[1]),
        reverse=True
    )[:num_results]
    
    # Display results in a grid
    cols = st.columns(3)
    
    for idx, (img_path, detections) in enumerate(sorted_images):
        # Sort detections by score
        detections = sorted(detections, key=lambda x: x[1], reverse=True)
        best_detection = detections[0]
        
        with cols[idx % 3]:
            # Load original image
            image = Image.open(img_path).convert('RGB')
            
            # Display original image
            st.image(image, caption=f"Match Score: {best_detection[1]:.3f}", use_container_width=True)
            
            # Add expander for detected regions if there are any non-full-image detections
            sub_regions = [(det, score) for det, score in detections if not det.is_full_image]
            if sub_regions:
                with st.expander("View Detected Regions"):
                    for det, score in sub_regions:
                        st.markdown(f"##### {det.class_name} (Score: {score:.3f})")
                        region = crop_image(image, det.bbox)
                        st.image(region, use_container_width=True)

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
        visual_weight = st.slider(
            "Visual-Text Weight Balance",
            min_value=0.0,
            max_value=1.0,
            value=0.7 if use_auto_tagging else 0.5,
            help="0 = Text only, 1 = Image only"
        )
    else:
        visual_weight = 0.7 if use_auto_tagging else 0.5  # Default weights
    
    # Add control for number of similar products
    num_results = st.slider("Number of similar products to show", min_value=1, max_value=20, value=6)
    
    # Process search
    if uploaded_file is not None or text_query:
        if uploaded_file is not None:
            # Load and display input image
            image = load_and_process_image(uploaded_file)
            
            # Display input image
            st.markdown("### 📤 Input Image")
            st.image(image, use_container_width=True)
            
            # Process image and search
            with st.spinner('🔍 Finding similar products...'):
                # Process image and get features for all regions
                all_features, detections = process_regions(image, preprocessor, feature_extractor, detector)
                
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
                    auto_text_query, query_embedding = feature_extractor.generate_text_query(
                        preprocessor.preprocess(image).unsqueeze(0)
                    )
                    tags, scores = feature_extractor.get_image_tags(
                        preprocessor.preprocess(image).unsqueeze(0)
                    )
                    
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
                    # For each region, compute combined similarity with text
                    region_scores = []
                    for i, (features, detection) in enumerate(zip(all_features, detections)):
                        region_features = features.unsqueeze(0)
                        
                        # Apply region-specific weighting
                        region_visual_weight = visual_weight
                        if detection['is_full_image']:
                            # Give slightly more weight to full image in hybrid search
                            region_visual_weight = min(1.0, visual_weight * 1.2)
                        
                        distances, indices = searcher.search(
                            region_features,
                            k=50,
                            text_features=text_features,
                            visual_weight=region_visual_weight
                        )
                        region_scores.append((distances[0], indices[0], detection))
                    
                    # Combine scores from all regions with detection confidence
                    combined_scores = defaultdict(float)
                    for distances, indices, detection in region_scores:
                        confidence_weight = 1.0 if detection['is_full_image'] else detection['confidence']
                        for d, idx in zip(distances, indices):
                            # Weight the score by detection confidence
                            weighted_score = d * confidence_weight
                            combined_scores[idx] = max(combined_scores[idx], weighted_score)
                else:
                    # Pure visual search using all regions
                    all_scores = []
                    for i, (features, detection) in enumerate(zip(all_features, detections)):
                        region_features = features.unsqueeze(0)
                        distances, indices = searcher.search(region_features, k=50)
                        
                        # Weight scores by detection confidence
                        confidence_weight = 1.0 if detection['is_full_image'] else detection['confidence']
                        weighted_distances = distances[0] * confidence_weight
                        all_scores.append((weighted_distances, indices[0], detection))
                    
                    # Combine results (take best score for each result)
                    combined_scores = defaultdict(float)
                    for distances, indices, _ in all_scores:
                        for d, idx in zip(distances, indices):
                            combined_scores[idx] = max(combined_scores[idx], d)
        
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
                
                # Convert to combined scores format for consistency
                combined_scores = {
                    idx: score for idx, score in zip(indices[0], distances[0])
                }
        
        # Convert to sorted lists
        sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        indices = np.array([[idx for idx, _ in sorted_items[:50]]])
        distances = np.array([[score for _, score in sorted_items[:50]]])
        
        # Get metadata for results
        results = list(zip(searcher.get_metadata(indices), distances[0]))
        
        # Display results
        st.markdown("### 🛍️ Similar Products")
        display_results(results, num_results)

if __name__ == "__main__":
    main() 