import streamlit as st
import torch
import yaml
from pathlib import Path
from PIL import Image
import sys
import os
import json

# Add src to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.preprocessing.image_processor import ImageProcessor
from src.feature_extraction.clip_extractor import CLIPExtractor
from src.search.faiss_index import FAISSSearcher

# Load configuration
with open("config/config.yaml", 'r') as f:
    config = yaml.safe_load(f)  

@st.cache_resource
def load_models():
    """Load and cache models and pre-computed index."""
    preprocessor = ImageProcessor()
    feature_extractor = CLIPExtractor()
    searcher = FAISSSearcher()
    
    # Load pre-computed index and paths
    index_dir = Path("data/index")
    index_path = index_dir / "product_index.faiss"
    paths_path = index_dir / "image_paths.json"
    
    if not index_path.exists() or not paths_path.exists():
        st.error("Index not found. Please run build_index.py first!")
        st.stop()
    
    # Load index
    searcher.load_index(str(index_path))
    
    # Load image paths
    with open(paths_path, 'r') as f:
        searcher.image_paths = json.load(f)
    
    return preprocessor, feature_extractor, searcher

def load_and_process_image(upload):
    """Load and process uploaded image."""
    if upload is not None:
        # Read image
        image = Image.open(upload).convert('RGB')
        return image
    return None

def main():
    st.title(config['app']['title'])
    st.write(config['app']['description'])
    
    # Load models and pre-computed index
    with st.spinner('Loading models...'):
        preprocessor, feature_extractor, searcher = load_models()
    
    # File uploader with custom styling
    st.markdown("### 🖼️ Upload an Image")
    uploaded_file = st.file_uploader("", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")
    
    # Add control for number of similar products
    num_results = st.slider("Number of similar products to show", min_value=1, max_value=20, value=6)
    
    if uploaded_file is not None:
        # Display uploaded image with border
        st.markdown("### 📤 Uploaded Image")
        image = load_and_process_image(uploaded_file)
        
        # Center the image using columns with higher quality
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.image(
                image,
                use_container_width=True,
                caption="Your uploaded image (click for full resolution)",
                output_format="PNG"  # Use lossless format
            )
        
        # Process image and search
        with st.spinner('🔍 Finding similar products...'):
            # Convert PIL image to tensor
            image_tensor = preprocessor.preprocess(image).unsqueeze(0)
            
            # Extract features
            features = feature_extractor.extract_features(image_tensor)
            
            # Search similar images
            distances, indices = searcher.search(features, k=num_results)
            
            # Display results with better layout
            st.markdown("### 🎯 Most Similar Products")
            similar_paths = searcher.get_image_paths(indices)
            
            # Create columns for results
            cols = st.columns(3)  # 3 images per row
            
            for idx, (path, score) in enumerate(zip(similar_paths, distances[0])):
                with cols[idx % 3]:
                    st.markdown(f"**Rank {idx + 1}**")
                    st.image(
                        path,
                        caption=f"Similarity: {score:.3f}",
                        width=200,
                        use_container_width=True,
                        output_format="JPEG"
                    )
                    st.progress(float(round(score, 2)))  # Add progress bar for similarity score

if __name__ == "__main__":
    main() 