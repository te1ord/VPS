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
    index_dir = Path("data/baseline/index")
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
        image = Image.open(upload).convert('RGB')
        return image
    return None

def main():
    st.title(config['app']['title'])
    st.write(config['app']['description'])
    
    # Load models and pre-computed index
    with st.spinner('Loading models...'):
        preprocessor, feature_extractor, searcher = load_models()
    
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
    
        # Show weighting slider only when doing hybrid search
        show_weight_slider = (uploaded_file is not None and 
                            (text_query or (use_auto_tagging and not text_query)))
        
        if show_weight_slider:
            visual_weight = st.slider(
                "Visual-Text Weight Balance",
                min_value=0.0,
                max_value=1.0,
                value=0.7 if use_auto_tagging and not text_query else 0.5,
                help="0 = Text only, 1 = Image only"
            )
    
    # Add control for number of similar products
    num_results = st.slider("Number of similar products to show", min_value=1, max_value=20, value=6)
    
    # Process search
    if uploaded_file is not None or text_query:
        image_features = None
        text_features = None
        auto_text_query = None
        
        # Process image if uploaded
        if uploaded_file is not None:
            st.markdown("### 📤 Uploaded Image")
            image = load_and_process_image(uploaded_file)
            
            # Center the image using columns
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                st.image(
                    image,
                    use_container_width=True,
                    caption="Your uploaded image",
                    output_format="PNG"
                )
            
            # Convert PIL image to tensor
            image_tensor = preprocessor.preprocess(image).unsqueeze(0)
            
            # Add shape verification
            if image_tensor.shape != (1, 3, 224, 224):
                st.error(f"Invalid tensor shape: {image_tensor.shape}. Should be (1, 3, 224, 224)")
                st.stop()
            
            # Extract image features
            image_features = feature_extractor.extract_features(image_tensor)
            
            # Generate automatic text query if enabled and no user query provided
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
                    text_features = None
        
        # Process text if provided by user
        if text_query:
            text_features = feature_extractor.extract_text_features(text_query)
        
        # Search similar images
        with st.spinner('🔍 Finding similar products...'):
            if image_features is not None and text_features is not None:
                # Combined search (either with user query or auto-generated tags)
                distances, indices = searcher.search(
                    image_features,
                    k=num_results,
                    text_features=text_features,
                    visual_weight=visual_weight
                )
            elif image_features is not None:
                # Pure visual search
                distances, indices = searcher.search(image_features, k=num_results)
            else:
                # Text-only search
                distances, indices = searcher.search(text_features, k=num_results)
            
            # Display search mode
            search_mode = (
                "🔄 Hybrid Search (Image + Text)" if image_features is not None and text_features is not None
                else "👁️ Visual Search" if image_features is not None
                else "📝 Text Search"
            )
            st.markdown(f"### {search_mode}")
            
            if image_features is not None and text_features is not None:
                st.markdown(f"*Using {visual_weight:.1%} visual and {(1-visual_weight):.1%} textual features*")
            
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
                    st.progress(float(round(score, 2)))

if __name__ == "__main__":
    main() 