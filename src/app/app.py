import streamlit as st
import yaml
import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.app.components.search_interface import render_search_inputs, render_search_settings
from src.app.components.detection_interface import display_image_with_detections
from src.app.components.results_display import display_results, display_auto_tags
from src.app.model_manager import initialize_models, load_index, DEFAULT_CONFIG_PATH
from src.app.search_engine import SearchEngine

def main():
    # Load configuration
    with open(DEFAULT_CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    
    st.title(config['app']['title'])
    st.write(config['app']['description'])
    
    # Initialize models (cached)
    preprocessor, feature_extractor, detector, searcher = initialize_models()
    
    # Render search inputs (now includes index selection)
    text_query, image, use_auto_tagging, index_type, index_params = render_search_inputs()
    
    # Load appropriate index with selected parameters
    searcher = load_index(
        searcher,
        index_type,
        nprobe=index_params.get('nprobe', 10),
        ef_search=index_params.get('ef_search', 16)
    )
    
    # Initialize search engine
    search_engine = SearchEngine(
        preprocessor=preprocessor,
        feature_extractor=feature_extractor,
        searcher=searcher,
        detector=detector,
        min_region_size=config['app']['min_region_size']
    )
    
    # Get search settings
    visual_weight, merge_strategy, num_results = render_search_settings(
        has_image=image is not None,
        has_text=bool(text_query),
        use_auto_tagging=use_auto_tagging
    )
    
    # Process search
    if image is not None or text_query:
        selected_region = None
        
        if image is not None:
            # Display input image with controlled size
            st.markdown("### 📤 Input Image")
            
            # Optional region selection
            use_detection = st.checkbox("Detect and select regions of interest", value=False,
                                      help="Enable to detect and select specific regions in the image")
            
            if use_detection:
                selected_region = display_image_with_detections(image, detector)
            else:
                # Display centered image
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image(image, use_container_width=True)
            
            # Handle auto-tagging display
            if use_auto_tagging and not text_query:
                image_tensor = preprocessor.preprocess(image).unsqueeze(0)
                tags, scores = feature_extractor.get_image_tags(image_tensor)
                auto_text_query, _ = feature_extractor.generate_text_query(image_tensor)
                if tags:
                    display_auto_tags(tags, scores, auto_text_query)
                else:
                    st.warning("No confident tags detected. Using pure visual search.")
        
        # Perform search
        with st.spinner('🔍 Finding similar products...'):
            # Display search mode
            search_mode = (
                "🔄 Hybrid Search (Image + Text)" if text_query or use_auto_tagging
                else "👁️ Visual Search" if image is not None
                else "📝 Text Search"
            )
            st.markdown(f"### {search_mode}")
            
            if text_query or use_auto_tagging:
                st.markdown(f"*Using {visual_weight:.1%} visual and {(1-visual_weight):.1%} textual features*")
            
            # Perform search
            processed_results, search_time = search_engine.search(
                image=image,
                text_query=text_query,
                selected_region=selected_region,
                use_auto_tagging=use_auto_tagging,
                visual_weight=visual_weight,
                merge_strategy=merge_strategy,
                k=config['app']['search_k']
            )
            
            # Display results
            st.markdown("### 🛍️ Similar Products")
            if not processed_results:
                st.warning("No similar products found. Try adjusting the search parameters or using a different query.")
            else:
                display_results(processed_results, num_results, search_time)

if __name__ == "__main__":
    main()
