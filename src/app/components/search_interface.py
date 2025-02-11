import streamlit as st
from typing import Optional, Tuple, Dict
from PIL import Image

def render_search_inputs() -> Tuple[Optional[str], Optional[Image.Image], bool]:
    """Render and handle search input components (text and image upload)."""
    # Create two columns for search inputs
    col1, col2 = st.columns(2)
    
    with col2:
        # Text search input
        st.markdown("### 💬 Text Search")
        text_query = st.text_input("", placeholder="Describe what you're looking for...", label_visibility="collapsed")
        
        # Add index selection under text search
        st.markdown("### 🔍 Search Settings")
        index_type = st.selectbox(
            "Index Type",
            options=["FlatIP", "HNSW", "IVF"],
            help="Select the search index type to use"
        )
        
        # Show relevant parameters based on index type
        index_params = {}
        if index_type == "IVF":
            index_params['nprobe'] = st.slider(
                "Search Clusters (nprobe)", 
                1, 100, 10,
                help="Number of clusters to search. Higher = better accuracy but slower"
            )
        elif index_type == "HNSW":
            index_params['ef_search'] = st.slider(
                "Search Quality (ef_search)", 
                1, 100, 16,
                help="Search time/quality trade-off. Higher = better accuracy but slower"
            )
    
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
    
    # Load image if uploaded
    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
    
    return text_query, image, use_auto_tagging, index_type, index_params

def render_search_settings(has_image: bool, has_text: bool, use_auto_tagging: bool) -> Tuple[float, str, int]:
    """Render and handle search settings components."""
    # Show weighting slider for hybrid search
    show_weight_slider = (has_image and (has_text or use_auto_tagging))
    
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
    
    return visual_weight, merge_strategy, num_results 