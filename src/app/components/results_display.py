import streamlit as st
from PIL import Image
from typing import List, Tuple
from src.search.faiss_index import IndexItem
from src.utils.image_utils import crop_image

def display_results(processed_results: List[Tuple[str, List[Tuple[IndexItem, float]]]], num_results: int, search_time: float = None):
    """Display search results grouped by original images."""
    if search_time is not None:
        st.info(f"🕒 Search completed in {search_time:.3f} seconds")
    
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

def display_auto_tags(tags: List[str], scores: List[float], query: str):
    """Display detected tags and generated query."""
    if tags:  # Only show tags section if we have non-zero confidence tags
        st.markdown("### 🏷️ Detected Tags")
        for tag, score in zip(tags, scores):
            st.write(f"- {tag}: {score:.1f}%")
        
        st.markdown("### 🤖 Generated Query")
        st.info(f'"{query}"') 