import streamlit as st
import yaml
from pathlib import Path
import faiss
from typing import Tuple

from src.preprocessing.image_processor import ImageProcessor
from src.feature_extraction.clip_extractor import CLIPExtractor
from src.search.faiss_index import FAISSSearcher
from src.preprocessing.object_detector import ObjectDetector

DEFAULT_CONFIG_PATH = Path("src/config/config.yaml")

@st.cache_resource
def initialize_models(config_path: str = str(DEFAULT_CONFIG_PATH)) -> Tuple[ImageProcessor, CLIPExtractor, ObjectDetector, FAISSSearcher]:
    """Load and cache models without UI components."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    preprocessor = ImageProcessor()
    feature_extractor = CLIPExtractor()
    detector = ObjectDetector(confidence_threshold=config['app']['confidence_threshold'])
    searcher = FAISSSearcher()
    
    return preprocessor, feature_extractor, detector, searcher

def load_index(searcher: FAISSSearcher, index_type: str, nprobe: int = 10, ef_search: int = 16) -> FAISSSearcher:
    """Load appropriate index based on type."""
    index_dir = Path("data/index")
    index_paths = {
        "FlatIP": index_dir / "product_index.faiss",
        "HNSW": index_dir / "product_index_hnsw.faiss",
        "IVF": index_dir / "product_index_ivf.faiss"
    }
    
    index_path = index_paths[index_type]
    if not index_path.exists():
        st.error(f"{index_type} index not found at {index_path}. Please run rebuild_index.py first!")
        st.stop()
    
    try:
        searcher.load_index(str(index_path))
        
        # Update search parameters
        if index_type == "IVF" and isinstance(searcher.index, faiss.IndexIVFFlat):
            searcher.index.nprobe = nprobe
        elif index_type == "HNSW" and isinstance(searcher.index, faiss.IndexHNSWFlat):
            searcher.index.hnsw.efSearch = ef_search
            
    except Exception as e:
        st.error(f"Failed to load index: {str(e)}")
        st.stop()
    
    return searcher 