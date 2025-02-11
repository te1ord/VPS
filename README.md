# Visual Product Search 🔍

A hybrid visual-semantic search system that combines CLIP embeddings with object detection for precise product similarity matching.

![System Overview](https://via.placeholder.com/800x400.png?text=Visual+Search+Workflow)

## Table of Contents

- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup-and-installation)
- [Usage Guide](#usage-guide)
- [Detailed Description](#detailed-description)
- [Potential Improvements](#potential-improvements)

## Key Features 
- **Visual Feature Extraction** - Extract visual embeddings from full images and detected regions
- **Embeddings Search Strategy** - Use cosine similarity for embeddings search 
- **Results Filtering**  
  - Region size filtering (<10% area)
  - Human element exclusion
- **Search refinement using text embeddings**  
  - User's text query if provided
  - Zero-shot image tagging and query generation using 
  - Embeddings fusion or Score-based merging of visual and textual embeddings
- **Interactive UI** - Region proposals & parameters tuning

## Tech Stack 
| Component              | Technology                          |
|------------------------|-------------------------------------|
| **Feature Extraction** | OpenAI CLIP (ViT-B/32)             |
| **Object Detection**   | YOLOv8 (Open Images fine-tuned)     |
| **Similarity Search**  | FAISS (HNSW/IVF/Flat indices)      |


## Project Structure 
```bash
visual_product_search/
├── data/
│   ├── images/           # Product image database
│   └── index/            # FAISS indices & metadata
├── src/
│   ├── app/              # Streamlit UI & controllers
│   ├── feature_extraction/ # CLIP embedding management
│   ├── search/           # FAISS index handlers
│   ├── preprocessing/    # Image transforms & detection
│   └── config/           # Model parameters & categories
│   └── scripts/          # Build index 
```

## Setup & Installation 

1. **Clone repository**
```bash
git clone https://github.com/te1ord/VPS.git
cd VPS
```

2. **Install dependencies**
```bash
conda create -n vps python=3.10
conda activate vps
pip install -r requirements.txt
```

4. **Build FAISS index**
Put your images in `data/images` folder and run script to build index.
```bash
python src/scripts/build_index.py --data-dir data/images --output-dir data/index
```
Also you can rebuiid inndex from Flat to HNSW/IVF using 'rebuild_index.py' script.
```bash
# Convert to IVF
python src/scripts/rebuild_index.py \
  --input-index data/index/product_index.faiss \
  --output-index data/index/product_index_ivf.faiss \
  --index-type IVF

# Convert to HNSW
python src/scripts/rebuild_index.py \
  --input-index data/index/product_index.faiss \
  --output-index data/index/product_index_hnsw.faiss \
  --index-type HNSW
```

5. **Launch application**
```bash
streamlit run src/app/streamlit_app.py
```

## Usage Guide 

1. **Basic Search**
   - Upload product image
   - Optional: Add text query
   - Adjust visual/text weighting (50/50 by default)

2. **Advanced Features**
   - Enable **Region Detection** to focus on specific product parts
   - Use **Auto-Tagging** for zero-shot query generation 
   - Choose index type (HNSW/IVF/Flat) in settings
   - Toggle between score merging strategies

## Detailed Description

## Potential Improvements
