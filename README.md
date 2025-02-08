# Visual Product Search

A Proof-of-Concept (PoC) for a visual product search system that allows users to find similar products using image-based search.

## Features

- Image-based product search using CLIP embeddings
- Fast similarity search with FAISS
- Simple and intuitive Streamlit web interface
- Modular architecture ready for future extensions (e.g., object detection)

## Project Structure

```
visual_product_search/
├── src/                    # Source code
│   ├── feature_extraction/ # Feature extraction using CLIP
│   ├── search/            # FAISS similarity search
│   ├── preprocessing/     # Image preprocessing
│   ├── utils/            # Utility functions
│   └── app/              # Streamlit web application
├── config/               # Configuration files
└── data/                # Data directory
    └── index/          # FAISS index storage
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run src/app/streamlit_app.py
```

## Usage

1. Upload an image through the web interface
2. The system will:
   - Preprocess the image
   - Extract features using CLIP
   - Find similar products using FAISS
   - Display results ranked by similarity

## Technical Details

- Feature Extraction: OpenAI's CLIP model
- Similarity Search: Facebook AI Similarity Search (FAISS)
- Frontend: Streamlit
- Image Processing: PyTorch and torchvision

## Future Improvements

- Add object detection for better product localization
- Implement image augmentation for better robustness
- Add support for multiple similarity metrics
- Optimize index for larger datasets 