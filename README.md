# Visual Product Search 🔍

A hybrid visual-semantic search system that combines CLIP embeddings with object detection for precise product similarity matching.

[System Overview](https://drive.google.com/file/d/1d60HvfKysHMbRIm6ZRCnDUk87ueSZzSp/view?usp=sharing)

## Table of Contents

- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage Guide](#usage-guide)
- [Detailed Description](#detailed-description)
- [Conclusion](#conclusion)
- [Potential Improvements](#potential-improvements)
- [Related Links](#related-links)

## Key Features 
- **Visual Feature Extraction** 
  - Extract visual embeddings from full images and detected regions
- **Embeddings Search Strategy** 
  - Use cosine similarity for embeddings search 
- **Results Filtering**  
  - Region size filtering (<10% area)
  - Human element exclusion
- **Search refinement using text embeddings**  
  - User's text query if provided
  - Zero-shot image tagging and query generation using 
  - Embeddings fusion or Score-based merging of visual and textual embeddings
- **Interactive UI** 
  - Region proposals & parameters tuning

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

## Setup 

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

3. **Build FAISS index**
Put your images in `data/images` folder and run script to build index
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

4. **Launch application**
```bash
streamlit run src/app/app.py
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

My approach involves building visual embeddings using CLIP and indexing them with FAISS, applied both to entire images and specific regions detected by YOLOv8 ([fine-tuned](https://docs.ultralytics.com/datasets/detect/open-images-v7/#dataset-yaml) on the [Open Images dataset](https://storage.googleapis.com/openimages/web/factsfigures_v7.html#class-definitions)) with confidence threshold 0.5 and . This process generated approximately 17,454 embedding vectors from an initial set of 8,000 images (+ 9454 detected objects). I stored the metadata of these detections to align the embeddings of detected regions with their corresponding original images.

For similarity search, I utilized Cosine Similarity as CLIP is optimized for this metric, experimenting with different [FAISS](https://github.com/facebookresearch/faiss) index types such as FlatIndex (exhastive search), HNSW(graph-based search), and IVF(involves clustering and approximate nearest neighbor search).

Additionally, I implemented a search refinement option using a hybrid approach (visual + textual). Users can provide a text query, or, if omitted, the system automatically generates tags using CLIP’s zero-shot capabilities to enhance search results using pre-defined set of classes. Those classes also have potential to be used for filtering out irrelevant results or categorize our serach into different areas.

Visual Product Search Functionality:
	1.	Image Upload & Search:
Users upload an image (with an optional text query). The system extracts embeddings and performs a similarity search across the entire embedding set. The matched embeddings are grouped by their corresponding original images, and users can inspect the specific regions that yielded high similarity scores.
	2.	Filtering:
Cropped regions occupying less than 10% of the original image are filtered out. Additionally, detections related to people are excluded to focus on product-specific results.
	3.	Region Proposals:
Users can opt to run the YOLO model on their uploaded image to generate region proposals. They can then select a region of interest, triggering the same embedding extraction and similarity search pipeline, but focused solely on the selected region rather than the whole image.
    4. Scoring:
Search results are scored based on the cosine similarity between the query embedding and the embeddings of the detected regions. 
    5.	Customizable Search Options:
1. Users can choose which FAISS index to use for their search (potential scaling our dataset to more than 17k embeddings).
 
 2. Choose the embedding combination strategy—either merge visual and textual embeddings using a linear combination or perform independent searches and combine the results by weighting their similarity scores. This functionality was added because search performance can degrade when using Approximate Nearest Neighbor (ANN) methods compared to a Flat index, especially with linear combinations. As a baseline, using linear combination weighting is acceptable since combining unit vectors typically results in a vector that balances between the textual and visual embeddings, pointing in a similar direction. This approach can perform well in exhaustive searches but may introduce uncertainty in ANN searches. Further exploration is needed to understand and address this behavior.

 Here are some papers that might be relevant to this issue with embeddings combination, yet I haven't read them yet:
 - [Training-free Zero-shot Composed Image Retrieval via Weighted Modality Fusion and Similarity](https://arxiv.org/pdf/2409.04918)
 - [Cross-modal Feature Alignment and Fusion for Composed Image Retrieval](https://www.semanticscholar.org/paper/Cross-modal-Feature-Alignment-and-Fusion-for-Image-Wan-Wang/a1bd6fdf403df0cdd816dc66b38cff722404b2d9)
 - [CLIP-ProbCR:CLIP-based Probability embedding Combination Retrieval](https://dl.acm.org/doi/abs/10.1145/3652583.3657611)
 - [Conditioned and composed image retrieval combining and partially fine-tuning CLIP-based features](https://ieeexplore.ieee.org/document/9857242)
 - [Composed Image Retrieval using Contrastive Learning and Task-oriented CLIP-based Features](https://dl.acm.org/doi/10.1145/3617597)


## Potential Improvements
- [ ] Elaborate on fixed classes dataset creation which allows object detector fine-tuning as well as CLIP fine tuning resulting in better embeddings and tagging
- [ ] Detector model architectures exploration
- [ ] Region detection thresholds tuning
- [ ] Tags list refinement 
- [ ] Embeddings combination strategy 
- [ ] Search results filtering based on tags/detection classes/confidence scores 
- [ ] Search results scoring/ranking tuning
- [ ] Search results on whole image/crops weighting

**TO DO:**
- [ ] Try other embedding models - I wanted to try [SWIN transformer](https://huggingface.co/microsoft/swinv2-base-patch4-window12-192-22k), for example, but have no time now
- [ ] Try other object detectors - I wanted to try something from  [BigDetection: A Large-scale Benchmark for Improved Object Detector Pre-training](https://github.com/amazon-science/bigdetection)

## Conclusion

I decided to approach Visual Product Search without a pre-filtering phase for my database, focusing instead on extracting relevant embeddings and using them directly for search. I assumed that embeddings of real products and noise examples would naturally separate in the embedding space. I believe this approach works even better when we have image-description pairs to fine-tune CLIP, as well as a large, diverse dataset of examples for performing the search. It these conditions are met, we also can use this fine-tune CLIP for pre-filtering phase if needed.

## Related Links

I have not explored all presented materials in detail, therefore I do not guarantee they contain correct implementation/information, but assume they may be useful.

**Repositories**
- [CLIP](https://github.com/openai/CLIP) + [brief description](https://openai.com/index/clip/)
- [Getting started withFAISS](https://github.com/facebookresearch/faiss/wiki/Getting-started)
- [End-to-End Visual Search](https://github.com/lmy1108/end2end-visualsearch/tree/master)
There are some project specifically tuned for fashion domain:
- [Visual Search with Image Embedding](https://github.com/eyereece/visual-search-with-image-embedding)
- [eBay Challenge](https://github.com/01BB01/eBayChallenge/tree/main) 
- [Fashion Visual Search](https://github.com/yainage90/fashion-visual-search) 

**Models**
- [Fashion Object Detection trained in repo above](https://huggingface.co/yainage90/fashion-object-detection)
- [Fashion CLIP](https://github.com/patrickjohncyh/fashion-clip?tab=readme-ov-file#user-content-fn-1-2e33f967522c8a4b4b393f6a84fd63e8)

**Datasets**
- [fashinopedia](https://fashionpedia.github.io/home/)
- [modanet](https://github.com/eBay/modanet)
- [Amazon dataset samples](https://github.com/luminati-io/Amazon-dataset-samples)
- [Amazon product reviews dataset](https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews)

**Challenges**
- [eBay eProduct Visual Search competition](https://eval.ai/web/challenges/challenge-page/1541/overview)

**Papers**
- [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/pdf/2103.00020)
- [V2L: Leveraging Vision and Vision-language Models into Large-scale Product Retrieval](https://arxiv.org/pdf/2207.12994)
- [Large-Scale Product Retrieval with Weakly Supervised Representation Learning](https://arxiv.org/pdf/2208.00955)
- [eProduct: A Million-Scale Visual Search Benchmark to Address Product Recognition Challenges](https://arxiv.org/pdf/2107.05856)



