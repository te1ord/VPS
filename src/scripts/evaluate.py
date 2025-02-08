import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.search.faiss_index import FAISSSearcher

def load_embeddings_and_paths(index_path, paths_path):
    """Load embeddings and image paths from index files."""
    searcher = FAISSSearcher()
    searcher.load_index(str(index_path))
    
    with open(paths_path, 'r') as f:
        image_paths = json.load(f)
    
    # Extract embeddings from FAISS index
    embeddings = np.array([searcher.index.reconstruct(i) 
                          for i in range(searcher.index.ntotal)])
    return embeddings, image_paths

def analyze_embeddings(embeddings, output_dir):
    """Perform embedding space analysis and visualization."""
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Dimensionality reduction
    pca = PCA(n_components=2)
    tsne = TSNE(n_components=2, perplexity=30)
    
    print("Running dimensionality reduction...")
    emb_pca = pca.fit_transform(embeddings)
    emb_tsne = tsne.fit_transform(embeddings)
    
    # Cluster number selection
    print("\nDetermining optimal cluster count...")
    range_n_clusters = range(2, 20)
    best_score = -1
    cluster_metrics = []
    
    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        silhouette = silhouette_score(embeddings, clusters)
        davies_bouldin = davies_bouldin_score(embeddings, clusters)
        
        cluster_metrics.append({
            'n_clusters': n_clusters,
            'silhouette': float(silhouette),
            'davies_bouldin': float(davies_bouldin)
        })
        
        # Prefer higher silhouette score but penalize very similar clusters
        combined_score = silhouette - (davies_bouldin/10)
        
        if combined_score > best_score:
            best_score = combined_score
            best_n = n_clusters
            best_kmeans = kmeans
    
    # Final clustering with selected number
    print(f"Selected optimal cluster count: {best_n}")
    clusters = best_kmeans.fit_predict(embeddings)
    silhouette = silhouette_score(embeddings, clusters)
    davies_bouldin = davies_bouldin_score(embeddings, clusters)

    # Visualization of cluster metrics
    plt.figure(figsize=(10, 5))
    plt.plot(range_n_clusters, [m['silhouette'] for m in cluster_metrics], label='Silhouette')
    plt.plot(range_n_clusters, [m['davies_bouldin'] for m in cluster_metrics], label='Davies-Bouldin')
    plt.xlabel('Number of clusters')
    plt.ylabel('Score')
    plt.title('Cluster Quality Metrics')
    plt.axvline(best_n, color='r', linestyle='--', label='Selected')
    plt.legend()
    plt.grid(True)
    plt.savefig(Path(output_dir) / 'cluster_metrics.png')
    plt.close()

    # Visualization
    plt.figure(figsize=(15, 6))
    
    plt.subplot(121)
    plt.scatter(emb_pca[:, 0], emb_pca[:, 1], c=clusters, cmap='tab10', alpha=0.6)
    plt.title(f'PCA Projection\nSilhouette: {silhouette:.2f}, Davies-Bouldin: {davies_bouldin:.2f}')
    plt.colorbar()
    
    plt.subplot(122)
    plt.scatter(emb_tsne[:, 0], emb_tsne[:, 1], c=clusters, cmap='tab10', alpha=0.6)
    plt.title('t-SNE Projection')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'embedding_analysis.png')
    plt.close()
    
    return {
        'silhouette_score': float(silhouette),
        'davies_bouldin_score': float(davies_bouldin),
        'selected_clusters': best_n,
        'cluster_metrics': cluster_metrics
    }

def evaluate_embeddings(index_dir="data/index", output_dir="data/evaluation"):
    """Main function to run embedding evaluation."""
    index_path = Path(index_dir) / "product_index.faiss"
    paths_path = Path(index_dir) / "image_paths.json"
    
    print("Loading embeddings and paths...")
    embeddings, image_paths = load_embeddings_and_paths(index_path, paths_path)
    
    print(f"Analyzing {len(embeddings)} embeddings...")
    metrics = analyze_embeddings(embeddings, output_dir)
    
    # Save metrics
    metrics_path = Path(output_dir) / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nSaved evaluation results to {output_dir}")

if __name__ == "__main__":
    evaluate_embeddings() 