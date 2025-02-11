import faiss
import numpy as np
import sys
import os
import argparse
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def verify_index_correspondence(original_index, new_index, vectors):
    """Verify that the new index returns the same nearest neighbors as the original."""
    # Test with a few random queries
    num_tests = min(10, vectors.shape[0])
    test_indices = np.random.choice(vectors.shape[0], num_tests, replace=False)
    
    for idx in test_indices:
        query = vectors[idx:idx+1]
        
        # Search in both indices
        D1, I1 = original_index.search(query, 5)
        D2, I2 = new_index.search(query, 5)
        
        
        # print(f"Original index returned: {I1}, New index returned: {I2}")
        # print(f"Original index returned: {I1[0][0]}, New index returned: {I2[0][0]}")
        # Check if the top result is the same (should be the query vector itself)
        if I1[0][0] != idx or I2[0][0] != idx:
            print(f"Warning: Index correspondence test failed for vector {idx}")
            print(f"Original index returned: {I1[0][0]}, New index returned: {I2[0][0]}")
            return False
    return True

def convert_index(input_path: str, output_path: str, index_type: str, nprobe: int = 10, ef_search: int = 16):
    """
    Convert FAISS index to a different type while preserving vector order.
    Args:
        input_path: Path to input index
        output_path: Path to save converted index
        index_type: Target index type ('IVF' or 'HNSW')
        nprobe: Number of clusters to probe for IVF
        ef_search: Search time/quality trade-off for HNSW
    """
    print(f"Converting index to {index_type}...")
    
    # Load original index
    original_index = faiss.read_index(input_path)
    
    # Extract vectors
    if not hasattr(original_index, 'reconstruct'):
        raise ValueError("Original index doesn't support reconstruction")
    
    num_vectors = original_index.ntotal
    dimension = original_index.d
    print(f"Extracting {num_vectors} vectors of dimension {dimension}")
    vectors = np.vstack([original_index.reconstruct(i) for i in range(num_vectors)])
    
    # Create new index based on type
    if index_type == 'IVF':
        nlist = min(int(np.sqrt(num_vectors)), 100)  # Rule of thumb for nlist
        quantizer = faiss.IndexFlatIP(dimension)
        new_index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        new_index.nprobe = nprobe
        print(f"Created IVF index with nlist={nlist}, nprobe={nprobe}")
        
        # Train IVF index
        print("Training IVF index...")
        new_index.train(vectors)
        
    elif index_type == 'HNSW':
        M = 16  # Number of connections per layer
        new_index = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_INNER_PRODUCT)
        new_index.hnsw.efConstruction = 40
        new_index.hnsw.efSearch = ef_search
        print(f"Created HNSW index with M={M}, ef_search={ef_search}")
    else:
        raise ValueError(f"Unsupported index type: {index_type}")
    
    # Add vectors to new index
    print("Adding vectors to new index...")
    new_index.add(vectors)
    
    # Verify index correspondence
    print("Verifying index correspondence...")
    if not verify_index_correspondence(original_index, new_index, vectors):
        raise ValueError("Index verification failed! The new index may not maintain the same order as the original.")
    
    # Save new index
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving index to {output_path}")
    faiss.write_index(new_index, str(output_path))
    
    print("Conversion complete!")
    return new_index

def main():
    parser = argparse.ArgumentParser(description="Convert FAISS index to different types")
    parser.add_argument("--input-index", type=str, required=True, help="Path to input index")
    parser.add_argument("--output-index", type=str, required=True, help="Path to save converted index")
    parser.add_argument("--index-type", type=str, choices=['IVF', 'HNSW'], required=True, help="Target index type")
    parser.add_argument("--nprobe", type=int, default=10, help="Number of clusters to probe for IVF")
    parser.add_argument("--ef-search", type=int, default=16, help="ef_search parameter for HNSW")
    
    args = parser.parse_args()
    
    try:
        convert_index(
            args.input_index,
            args.output_index,
            args.index_type,
            args.nprobe,
            args.ef_search
        )
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()