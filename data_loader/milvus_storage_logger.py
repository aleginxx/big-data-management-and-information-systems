"""
milvus_storage_logger_simple.py

Inserts vectors into Milvus for different configurations and logs:
- NumEntities
- ApproxBytes
"""

import os
import time
import csv
import json
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# ---------------- Parameters ----------------
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
BASE_COLLECTION_NAME = "NewsArticle"
OUTPUT_CSV = "milvus_storage_stats.csv"
CACHE_FILE = "milvus_vector_data.json"

BATCH_SIZES = [4, 8, 16, 32, 128]
METRICS = ["L2", "COSINE", "IP"]
VECTOR_DIMS = [32, 64, 128, 384]
VECTOR_SUBSETS = [1000, 5000]

# ---------------- Helpers ----------------
def load_embeddings():
    if not os.path.exists(CACHE_FILE):
        raise FileNotFoundError(f"{CACHE_FILE} not found. Generate embeddings first.")
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def approx_storage_bytes(collection, vector_dim):
    n = collection.num_entities
    vector_bytes = n * vector_dim * 4  # float32
    meta_bytes = n * (8 + 200 + 50)    # id (int64=8 bytes), title 200 chars, category 50 chars
    return vector_bytes + meta_bytes

# ---------------- Main ----------------
def main():
    # Connect to Milvus
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    print(f"Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")

    # Load embeddings
    vectors_data = load_embeddings()

    # Prepare CSV
    new_file = not os.path.exists(OUTPUT_CSV)
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        fieldnames = ["Collection", "NumEntities", "ApproxBytes", "VectorDim", "VectorSubset", "BatchSize", "Metric", "Timestamp"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()

        # Loop over configurations
        for dim in VECTOR_DIMS:
            for subset in VECTOR_SUBSETS:
                subset_vectors = [
                    {
                        "id": obj["id"],
                        "title": obj["title"],
                        "content": obj["content"],
                        "category": obj["category"],
                        "vector": obj["vector"][:dim]
                    }
                    for obj in vectors_data[:subset]
                ]

                ids = [obj["id"] for obj in subset_vectors]
                titles = [obj["title"] for obj in subset_vectors]
                contents = [obj["content"] for obj in subset_vectors]
                categories = [obj["category"] for obj in subset_vectors]
                vectors = [obj["vector"] for obj in subset_vectors]

                for metric in METRICS:
                    for batch_size in BATCH_SIZES:
                        coll_name = f"{BASE_COLLECTION_NAME}_{metric}_{dim}d_{subset}v_bs{batch_size}"

                        # Drop if exists
                        if utility.has_collection(coll_name):
                            print(f"Dropping existing collection '{coll_name}'...")
                            utility.drop_collection(coll_name)

                        # Create collection
                        fields = [
                            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=200),
                            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=1500),
                            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=50),
                            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
                        ]
                        schema = CollectionSchema(fields, description=f"News Articles with {metric} embeddings")
                        collection = Collection(name=coll_name, schema=schema)

                        # Insert in batches
                        for i in range(0, subset, batch_size):
                            collection.insert([
                                ids[i:i+batch_size],
                                titles[i:i+batch_size],
                                contents[i:i+batch_size],
                                categories[i:i+batch_size],
                                vectors[i:i+batch_size]
                            ])
                        collection.flush()
                        print(f"[{coll_name}] Inserted {collection.num_entities} entities")

                        # Log to CSV
                        approx_bytes = approx_storage_bytes(collection, dim)
                        writer.writerow({
                            "Collection": coll_name,
                            "NumEntities": collection.num_entities,
                            "ApproxBytes": approx_bytes,
                            "VectorDim": dim,
                            "VectorSubset": subset,
                            "BatchSize": batch_size,
                            "Metric": metric,
                            "Timestamp": time.time()
                        })

    connections.disconnect("default")
    print("Done logging Milvus storage stats.")

if __name__ == "__main__":
    main()
