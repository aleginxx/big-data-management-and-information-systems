"""
This script loads a HuggingFace dataset (AG News by default), generates embeddings using a SentenceTransformer model, and inserts the data into Milvus for benchmarking. 
If embeddings already exist in milvus_vector_data.json, it skips regeneration and uses the saved file.
"""

import time
import numpy as np
import os
import torch
import json
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
BASE_COLLECTION_NAME = "NewsArticle"
VECTOR_FILE = "milvus_vector_data.json"

INSERT_BATCH_SIZE = 128
METRICS = ["L2", "COSINE", "IP"]
NUM_THREADS = 4

HF_DATASET = "ag_news"
HF_SPLIT = "train"
NO_SAMPLES = 20000
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32  # for embedding generation

LABEL_MAP = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}

# Load/Generate embeddings
if os.path.exists(VECTOR_FILE):
    print(f"Found existing {VECTOR_FILE}, loading vectors...")
    with open(VECTOR_FILE, "r", encoding="utf-8") as f:
        vectors_data = json.load(f)

else:
    print(f"No existing {VECTOR_FILE}, generating embeddings...")

    # Load dataset
    print(f"Loading dataset {HF_DATASET}.")
    dataset = load_dataset(HF_DATASET, split=HF_SPLIT)
    dataset = dataset.select(range(NO_SAMPLES))
    texts = [row["text"] for row in dataset]
    categories = [LABEL_MAP[row["label"]] for row in dataset]

    # Load embedding model
    print(f"Loading model {MODEL_NAME}.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    # Generate embeddings
    print("Generating embeddings...")
    vectors_data = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]
        with torch.no_grad():
            inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
            outputs = model(**inputs)
            batch_vectors = outputs.last_hidden_state.mean(dim=1).cpu().numpy().astype(np.float32)

        for idx, vec in enumerate(batch_vectors):
            vectors_data.append({
                "id": i+idx,
                "title": batch_texts[idx][:50],
                "content": batch_texts[idx],
                "category": categories[i+idx],
                "vector": vec.tolist()
            })

    # Save to JSON for reuse
    with open(VECTOR_FILE, "w", encoding="utf-8") as f:
        json.dump(vectors_data, f)
    print(f"Saved embeddings to {VECTOR_FILE}")

vector_dim = len(vectors_data[0]["vector"])
subset_size = len(vectors_data)
print(f"Using {subset_size} embeddings with dimension {vector_dim}")

# Insert data to milvus
connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
print(f"Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")

ids = [obj["id"] for obj in vectors_data]
titles = [obj["title"] for obj in vectors_data]
contents = [obj["content"] for obj in vectors_data]
vectors = [obj["vector"] for obj in vectors_data]
categories = [obj["category"] for obj in vectors_data]

for metric in METRICS:
    collection_name = f"{BASE_COLLECTION_NAME}_{metric}_{vector_dim}d_{subset_size}v"

    if utility.has_collection(collection_name):
        print(f"Dropping existing collection '{collection_name}'.")
        utility.drop_collection(collection_name)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=1500),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=50),  
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim)
    ]
    schema = CollectionSchema(fields, description=f"News Articles with {metric} embeddings")
    collection = Collection(name=collection_name, schema=schema)
    print(f"Created collection '{collection_name}' with metric {metric}")

    batches = [
        (ids[i:i+INSERT_BATCH_SIZE],
            titles[i:i+INSERT_BATCH_SIZE],
            contents[i:i+INSERT_BATCH_SIZE],
            categories[i:i+INSERT_BATCH_SIZE],
            vectors[i:i+INSERT_BATCH_SIZE]
        )
        for i in range(0, subset_size, INSERT_BATCH_SIZE)
    ]

    def insert_batch(batch_tuple):
        batch_ids, batch_titles, batch_contents, batch_categories, batch_vectors = batch_tuple
        start = time.perf_counter()
        collection.insert([batch_ids, batch_titles, batch_contents, batch_categories, batch_vectors])
        end = time.perf_counter()
        return end-start, len(batch_vectors)

    print(f"Inserting into Milvus with metric {metric}.")
    start_total = time.perf_counter()
    batch_times = []
    total_inserted = 0

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        future_to_batch = {executor.submit(insert_batch, b): idx for idx, b in enumerate(batches)}
        for future in as_completed(future_to_batch):
            batch_time, batch_len = future.result()
            batch_times.append(batch_time)
            total_inserted += batch_len
            print(f"[{collection_name}] Batch inserted in {batch_time:.3f}s | "
                  f"Total inserted: {total_inserted}/{subset_size}")
            
    end_total = time.perf_counter()
    print(f"[{collection_name}] Total insertion time: {end_total-start_total:.3f}s")
    print(f"[{collection_name}] Average batch time: {np.mean(batch_times):.3f}s")

    print(f"Creating IVF_FLAT index on '{collection_name}' with metric {metric}...")
    collection.create_index(
        field_name="vector",
        index_params={"index_type": "IVF_FLAT", "metric_type": metric, "params": {"nlist": 128}}
    )
    collection.load()
    print(f"Collection '{collection_name}' loaded into memory.")

connections.disconnect("default")
print("Disconnected from Milvus.")