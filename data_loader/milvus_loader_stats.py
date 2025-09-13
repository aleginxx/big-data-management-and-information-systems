"""
This script benchmarks data loading performance in Milvus DB by systematically inserting synthetic vector data under varying configurations. 
It is designed to evaluate how batch size, vector dimension, number of threads, and collection size affect insertion latency, throughput, and resource utilization.

Key Features:
- Dynamically generates random "news article"-like records with text fields, categorical labels, and float vectors of different dimensions.
- Creates Milvus collections with different distance metrics (L2, COSINE, IP).
- Tests multiple workloads by varying batch sizes and thread counts.
- Logs detailed statistics for each insertion batch, including:
  - Average and per-batch insertion times
  - CPU and memory usage during insertions
  - Total vectors inserted over time
- Outputs results into `data_loader_stats.csv`, enabling further analysis and 
  visualization with plotting scripts.

This script is complementary to `milvus_data_loader.py`, with extended dynamic options for benchmarking multiple experimental configurations. 
It is part of the overall benchmarking suite used to compare Milvus DB against other Weaviate DB.
"""

import time
import csv
import numpy as np
import psutil
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import os
import random
import string
from concurrent.futures import ThreadPoolExecutor, as_completed

# Connection parameters
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
BASE_COLLECTION_NAME = "NewsArticle"
CSV_FILE = "data_loader_stats.csv"

# Dynamic options
# This is the only thing that is different rom milvus_data_loader.py
BATCH_SIZES = [4, 8, 16, 32, 128]
METRICS = ["L2", "COSINE", "IP"]
VECTOR_DIMS = [32, 64, 128, 384]
VECTOR_SUBSETS = [1000, 5000]
NUM_THREADS = [1, 2, 4, 10]
CATEGORIES = ["sports", "politics", "tech", "finance"]  

# Log output to data_loader_stats.csv
with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "DB", "Collection", "Metric", "VectorDim", "NumVectors", "BatchID",
        "BatchSize", "ThreadCount", "BatchTime(s)", "TotalInsertedSoFar",
        "TotalTimeElapsed(s)", "CPU(%)", "Memory(MB)"
    ])

# Log CPU & memory usage
def get_resource_usage():
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    return cpu, mem

# Generate vectors for random data
def generate_vectors(num_vectors, dim):
    data = []
    for i in range(num_vectors):
        vec = np.random.random(dim).astype(np.float32).tolist()
        title = ''.join(random.choices(string.ascii_letters + string.digits, k=20))
        content = ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=100))
        category = random.choice(CATEGORIES)
        data.append({"id": i, "title": title, "content": content, "vector": vec, "category": category})
    return data

# Connect to milvus
connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
print(f"Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")

for dim in VECTOR_DIMS:
    for subset in VECTOR_SUBSETS:
        vectors_data = generate_vectors(subset, dim)
        num_vectors = len(vectors_data)
        print(f"\nGenerated {num_vectors} vectors of dimension {dim}")

        ids = [obj["id"] for obj in vectors_data]
        titles = [obj["title"] for obj in vectors_data]
        contents = [obj["content"] for obj in vectors_data]
        vectors = [obj["vector"] for obj in vectors_data]
        categories = [obj["category"] for obj in vectors_data]

        for metric in METRICS:
            collection_name = f"{BASE_COLLECTION_NAME}_{metric}_{dim}d_{subset}v"

            if utility.has_collection(collection_name):
                print(f"Dropping existing collection '{collection_name}'...")
                utility.drop_collection(collection_name)

            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=200),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=1500),
                FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=50),  
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim)
            ]
            schema = CollectionSchema(fields, description=f"News Articles with {metric} embeddings")
            collection = Collection(name=collection_name, schema=schema)
            print(f"Created collection '{collection_name}' with metric {metric}")

            for batch_size in BATCH_SIZES:
                for num_threads in NUM_THREADS:
                    print(f"Inserting with batch size {batch_size} using {num_threads} threads...")
                    total_inserted = 0
                    batch_times = []
                    start_total = time.perf_counter()

                    # Split data into batches
                    batches = [
                        (ids[i:i+batch_size],
                         titles[i:i+batch_size],
                         contents[i:i+batch_size],
                         categories[i:i+batch_size],
                         vectors[i:i+batch_size])
                        for i in range(0, num_vectors, batch_size)
                    ]

                    # Function to insert one batch
                    def insert_batch(batch_tuple):
                        batch_ids, batch_titles, batch_contents, batch_categories, batch_vectors = batch_tuple
                        start = time.perf_counter()
                        collection.insert([batch_ids, batch_titles, batch_contents, batch_categories, batch_vectors])
                        end = time.perf_counter()
                        return end - start, len(batch_vectors)

                    # Multi-threaded insertion
                    with ThreadPoolExecutor(max_workers=num_threads) as executor:
                        future_to_batch = {executor.submit(insert_batch, b): idx for idx, b in enumerate(batches)}
                        for future in as_completed(future_to_batch):
                            batch_time, batch_len = future.result()
                            batch_times.append(batch_time)
                            total_inserted += batch_len
                            cpu, mem = get_resource_usage()
                            batch_id = future_to_batch[future] + 1
                            print(f"[{collection_name}] Batch {batch_id} inserted in {batch_time:.3f}s | Threads: {num_threads} | CPU: {cpu:.1f}% | Mem: {mem:.1f}MB")
                            with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
                                writer = csv.writer(f)
                                writer.writerow([
                                    "Milvus", collection_name, metric, dim, num_vectors,
                                    batch_id, batch_size, num_threads, f"{batch_time:.3f}",
                                    total_inserted, f"{time.perf_counter() - start_total:.3f}",
                                    f"{cpu:.1f}", f"{mem:.1f}"
                                ])

                    end_total = time.perf_counter()
                    print(f"Total insertion time for {collection_name} with batch {batch_size} and {num_threads} threads: {end_total - start_total:.3f}s")
                    print(f"Average batch time: {np.mean(batch_times):.3f}s")

            print(f"Creating IVF_FLAT index on '{collection_name}' with metric {metric}...")
            collection.create_index(
                field_name="vector",
                index_params={"index_type": "IVF_FLAT", "metric_type": metric, "params": {"nlist": 128}}
            )
            collection.load()
            print(f"Collection '{collection_name}' loaded into memory.")

connections.disconnect("default")
print("Disconnected from Milvus.")