"""
This script benchmarks data ingestion performance in a Weaviate database using synthetic vector data. 
It systematically varies vector dimensions, dataset sizes, batch sizes, and thread counts while testing multiple distance metrics (COSINE, L2, and DOT). 
For each configuration, the script measures per-batch insertion time, CPU utilization, and memory usage, and logs the results to a CSV file for further analysis.
The primary goal is to evaluate how different parameters and workload patterns affect Weaviate’s ingestion throughput and resource usage, enabling comparison with Milvus.
"""

import weaviate
import time
import csv
import psutil
import os
import numpy as np
import random
import string
from weaviate.classes.config import Configure, Property, DataType, VectorDistances

# Dynamic options (same as Milvus loader)
CSV_FILE = "weaviate_data_loader_stats.csv"
BASE_NAME = "NewsArticle"
BATCH_SIZES = [4, 8, 16, 32, 128]
METRICS = {
    "COSINE": VectorDistances.COSINE,
    "L2": VectorDistances.L2_SQUARED,
    "DOT": VectorDistances.DOT
}
VECTOR_DIMS = [32, 64, 128, 384]
VECTOR_SUBSETS = [1000, 5000]
NUM_THREADS = [1, 2, 4, 10]
CATEGORIES = ["sports","politics","tech","finance"]

# Log output to data_loader_stats.csv
with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "DB", "Collection", "Metric", "VectorDim", "NumVectors", "BatchID",
        "BatchSize", "BatchTime(s)", "TotalInsertedSoFar", "TotalTimeElapsed(s)",
        "CPU(%)", "Memory(MB)", "ThreadCount"
    ])

# Log CPU & memory usage
def get_resource_usage():
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.Process(os.getpid()).memory_info().rss / (1024*1024)
    return cpu, mem

# Generate vectors for random data
def generate_vectors(num_vectors, dim):
    data = []
    for i in range(num_vectors):
        vec = np.random.random(dim).astype(np.float32).tolist()
        title = ''.join(random.choices(string.ascii_letters + string.digits, k=20))
        content = ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=100))
        category = random.choice(CATEGORIES)
        data.append({"id": i, "title": title, "content": content, "category": category, "vector": vec})
    return data

# Connect to Weiaviate
client = weaviate.connect_to_local()
if client.is_ready():
    print("Connected to Weaviate")
else:
    raise RuntimeError("Failed to connect to Weaviate")

for dim in VECTOR_DIMS:
    for subset in VECTOR_SUBSETS:
        vectors_data = generate_vectors(subset, dim)
        num_vectors = len(vectors_data)
        print(f"\nGenerated {num_vectors} vectors of dimension {dim}")

        for metric_name, metric in METRICS.items():
            coll_name = f"{BASE_NAME}_{metric_name}_{dim}d_{subset}v"

            # Drop existing collection
            if client.collections.exists(coll_name):
                print(f"Dropping existing collection '{coll_name}'...")
                client.collections.delete(coll_name)

            # Create collection
            client.collections.create(
                name=coll_name,
                description=f"News Articles with {metric_name} embeddings",
                properties=[
                    Property(name="title", data_type=DataType.TEXT),
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="category", data_type=DataType.TEXT),
                ],
                vectorizer_config=Configure.Vectorizer.none(),
                vector_index_config=Configure.VectorIndex.hnsw(distance_metric=metric)
            )
            print(f"Created collection '{coll_name}' with metric {metric_name}")

            collection = client.collections.get(coll_name)

            for batch_size in BATCH_SIZES:
                print(f"Inserting with batch size {batch_size}...")

                total_inserted = 0
                batch_times = []
                start_total = time.perf_counter()

                for thread_count in NUM_THREADS:
                    for i in range(0, num_vectors, batch_size):
                        batch_vectors = vectors_data[i:i+batch_size]

                        start_batch = time.perf_counter()
                        with collection.batch.dynamic() as batch:
                            for obj in batch_vectors:
                                batch.add_object(
                                    properties={
                                        "title": obj["title"],
                                        "content": obj["content"],
                                        "category": obj["category"]
                                    },
                                    vector=obj["vector"]
                                )
                        end_batch = time.perf_counter()

                        batch_time = end_batch - start_batch
                        batch_times.append(batch_time)
                        total_inserted += len(batch_vectors)

                        cpu, mem = get_resource_usage()

                        print(f"[{coll_name}] Batch {i//batch_size + 1} inserted in {batch_time:.3f}s | "
                              f"CPU: {cpu:.1f}% | Mem: {mem:.1f}MB | Threads: {thread_count}")

                        # Log per batch
                        with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                "Weaviate", coll_name, metric_name, dim, num_vectors,
                                i//batch_size + 1, len(batch_vectors),
                                f"{batch_time:.3f}", total_inserted, f"{end_batch - start_total:.3f}",
                                f"{cpu:.1f}", f"{mem:.1f}", thread_count
                            ])

                print(f"Average batch time: {np.mean(batch_times):.3f}s for collection '{coll_name}'")
                print(f"Total vectors inserted: {total_inserted}")

client.close()
print("Disconnected from Weaviate. All collections created and populated successfully.")
