import os
import time
import statistics
import csv
import numpy as np
from tqdm import tqdm
import weaviate
from weaviate.collections.classes.filters import Filter
import json
import random

# ---------------- Parameters ----------------
CSV_FILE = "results_weaviate_hf.csv"
QUERY_LIMIT = 5
FILTER_VALUES = ["World", "Sports", "Business", "Sci/Tech"]  # actual labels
WORKLOAD_TYPES = {
    "small": 100,
    "medium": 1000,
    "full": None  # will be replaced with full dataset length
}
VEC_FILE = "./weaviate_vector_data.json"

# ---------------- Load query vectors ----------------
if not os.path.exists(VEC_FILE):
    raise FileNotFoundError(f"{VEC_FILE} not found! Run the data loader first.")

with open(VEC_FILE, "r", encoding="utf-8") as f:
    vectors_data = json.load(f)

WORKLOAD_TYPES["full"] = len(vectors_data)

# Prepare vectors and category mapping
all_vectors = [np.array(obj["vector"], dtype=np.float32) for obj in vectors_data]
all_categories = [obj.get("category", "-") for obj in vectors_data]

def get_query_vectors(size: int):
    sampled_indices = random.sample(range(len(all_vectors)), size)
    return [all_vectors[i] for i in sampled_indices]

def get_query_categories(size: int):
    sampled_indices = random.sample(range(len(all_vectors)), size)
    return [all_categories[i] for i in sampled_indices]

# ---------------- Connect ----------------
client = weaviate.connect_to_local(
    host="localhost",
    port=8080,
    grpc_port=50051
)
print("Connected to Weaviate")

# ---------------- Setup CSV ----------------
file_exists = os.path.isfile(CSV_FILE)
with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow([
            "DB", "Collection", "Metric", "VectorDim", "VectorSubset", "Workload",
            "QueryLimit", "FilterApplied", "FilterValue",
            "AvgLatency(ms)", "P95Latency(ms)", "Throughput(q/s)"
        ])

# ---------------- Identify collections ----------------
all_collections = client.collections.list_all()
weaviate_collections = [c for c in all_collections if c.startswith("NewsArticle_")]

# ---------------- Benchmark loop ----------------
for coll_name in weaviate_collections:
    # Extract metric, dimension, and subset
    try:
        parts = coll_name.split("_")
        metric = parts[1]
        vector_dim = int(parts[2][:-1])
        vector_subset = int(parts[3][:-1])
    except Exception as e:
        print(f"Skipping collection {coll_name}, cannot parse info: {e}")
        continue

    print(f"\n=== Benchmarking collection: {coll_name} | Metric: {metric} | Dim: {vector_dim} | Subset: {vector_subset} ===")

    try:
        collection = client.collections.get(coll_name)
    except Exception as e:
        print(f"Skipping {coll_name}, failed to load: {e}")
        continue

    for workload, workload_size in WORKLOAD_TYPES.items():
        print(f"Running {workload} workload")
        query_vectors = get_query_vectors(workload_size)

        # ---------------- Unfiltered queries ----------------
        latencies = []
        start_all = time.perf_counter()
        for vec in tqdm(query_vectors, desc=f"{coll_name} ({metric}) no filter", unit="query", mininterval=1.0):
            start = time.perf_counter()
            _ = collection.query.near_vector(
                near_vector=vec.tolist(),
                limit=QUERY_LIMIT,
                return_properties=["title", "category"]
            )
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
        end_all = time.perf_counter()

        avg_latency = statistics.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        throughput = len(query_vectors) / (end_all - start_all)

        print(f"{metric} (no filter) -> avg={avg_latency:.2f} ms | p95={p95_latency:.2f} ms | throughput={throughput:.2f} q/s")

        with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Weaviate", coll_name, metric, vector_dim, vector_subset,
                workload, QUERY_LIMIT, "No", "-",
                f"{avg_latency:.2f}", f"{p95_latency:.2f}", f"{throughput:.2f}"
            ])

        # ---------------- Filtered queries ----------------
        for filter_value in FILTER_VALUES:
            latencies = []
            start_all = time.perf_counter()
            for vec in tqdm(query_vectors, desc=f"{coll_name} ({metric}) filter={filter_value}", unit="query", mininterval=1.0):
                start = time.perf_counter()
                _ = collection.query.near_vector(
                    near_vector=vec.tolist(),
                    limit=QUERY_LIMIT,
                    return_properties=["title", "category"],
                    filters=Filter.by_property("category").equal(filter_value)
                )
                end = time.perf_counter()
                latencies.append((end - start) * 1000)
            end_all = time.perf_counter()

            avg_latency = statistics.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            throughput = len(query_vectors) / (end_all - start_all)

            print(f"{metric} (filter={filter_value}) -> avg={avg_latency:.2f} ms | p95={p95_latency:.2f} ms | throughput={throughput:.2f} q/s")

            with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Weaviate", coll_name, metric, vector_dim, vector_subset,
                    workload, QUERY_LIMIT, "Yes", filter_value,
                    f"{avg_latency:.2f}", f"{p95_latency:.2f}", f"{throughput:.2f}"
                ])

# ---------------- Disconnect ----------------
client.close()
print("\nBenchmarking complete. Results saved to ", CSV_FILE)
