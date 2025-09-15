import os
import time
import statistics
import csv
import numpy as np
from tqdm import tqdm
from pymilvus import connections, Collection, utility
from workload import get_query_vectors, WORKLOADS

CSV_FILE = "results_hf.csv"
INDEX_TYPE = "IVF_FLAT"
NLIST = 128
QUERY_LIMIT = 5
FILTER_VALUES = ["World", "Sports", "Business", "Sci/Tech"]
WORKLOAD_TYPES = ["small", "medium", "full"]

# ---------------- Connect ----------------
connections.connect("default", host="localhost", port="19530")
print("Connected to Milvus")

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
all_collections = utility.list_collections()
milvus_collections = [c for c in all_collections if c.startswith("NewsArticle_")]

# ---------------- Benchmark loop ----------------
for coll_name in milvus_collections:
    # Extract metric, dimension, and subset from collection name
    try:
        parts = coll_name.split("_")
        metric = parts[1]
        vector_dim = int(parts[2][:-1])  # e.g., '64d' -> 64
        vector_subset = int(parts[3][:-1])  # e.g., '5000v' -> 5000
    except Exception as e:
        print(f"Skipping collection {coll_name}, cannot parse info: {e}")
        continue

    print(f"\n=== Benchmarking collection: {coll_name} | Metric: {metric} | Dim: {vector_dim} | Subset: {vector_subset} ===")

    try:
        collection = Collection(coll_name)
        collection.load()
    except Exception as e:
        print(f"Skipping {coll_name}, failed to load: {e}")
        continue

    for workload in WORKLOAD_TYPES:
        print(f"Running {workload} workload")
        # Adjust query vectors to match collection dimension
        query_vectors_raw = get_query_vectors(WORKLOADS[workload])
        query_vectors = [vec[:vector_dim] for vec in query_vectors_raw]  # slice to match dimension

        # ---------------- Unfiltered queries ----------------
        latencies = []
        start_all = time.perf_counter()
        for vec in tqdm(query_vectors, desc=f"{coll_name} ({metric}) no filter", unit="query", mininterval=1.0):
            start = time.perf_counter()
            _ = collection.search(
                data=[vec.tolist()],
                anns_field="vector",
                param={"metric_type": metric, "params": {"nprobe": 10}},
                limit=QUERY_LIMIT,
                output_fields=["title"]
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
                "Milvus", coll_name, metric, vector_dim, vector_subset,
                workload, QUERY_LIMIT, "No", "-", f"{avg_latency:.2f}", f"{p95_latency:.2f}", f"{throughput:.2f}"
            ])

        # ---------------- Filtered queries ----------------
        for filter_value in FILTER_VALUES:
            expr = f'category == "{filter_value}"'
            latencies = []
            start_all = time.perf_counter()
            for vec in tqdm(query_vectors, desc=f"{coll_name} ({metric}) filter={filter_value}", unit="query", mininterval=1.0):
                start = time.perf_counter()
                _ = collection.search(
                    data=[vec.tolist()],
                    anns_field="vector",
                    param={"metric_type": metric, "params": {"nprobe": 10}},
                    limit=QUERY_LIMIT,
                    output_fields=["title", "category"],
                    expr=expr
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
                    "Milvus", coll_name, metric, vector_dim, vector_subset,
                    workload, QUERY_LIMIT, "Yes", filter_value,
                    f"{avg_latency:.2f}", f"{p95_latency:.2f}", f"{throughput:.2f}"
                ])

# ---------------- Disconnect ----------------
connections.disconnect("default")
print("\nBenchmarking complete. Results saved to ", CSV_FILE)
