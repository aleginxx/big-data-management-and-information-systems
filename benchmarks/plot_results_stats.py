import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

INPUT_FILE = "benchmarks/results_hf.csv"
OUTPUT_DIR = "./plots/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    df = pd.read_csv(INPUT_FILE, on_bad_lines="skip")
except Exception as e:
    print(f"Failed to load {INPUT_FILE}: {e}")
    exit(1)

print(f"Loaded {len(df)} rows from {INPUT_FILE}")

# Convert numeric columns
for col in ["AvgLatency(ms)", "P95Latency(ms)", "Throughput(q/s)"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Normalize metrics
if "Metric" in df.columns:
    df["Metric"] = df["Metric"].replace({
        "IP": "DOT",   
        "cosine": "COSINE",  
        "l2": "L2"
    })

palette = {
    "Milvus": "#2d223a",
    "Weaviate": "#c08fa0"
}

# Plot 1: Avg Latency vs Workload
plt.figure(figsize=(10,6))
sns.barplot(data=df, x="Workload", y="AvgLatency(ms)", hue="DB", palette=palette)
plt.title("Average Latency vs Workload")
plt.xlabel("Workload")
plt.ylabel("Avg Latency (ms)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "avg_latency_vs_workload.png"))
plt.close()

# Plot 2: P95 Latency vs Workload
plt.figure(figsize=(10,6))
sns.barplot(data=df, x="Workload", y="P95Latency(ms)", hue="DB", palette=palette)
plt.title("P95 Latency vs Workload")
plt.xlabel("Workload")
plt.ylabel("P95 Latency (ms)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "p95_latency_vs_workload.png"))
plt.close()

# Plot 3: Throughput vs Workload
plt.figure(figsize=(10,6))
sns.barplot(data=df, x="Workload", y="Throughput(q/s)", hue="DB", palette=palette)
plt.title("Throughput vs Workload")
plt.xlabel("Workload")
plt.ylabel("Throughput (q/s)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "throughput_vs_workload.png"))
plt.close()

# Per-Metric Breakdowns
if "Metric" in df.columns:
    unique_metrics = df["Metric"].dropna().unique()
    for metric_name in unique_metrics:
        df_metric = df[df["Metric"] == metric_name]

        # Avg Latency
        plt.figure(figsize=(10,6))
        sns.barplot(data=df_metric, x="Workload", y="AvgLatency(ms)", hue="DB", palette=palette)
        plt.title(f"Average Latency vs Workload — Metric: {metric_name}")
        plt.xlabel("Workload")
        plt.ylabel("Avg Latency (ms)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{metric_name}_avg_latency.png"))
        plt.close()

        # P95 Latency
        plt.figure(figsize=(10,6))
        sns.barplot(data=df_metric, x="Workload", y="P95Latency(ms)", hue="DB", palette=palette)
        plt.title(f"P95 Latency vs Workload — Metric: {metric_name}")
        plt.xlabel("Workload")
        plt.ylabel("P95 Latency (ms)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{metric_name}_p95_latency.png"))
        plt.close()

        # Throughput
        plt.figure(figsize=(10,6))
        sns.barplot(data=df_metric, x="Workload", y="Throughput(q/s)", hue="DB", palette=palette)
        plt.title(f"Throughput vs Workload — Metric: {metric_name}")
        plt.xlabel("Workload")
        plt.ylabel("Throughput (q/s)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{metric_name}_throughput.png"))
        plt.close()

print(f"All plots saved in '{OUTPUT_DIR}' folder.")
