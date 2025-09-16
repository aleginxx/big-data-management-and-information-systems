import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# ===================== IEEE STYLE SETTINGS =====================
mpl.rcParams.update({
    "font.family": "Times New Roman",  # IEEE font
    "font.size": 8,                     # base font size
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1,
    "lines.markersize": 4,
    "savefig.dpi": 300
})
FIGSIZE = (3.5, 2.5)  # Single-column IEEE figure size in inches

# ===================== OUTPUT FOLDER ===========================
os.makedirs("plots/results", exist_ok=True)

# ===================== LOAD DATA ===============================
loader_df = pd.read_csv("data_loader/loader_stats.csv")
results_df = pd.read_csv("benchmarks/results_hf.csv")

# ===================== PREPROCESSING ===========================
results_df["Metric"] = results_df["Metric"].replace({"DOT": "IP"})

loader_df["throughput"] = loader_df["BatchSize"] / loader_df["BatchTime(s)"]
loader_summary = (
    loader_df.groupby(["DB", "BatchSize", "NumVectors"], as_index=False)
    .agg({
        "throughput": "mean",
        "BatchTime(s)": "mean",
        "CPU(%)": "mean",
        "Memory(MB)": "mean"
    })
    .rename(columns={"BatchTime(s)": "latency"})
)

# ===================== SAVE FUNCTION ===========================
def save_plot(fig, name):
    fig.savefig(f"plots/results/{name}.eps", bbox_inches="tight", format="eps")
    plt.close(fig)

# ===================== PLOTTING SETTINGS =======================
linestyles = ['-', '--', '-.', ':']
markers = ['o', 's', '^', 'd']

# ===================== DATA LOADING PLOTS ======================

# Insertion throughput vs Batch Size
fig, ax = plt.subplots(figsize=FIGSIZE)
for i, db in enumerate(loader_summary["DB"].unique()):
    subset = loader_summary[loader_summary["DB"] == db]
    ax.plot(subset["BatchSize"], subset["throughput"],
            linestyle=linestyles[i % 4], marker=markers[i % 4], color="k",
            label=db)
ax.set_xlabel("Batch Size")
ax.set_ylabel("Throughput (vectors/sec)")
ax.legend()
save_plot(fig, "insertion_throughput_vs_batch_size")

# Insertion latency vs Subset Size
fig, ax = plt.subplots(figsize=FIGSIZE)
for i, db in enumerate(loader_summary["DB"].unique()):
    subset = loader_summary[loader_summary["DB"] == db]
    ax.plot(subset["NumVectors"], subset["latency"],
            linestyle=linestyles[i % 4], marker=markers[i % 4], color="k",
            label=db)
ax.set_xlabel("Subset Size (# vectors)")
ax.set_ylabel("Latency (s)")
ax.legend()
save_plot(fig, "insertion_latency_vs_subset_size")


# Latency vs Batch Size
fig, ax = plt.subplots(figsize=FIGSIZE)
for i, db in enumerate(loader_summary["DB"].unique()):
    subset = loader_summary[loader_summary["DB"] == db]
    ax.plot(subset["BatchSize"], subset["latency"],
            linestyle=linestyles[i % 4], marker=markers[i % 4], color="k",
            label=db)
ax.set_xlabel("Batch Size")
ax.set_ylabel("Latency (s)")
ax.legend()
save_plot(fig, "dataloader_latency_vs_batch_size_side_by_side")

# CPU usage vs Subset Size
fig, ax = plt.subplots(figsize=FIGSIZE)
for i, db in enumerate(loader_summary["DB"].unique()):
    subset = loader_summary[loader_summary["DB"] == db]
    ax.plot(subset["NumVectors"], subset["CPU(%)"],
            linestyle=linestyles[i % 4], marker=markers[i % 4], color="k",
            label=db)
ax.set_xlabel("Subset Size (# vectors)")
ax.set_ylabel("CPU (%)")
ax.legend()
save_plot(fig, "cpu_usage_during_insertion")

# Memory usage vs Subset Size
fig, ax = plt.subplots(figsize=FIGSIZE)
for i, db in enumerate(loader_summary["DB"].unique()):
    subset = loader_summary[loader_summary["DB"] == db]
    ax.plot(subset["NumVectors"], subset["Memory(MB)"],
            linestyle=linestyles[i % 4], marker=markers[i % 4], color="k",
            label=db)
ax.set_xlabel("Subset Size (# vectors)")
ax.set_ylabel("Memory (MB)")
ax.legend()
save_plot(fig, "memory_usage_during_insertion")

# ===================== QUERY PLOTS ============================

width = 0.35
metrics = results_df["Metric"].unique()
x = range(len(metrics))

# Helper function for grayscale colors: first DB light gray
def get_color(i):
    return "0.7" if i == 0 else "0"

# Query Avg Latency by Metric
fig, ax = plt.subplots(figsize=FIGSIZE)
for i, db in enumerate(results_df["DB"].unique()):
    subset = results_df[results_df["DB"] == db]
    vals = subset.groupby("Metric")["AvgLatency(ms)"].mean().reindex(metrics)
    ax.bar([p + i*width for p in x], vals, width=width, color=get_color(i),
           hatch=['/', '\\', '-', '+'][i % 4], label=db)
ax.set_ylabel("Latency (ms)")
ax.set_xticks([p + width/2 for p in x])
ax.set_xticklabels(metrics, rotation=30)
ax.legend()
save_plot(fig, "query_avg_latency_by_metric")

# P95 Latency by Metric
fig, ax = plt.subplots(figsize=FIGSIZE)
for i, db in enumerate(results_df["DB"].unique()):
    subset = results_df[results_df["DB"] == db]
    vals = subset.groupby("Metric")["P95Latency(ms)"].mean().reindex(metrics)
    ax.bar([p + i*width for p in x], vals, width=width, color=get_color(i),
           hatch=['/', '\\', '-', '+'][i % 4], label=db)
ax.set_ylabel("Latency (ms)")
ax.set_xticks([p + width/2 for p in x])
ax.set_xticklabels(metrics, rotation=30)
ax.legend()
save_plot(fig, "query_p95_latency_by_metric")

# Throughput by Metric
fig, ax = plt.subplots(figsize=FIGSIZE)
for i, db in enumerate(results_df["DB"].unique()):
    subset = results_df[results_df["DB"] == db]
    vals = subset.groupby("Metric")["Throughput(q/s)"].mean().reindex(metrics)
    ax.bar([p + i*width for p in x], vals, width=width, color=get_color(i),
           hatch=['/', '\\', '-', '+'][i % 4], label=db)
ax.set_ylabel("Queries/sec")
ax.set_xticks([p + width/2 for p in x])
ax.set_xticklabels(metrics, rotation=30)
ax.legend()
save_plot(fig, "query_throughput_by_metric")

# With vs Without Filters (Avg Latency)
fig, ax = plt.subplots(figsize=FIGSIZE)
filter_latency = results_df.groupby(["DB", "FilterApplied"])["AvgLatency(ms)"].mean().unstack()
filter_latency.plot(kind="bar", ax=ax, color=["0.7", "0"], legend=True, hatch="/")
ax.set_ylabel("Avg Latency (ms)")
save_plot(fig, "query_latency_with_vs_without_filters")

# Latency vs Vector Dimensionality
dbs = results_df["DB"].unique()
fig, axes = plt.subplots(1, len(dbs), figsize=(FIGSIZE[0]*len(dbs), FIGSIZE[1]), sharey=True)
for i, db in enumerate(dbs):
    ax = axes[i]
    subset = results_df[results_df["DB"] == db]
    ax.plot(subset["VectorDim"], subset["AvgLatency(ms)"],
            linestyle=linestyles[0], marker=markers[0], color="k")
    ax.set_xlabel("Vector Dimension")
    if i == 0:
        ax.set_ylabel("Latency (ms)")
    ax.set_title(f"{db}")
    
plt.tight_layout()
save_plot(fig, "query_latency_vs_vector_dim_side_by_side")

# Latency vs Dataset Size
fig, axes = plt.subplots(1, len(dbs), figsize=(FIGSIZE[0]*len(dbs), FIGSIZE[1]), sharey=True)

for i, db in enumerate(dbs):
    ax = axes[i]
    subset = results_df[results_df["DB"] == db]
    ax.plot(subset["VectorSubset"], subset["AvgLatency(ms)"],
            linestyle=linestyles[0], marker=markers[0], color="k")
    ax.set_xlabel("Subset Size (# vectors)")
    if i == 0:
        ax.set_ylabel("Latency (ms)")
    ax.set_title(f"{db}")
    
plt.tight_layout()
save_plot(fig, "query_latency_vs_dataset_size_side_by_side")

print("All IEEE-style EPS plots generated in plots/results/")
