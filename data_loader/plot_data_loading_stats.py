import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import os

# ---------------- Load data ----------------
df = pd.read_csv("data_loader/milvus_data_loader_stats.csv")
df_batches = df[df["BatchID"] != "TOTAL"]

# Create output folder if it doesn't exist
output_dir = "./plots"
os.makedirs(output_dir, exist_ok=True)

# ---------------- Plot 1: Batch Time vs Batch Size ----------------
plt.figure(figsize=(10,6))
sns.lineplot(
    data=df_batches,
    x="BatchSize",
    y="BatchTime(s)",
    hue="ThreadCount",
    style="VectorDim",
    markers=True
)
plt.title("Average Batch Insertion Time vs Batch Size")
plt.xlabel("Batch Size")
plt.ylabel("Batch Time (s)")
plt.xscale("log")
plt.yscale("log")
plt.legend(title="ThreadCount / VectorDim")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "batch_time_vs_batch_size.png"))
plt.close()

# ---------------- Plot 2: CPU Usage vs Batch ----------------
plt.figure(figsize=(10,6))
sns.lineplot(data=df_batches, x="BatchID", y="CPU(%)", hue="ThreadCount")
plt.title("CPU Usage During Batch Inserts")
plt.xlabel("Batch ID")
plt.ylabel("CPU (%)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cpu_usage_vs_batch.png"))
plt.close()

# ---------------- Plot 3: Memory Usage vs Batch ----------------
plt.figure(figsize=(10,6))
sns.lineplot(data=df_batches, x="BatchID", y="Memory(MB)", hue="ThreadCount")
plt.title("Memory Usage During Batch Inserts")
plt.xlabel("Batch ID")
plt.ylabel("Memory (MB)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "memory_usage_vs_batch.png"))
plt.close()

# ---------------- Plot 4: Batch Time vs Vector Dimension ----------------
plt.figure(figsize=(10,6))
sns.barplot(data=df_batches, x="VectorDim", y="BatchTime(s)", hue="ThreadCount")
plt.title("Batch Insertion Time vs Vector Dimension")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "batch_time_vs_vector_dim.png"))
plt.close()

# ---------------- Plot 5: Distribution of Batch Times ----------------
plt.figure(figsize=(10,6))
sns.violinplot(
    data=df_batches,
    x="VectorDim",
    y="BatchTime(s)",
    hue="ThreadCount",
    split=True
)
plt.title("Distribution of Batch Times by Vector Dimension")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "batch_time_distribution.png"))
plt.close()

# ---------------- Plot 6: 3D Surface - BatchSize x ThreadCount vs Avg Batch Time ----------------
subset = df_batches.groupby(["BatchSize", "ThreadCount"])["BatchTime(s)"].mean().reset_index()
X = subset["BatchSize"].values
Y = subset["ThreadCount"].values
Z = subset["BatchTime(s)"].values

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_trisurf(X, Y, Z, cmap=cm.viridis, edgecolor='k', linewidth=0.5)
ax.set_xlabel("Batch Size")
ax.set_ylabel("Thread Count")
ax.set_zlabel("Avg Batch Time (s)")
ax.set_title("3D Surface: BatchTime vs BatchSize vs ThreadCount")
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "3d_surface_batch_time.png"))
plt.close()

print(f"All plots saved in '{output_dir}' folder.")
