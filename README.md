# Big Data Management and Information Systems

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Docker](https://img.shields.io/badge/Docker-Desktop-blue?logo=docker)
![Git](https://img.shields.io/badge/Git-Bash-lightgrey?logo=git)

This repository contains a benchmarking project comparing **Milvus** and **Weaviate**, two leading vector databases.  
The main goal is to evaluate how each system performs when **loading large-scale vector embeddings** generated from real datasets, as well as under artificial stress tests with random input data.

---

## 🚀 Project Overview

The project is structured around **two types of experiments**:

1. **Real-world data ingestion**
   - Load the [AG News dataset](https://huggingface.co/datasets/ag_news) from Hugging Face.
   - Generate semantic embeddings using a pretrained **SentenceTransformer** model.
   - Insert these embeddings into **Milvus** and **Weaviate** collections using different similarity metrics (Cosine, L2, Inner Product).
   - Build indexes and prepare collections for downstream querying and benchmarking.

2. **Synthetic benchmarking (`*_loader_stats` scripts)**
   - Instead of Hugging Face data, random vectors are generated.
   - Used **only** to measure **raw ingestion throughput, batch performance, CPU and memory utilization** of each database.
   - These are not designed for queries, but purely for evaluating insertion capacities under controlled settings.

---

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/aleginxx/big-data-management-and-information-systems.git
cd big-data-management-and-information-systems
```

### 2. Install dependencies
We recommend using a virtual environment (Conda or venv).

```bash
pip install -r requirements.txt
```

### 3. Run Milvus and Weaviate in Docker

#### Milvus
Download file docker-compose.yaml from Milvus github repo: https://github.com/milvus-io/milvus/blob/master/deployments/docker/gpu/standalone/docker-compose.yml
```bash
docker compose up -d
```

#### Weaviate
```bash
docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.32.8
```

---

## Running the Scripts

### Load real embeddings
```bash
# Insert AG News embeddings into Milvus
python data_loader/milvus_data_loader.py

# Insert AG News embeddings into Weaviate
python data_loader/weaviate_data_loader.py
```

### Benchmark insertion throughput with synthetic data
```bash
# Milvus
python data_loader/milvus_loader_stats.py

# Weaviate
python data_loader/weaviate_loader_stats.py
```

### Drop collections
```bash
# Drop collections from Milvus
python data_loader/drop_collections.py --db milvus

# Drop collections from Weaviate
python data_loader/drop_collections.py --db weaviate
```

---

## 📊 Results

- `*_loader_stats.py` scripts produce benchmarking results with performance metrics (insertion speed, CPU, memory).
- Real data loaders (`milvus_data_loader.py`, `weaviate_data_loader.py`) prepare the vector DBs for similarity search experiments.

---