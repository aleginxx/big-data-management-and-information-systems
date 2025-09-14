import os
import json
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import weaviate
from weaviate.classes.config import Configure, Property, DataType, VectorDistances

BASE_NAME = "NewsArticle"
BATCH_SIZES = [32]
METRICS = {
    "COSINE": VectorDistances.COSINE,
    "L2": VectorDistances.L2_SQUARED,
    "DOT": VectorDistances.DOT
}
NO_SAMPLES = 20000
HF_DATASET = "ag_news"
HF_SPLIT = "train"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE_EMBEDDING = 8
VEC_FILE = "weaviate_vector_data.json"

LABEL_MAP = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}

# Load/Generate embeddings
if os.path.exists(VEC_FILE):
    print(f"Loading vectors from {VEC_FILE}...")
    with open(VEC_FILE, "r", encoding="utf-8") as f:
        vectors_data = json.load(f)
else:
    print(f"No existing {VEC_FILE}, generating embeddings...")

    # Load dataset
    dataset = load_dataset(HF_DATASET, split=HF_SPLIT)
    dataset = dataset.select(range(NO_SAMPLES))
    texts = [row["text"] for row in dataset]
    categories = [LABEL_MAP[row["label"]] for row in dataset]

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    vectors_data = []
    print(f"Generating embeddings for {len(texts)} texts...")
    for i in tqdm(range(0, len(texts), BATCH_SIZE_EMBEDDING), desc="Embedding", unit="batch"):
        batch_texts = texts[i:i + BATCH_SIZE_EMBEDDING]
        batch_categories = categories[i:i + BATCH_SIZE_EMBEDDING]
        with torch.no_grad():
            inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
            outputs = model(**inputs)
            batch_vectors = outputs.last_hidden_state.mean(dim=1).cpu().numpy().astype(np.float32)

        for idx, vec in enumerate(batch_vectors):
            vectors_data.append({
                "id": i + idx,
                "title": batch_texts[idx][:50],
                "content": batch_texts[idx],
                "category": batch_categories[idx],
                "vector": vec.tolist()
            })

    # Save to JSON
    with open(VEC_FILE, "w", encoding="utf-8") as f:
        json.dump(vectors_data, f)
    print(f"Saved embeddings to {VEC_FILE}")

VECTOR_DIM = len(vectors_data[0]["vector"])
print(f"Ready to insert {len(vectors_data)} vectors of dimension {VECTOR_DIM}")

# weaviate connection
client = weaviate.connect_to_local()
if client.is_ready():
    print("Connected to Weaviate")
else:
    raise RuntimeError("Failed to connect to Weaviate")

# Create collections
for metric_name, metric in METRICS.items():
    coll_name = f"{BASE_NAME}_{metric_name}_{VECTOR_DIM}d_{len(vectors_data)}v"

    # Drop existing collection if it exists
    if client.collections.exists(coll_name):
        print(f"Dropping existing collection '{coll_name}'")
        client.collections.delete(coll_name)

    # Create collection
    client.collections.create(
        name=coll_name,
        description=f"News Articles with {metric_name} embeddings",
        properties=[
            Property(name="title", data_type=DataType.TEXT),
            Property(name="content", data_type=DataType.TEXT),
            Property(name="category", data_type=DataType.TEXT),  # Important for filtering
        ],
        vectorizer_config=Configure.Vectorizer.none(),
        vector_index_config=Configure.VectorIndex.hnsw(distance_metric=metric)
    )
    print(f"Created collection '{coll_name}' with metric {metric_name}")

    collection = client.collections.get(coll_name)

    # Insert data in batches
    for batch_size in BATCH_SIZES:
        print(f"Inserting with batch size {batch_size}...")
        for i in range(0, len(vectors_data), batch_size):
            batch_vectors = vectors_data[i:i + batch_size]
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
        print(f"Finished inserting into '{coll_name}' with batch size {batch_size}")

client.close()
print("Disconnected from Weaviate. All collections created and populated successfully.")
