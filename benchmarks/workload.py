import json
import random
import numpy as np

with open("./milvus_vector_data.json", "r", encoding="utf-8") as f:
	vectors_data = json.load(f)

all_vectors = [np.array(obj["vector"], dtype=np.float32) for obj in vectors_data]

def get_query_vectors(size: int):
	"""Pick 'size' query vectors from dataset."""
	return random.sample(all_vectors, size)

WORKLOADS = {
	"small": 100,
	"medium": 1000,
	"full": len(all_vectors)
}
