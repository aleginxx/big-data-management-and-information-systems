import argparse
from pymilvus import connections, utility
import weaviate

MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

WEAVIATE_HOST = "localhost"
WEAVIATE_PORT = 8080
WEAVIATE_GRPC_PORT = 50051


def drop_milvus():
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    print("Connected to Milvus")

    collections = utility.list_collections()
    print("Found Milvus collections:", collections)

    for name in collections:
        print(f"Dropping Milvus collection: {name}")
        utility.drop_collection(name)

    print("All Milvus collections dropped.")
    connections.disconnect("default")


def drop_weaviate():
    client = weaviate.connect_to_local(
        host=WEAVIATE_HOST,
        port=WEAVIATE_PORT,
        grpc_port=WEAVIATE_GRPC_PORT
    )

    if not client.is_ready():
        raise RuntimeError("Failed to connect to Weaviate")
    print("Connected to Weaviate")

    collections = client.collections.list_all()
    print("Found Weaviate collections:", collections)

    for name in collections:
        print(f"Dropping Weaviate collection: {name}")
        client.collections.delete(name)

    print("All Weaviate collections dropped.")
    client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drop all collections from Milvus or Weaviate")
    parser.add_argument("--db", choices=["milvus", "weaviate"], required=True,
                        help="Which database to drop collections from")
    args = parser.parse_args()

    if args.db == "milvus":
        drop_milvus()
    elif args.db == "weaviate":
        drop_weaviate()