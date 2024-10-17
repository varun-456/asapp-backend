import os
from qdrant_client import QdrantClient, models

# Ensure Qdrant API key is available
qdrant_api_key = os.getenv("QDRANT_API_KEY")
if not qdrant_api_key:
    raise ValueError("Qdrant API key not set. Please set it using 'QDRANT_API_KEY' environment variable.")

# Qdrant setup with REST fallback (prefer_grpc=False)
qdrant_url = "https://91192707-948f-4785-b4f6-348c269428ab.europe-west3-0.gcp.cloud.qdrant.io"
qdrant_client = QdrantClient(qdrant_url, prefer_grpc=False, api_key=qdrant_api_key)

# Check if the collection exists and recreate if necessary
collection_name = "research_papers"
try:
    qdrant_client.recreate_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(size=4096, distance=models.Distance.COSINE)  # Set size to 4096
)
    print(f"Collection {collection_name} created successfully.")
except Exception as e:
    print(f"Error creating collection: {str(e)}")
