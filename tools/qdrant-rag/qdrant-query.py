from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Document

# generate query embedding
query_text = "vegetarian dishes"

# search for similar menu items
results = client.query_points(
    collection_name="items",
    query=Document(text=query_text, model="sentence-transformers/all-MiniLM-L6-v2"),
    with_payload=True,
    limit=5
)

# print results
for result in results.points:
    print(f"Item: {result.payload.get('item_name', 'N/A')}")
    print(f"Score: {result.score}")
    print(f"Description: {result.payload['description'][:150]}...")
    print(f"Price: {result.payload.get('price', 'N/A')}")
    print("---")