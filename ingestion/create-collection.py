import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import VectorParams, Distance

# Carrega variáveis de ambiente
load_dotenv()

# Configurações da collection
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "portiq")

# Inicializa o cliente Qdrant
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

# Verifica se a collection já existe e remove se necessário
collections = client.get_collections().collections
collection_names = [collection.name for collection in collections]
if COLLECTION_NAME in collection_names:
    print(f"Collection '{COLLECTION_NAME}' já existe. Removendo...")
    client.delete_collection(COLLECTION_NAME)

# Cria a collection com configuração para busca híbrida
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        # Vetor denso (semântico)
        "dense": VectorParams(size=384, distance=Distance.COSINE),
        # Vetor de interação tardia (ColBERT)
        "colbertv2.0": VectorParams(
            size=128,
            distance=Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM,
            ),
        ),
    },
    sparse_vectors_config={
        # Vetor esparso (BM25)
        "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF),
    },
)

print(f"Collection '{COLLECTION_NAME}' criada com sucesso!")

# Exibe informações da collection
collection_info = client.get_collection(COLLECTION_NAME)
print(f"Status: {collection_info.status}")
print(f"Vetores configurados: {list(collection_info.config.params.vectors.keys())}")
print(
    f"Vetores esparsos: {list(collection_info.config.params.sparse_vectors.keys() if collection_info.config.params.sparse_vectors else [])}"
)
print(f"Pontos: {collection_info.points_count}")
