from fastembed import TextEmbedding
from fastembed.sparse.bm25 import Bm25
from app.models.embeddings import QueryEmbeddings, SparseVector
from fastembed.late_interaction import LateInteractionTextEmbedding
import os


class QueryEmbedder:
    def __init__(
        self,
        dense_model_name: str,
        bm25_model_name: str,
        late_interaction_model_name: str,
    ):
        # Disable tokenizer parallelism to prevent deadlocks
        if "TOKENIZERS_PARALLELISM" not in os.environ:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Initialize the three embedding models
        self.dense_embedding_model = TextEmbedding(
            dense_model_name, cache_dir="/tmp/vector", local_files_only=True
        )  # For dense vectors
        self.bm25_embedding_model = Bm25(
            bm25_model_name, cache_dir="/tmp/vector", local_files_only=True
        )  # For sparse vectors
        self.late_interaction_model = (
            LateInteractionTextEmbedding(  # For token-level vectors
                late_interaction_model_name,
                cache_dir="/tmp/vector",
                local_files_only=True,
            )
        )

    def embed_query(self, query: str) -> QueryEmbeddings:
        # Get dense embeddings (e.g., [0.1, 0.2, ...])
        dense_vector = next(self.dense_embedding_model.embed(query)).tolist()

        # Get sparse BM25 embeddings (keyword weights)
        sparse_vector = next(self.bm25_embedding_model.embed(query))

        # Get late interaction embeddings (token-level vectors)
        late_vector = next(self.late_interaction_model.embed(query)).tolist()

        # Combine all embeddings into a single object
        return QueryEmbeddings(
            dense=dense_vector,
            sparse_bm25=SparseVector(**sparse_vector.as_object()),
            late=late_vector,
        )
