from typing import List, Optional, Dict
from qdrant_client import QdrantClient
from app.models.embeddings import Document, QueryEmbeddings
from app.config.settings import Settings
from qdrant_client.http.exceptions import UnexpectedResponse
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)


class QdrantRetriever:
    def __init__(self, settings: Settings):
        # Basic client setup
        client_params = {"url": settings.qdrant_url, "timeout": settings.qdrant_timeout}

        # Add API key if provided
        if settings.qdrant_api_key:
            client_params["api_key"] = settings.qdrant_api_key

        self.client = QdrantClient(**client_params)
        self.collection_name = settings.collection_name
        self.prefetch_limit = settings.prefetch_limit

    def search_documents(
        self,
        embeddings: QueryEmbeddings,
        limit: int = 5,
        filters: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        try:
            # Qdrant filter
            query_filter = None
            if filters:
                must_conditions = []
                for key, value in filters.items():
                    must_conditions.append(
                        {"key": f"metadata.{key}", "match": {"value": value}}
                    )
                query_filter = {"must": must_conditions}

            # Search using all vector types
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                # First stage: Get candidates using dense and sparse search
                prefetch=[
                    {
                        "query": embeddings.dense,
                        "using": "dense",
                        "limit": self.prefetch_limit,
                    },
                    {
                        "query": embeddings.sparse_bm25.model_dump(),
                        "using": "sparse",
                        "limit": self.prefetch_limit,
                    },
                ],
                # Second stage: Rerank using late interaction
                query=embeddings.late,
                using="colbertv2.0",
                with_payload=True,
                limit=limit,
                query_filter=query_filter,
            )

            # Convert results to Document objects
            return [
                Document(
                    page_content=point.payload.get("text", ""),
                    metadata=point.payload.get("metadata", {}),
                )
                for point in search_result.points
            ]

        except UnexpectedResponse as e:
            # Handle Qdrant-specific errors
            logger.error(
                "Qdrant search failed",
                extra={"error": str(e), "collection": self.collection_name},
            )
            raise HTTPException(
                status_code=503, detail="Search service temporarily unavailable"
            )
        except Exception as e:
            # Handle any other errors
            logger.error(
                "Unexpected error during search",
                extra={"error": str(e), "collection": self.collection_name},
            )
            raise HTTPException(status_code=500, detail="Internal server error")
