from fastapi import APIRouter, Depends, HTTPException
from app.models.api import SearchRequest, SearchResponse
from app.services.retriever import QdrantRetriever
from app.services.embedder import QueryEmbedder
from app.config.settings import Settings

router = APIRouter(prefix="/search", tags=["search"])


def get_settings():
    return Settings()


def get_embedder(settings: Settings = Depends(get_settings)):
    return QueryEmbedder(
        dense_model_name=settings.dense_model_name,
        bm25_model_name=settings.bm25_model_name,
        late_interaction_model_name=settings.late_interaction_model_name,
    )


def get_retriever(settings: Settings = Depends(get_settings)):
    return QdrantRetriever(settings=settings)


@router.post("", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    embedder: QueryEmbedder = Depends(get_embedder),
    retriever: QdrantRetriever = Depends(get_retriever),
):
    try:
        # Generate embeddings for the query
        query_embeddings = embedder.embed_query(request.query)

        # Search documents using the generated embeddings
        results = retriever.search_documents(
            embeddings=query_embeddings, limit=request.limit, filters=request.filters
        )

        return SearchResponse(results=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
