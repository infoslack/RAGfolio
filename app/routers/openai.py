from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from app.models.api import OpenAIRequest, OpenAIResponse
from app.services.retriever import QdrantRetriever
from app.services.embedder import QueryEmbedder
from app.services.openai_service import OpenAIService
from app.config.settings import Settings

import logging
import json
from typing import AsyncGenerator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/openai", tags=["openai"])


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


def get_openai_service(settings: Settings = Depends(get_settings)):
    return OpenAIService(settings=settings)


@router.post("", response_model=OpenAIResponse)
async def generate_openai_response(
    request: OpenAIRequest,
    embedder: QueryEmbedder = Depends(get_embedder),
    retriever: QdrantRetriever = Depends(get_retriever),
    openai_service: OpenAIService = Depends(get_openai_service),
):
    try:
        query_embeddings = embedder.embed_query(request.query)

        context_documents = retriever.search_documents(
            embeddings=query_embeddings, limit=request.limit, filters=request.filters
        )

        if not context_documents:
            logger.warning(
                "No relevant documents found for query", extra={"query": request.query}
            )

        answer = openai_service.generate_response(
            query=request.query,
            context_documents=context_documents,
            model=request.model,
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens,
        )

        return OpenAIResponse(answer=answer, source_documents=context_documents)

    except Exception as e:
        logger.error(
            "OpenAI generation failed", extra={"error": str(e), "query": request.query}
        )
        raise HTTPException(
            status_code=500, detail=f"OpenAI generation failed: {str(e)}"
        )


# Stream API
@router.post("/stream")
async def generate_openai_stream_response(
    request: OpenAIRequest,
    embedder: QueryEmbedder = Depends(get_embedder),
    retriever: QdrantRetriever = Depends(get_retriever),
    openai_service: OpenAIService = Depends(get_openai_service),
):
    try:
        query_embeddings = embedder.embed_query(request.query)

        context_documents = retriever.search_documents(
            embeddings=query_embeddings, limit=request.limit
        )

        if not context_documents:
            logger.warning(
                "No relevant documents found for query", extra={"query": request.query}
            )

        async def event_generator() -> AsyncGenerator[str, None]:
            try:
                # First, send the source documents
                yield f"data: {json.dumps({'type': 'source_documents', 'documents': [doc.model_dump() for doc in context_documents]})}\n\n"

                # Then stream the response
                async for chunk in openai_service.generate_stream_response(
                    query=request.query,
                    context_documents=context_documents,
                    model=request.model,
                    temperature=request.temperature,
                    max_output_tokens=request.max_output_tokens,
                ):
                    yield f"data: {chunk}\n\n"

                # Send completion event
                yield f"data: {json.dumps({'type': 'stream_completed'})}\n\n"

            except Exception as e:
                logger.error(
                    "Stream generation failed",
                    extra={"error": str(e), "query": request.query},
                )
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable Nginx buffering
            },
        )

    except Exception as e:
        logger.error(
            "OpenAI stream generation failed",
            extra={"error": str(e), "query": request.query},
        )
        raise HTTPException(
            status_code=500, detail=f"OpenAI stream generation failed: {str(e)}"
        )
