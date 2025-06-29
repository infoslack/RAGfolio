from fastapi import APIRouter, Depends, HTTPException
from app.models.agent import AgentRequest, AgentResponse
from app.services.agent_service import AgentService
from app.services.retriever import QdrantRetriever
from app.services.embedder import QueryEmbedder
from app.config.settings import Settings

import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])


def get_settings():
    return Settings()


def get_embedder(settings: Settings = Depends(get_settings)):
    return QueryEmbedder(
        dense_model_name=settings.dense_model_name,
        bm25_model_name=settings.bm25_model_name,
        late_interaction_model_name=settings.late_interaction_model_name,
        cache_dir=settings.embedder_cache_dir,
        local_files_only=settings.embedder_local_files_only,
    )


def get_retriever(settings: Settings = Depends(get_settings)):
    return QdrantRetriever(settings=settings)


def get_agent_service(
    embedder: QueryEmbedder = Depends(get_embedder),
    retriever: QdrantRetriever = Depends(get_retriever),
    settings: Settings = Depends(get_settings),
):
    return AgentService(embedder=embedder, retriever=retriever, settings=settings)


@router.post("", response_model=AgentResponse)
async def analyze_investment(
    request: AgentRequest,
    agent_service: AgentService = Depends(get_agent_service),
):
    """
    Run complete investment analysis with 3 streams:
    - Stream 1: Fundamental Analysis (10-K)
    - Stream 2: Momentum Analysis (10-Q)
    - Stream 3: Market Sentiment (News)
    - Aggregation: Final recommendation
    """
    try:
        if not request.ticker and not request.message:
            raise HTTPException(
                status_code=400, detail="Either ticker or message must be provided"
            )

        # Service handles all business logic
        result = await agent_service.analyze_investment(
            ticker=request.ticker,
            message=request.message,
        )

        logger.info(
            f"Completed investment analysis for {result.ticker} in {result.execution_time:.2f}s"
        )
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Investment analysis failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Investment analysis failed: {str(e)}"
        )
