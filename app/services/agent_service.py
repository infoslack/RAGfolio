import asyncio
import time
from typing import Optional
from pathlib import Path

from openai import AsyncOpenAI

from app.config.settings import Settings
from app.services.embedder import QueryEmbedder
from app.services.retriever import QdrantRetriever
from app.services.ticker_extractor import TickerExtractor
from app.services.document_retriever import DocumentRetriever
from app.services.prompt_manager import PromptManager
from app.services.config_loader import ConfigLoader

from app.analyzers import (
    FundamentalAnalyzer,
    MomentumAnalyzer,
    SentimentAnalyzer,
)

from app.models.agent import (
    FundamentalAnalysis,
    MomentumAnalysis,
    MarketSentiment,
    FinalRecommendation,
    AgentResponse,
)

import logging

logger = logging.getLogger(__name__)


class AgentService:
    """Investment Analysis Agent Service - Orchestrates all analysis streams"""

    def __init__(
        self, embedder: QueryEmbedder, retriever: QdrantRetriever, settings: Settings
    ):
        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model

        # Initialize services
        prompts_dir = Path(__file__).parent.parent / "prompts"
        self.prompt_manager = PromptManager(prompts_dir)

        # Initialize config loader
        self.config_loader = ConfigLoader(
            queries_path=settings.queries_config_path,
            ticker_mappings_path=settings.ticker_mappings_path,
        )

        self.ticker_extractor = TickerExtractor(
            openai_api_key=settings.openai_api_key,
            model=self.model,
            prompt_manager=self.prompt_manager,
            config_loader=self.config_loader,
            temperature=settings.ticker_extraction_temperature,
            max_tokens=settings.ticker_extraction_max_tokens,
        )

        self.document_retriever = DocumentRetriever(embedder, retriever)

        # Initialize analyzers with config
        self.fundamental_analyzer = FundamentalAnalyzer(
            openai_client=self.client,
            document_retriever=self.document_retriever,
            prompt_manager=self.prompt_manager,
            config_loader=self.config_loader,
            model=self.model,
            temperature=settings.analysis_temperature,
            document_limit=settings.document_search_limit,
        )

        self.momentum_analyzer = MomentumAnalyzer(
            openai_client=self.client,
            document_retriever=self.document_retriever,
            prompt_manager=self.prompt_manager,
            config_loader=self.config_loader,
            model=self.model,
            temperature=settings.analysis_temperature,
            document_limit=settings.document_search_limit,
        )

        self.sentiment_analyzer = SentimentAnalyzer(
            openai_client=self.client,
            document_retriever=self.document_retriever,
            prompt_manager=self.prompt_manager,
            config_loader=self.config_loader,
            model=self.model,
            temperature=settings.analysis_temperature,
            document_limit=settings.news_search_limit,
        )

    async def analyze_investment(
        self,
        ticker: Optional[str] = None,
        message: Optional[str] = None,
    ) -> AgentResponse:
        """Run complete investment analysis with all 3 streams + aggregation"""
        start_time = time.time()

        try:
            # Extract ticker if not provided directly
            if not ticker and message:
                ticker = await self.ticker_extractor.extract_ticker(message)

            if not ticker:
                raise ValueError("Could not determine ticker symbol from input")

            logger.info(f"Starting complete investment analysis for {ticker}")

            # Execute all 3 streams in parallel
            stream1_result, stream2_result, stream3_result = await asyncio.gather(
                self.fundamental_analyzer.analyze(ticker),
                self.momentum_analyzer.analyze(ticker),
                self.sentiment_analyzer.analyze(ticker),
            )

            # Run final aggregation
            final_recommendation = await self._aggregate_analyses(
                ticker, stream1_result, stream2_result, stream3_result
            )

            execution_time = time.time() - start_time

            logger.info(
                f"Completed investment analysis for {ticker} in {execution_time:.2f}s"
            )

            return AgentResponse(
                ticker=ticker,
                execution_time=execution_time,
                fundamental_analysis=stream1_result,
                momentum_analysis=stream2_result,
                market_sentiment=stream3_result,
                final_recommendation=final_recommendation,
            )

        except Exception as e:
            logger.error(f"Investment analysis failed for {ticker}: {str(e)}")
            raise Exception(f"Investment analysis failed: {str(e)}")

    async def _aggregate_analyses(
        self,
        ticker: str,
        fundamental_analysis: FundamentalAnalysis,
        momentum_analysis: MomentumAnalysis,
        market_sentiment: MarketSentiment,
    ) -> FinalRecommendation:
        """Aggregate all three streams into final recommendation"""
        logger.info(f"Starting final aggregation for {ticker}")

        aggregation_input = f"""
        STREAM 1 - FUNDAMENTAL ANALYSIS for {ticker}:
        {fundamental_analysis.model_dump()}
        
        STREAM 2 - MOMENTUM ANALYSIS for {ticker}:
        {momentum_analysis.model_dump()}
        
        STREAM 3 - MARKET SENTIMENT for {ticker}:
        {market_sentiment.model_dump()}
        """

        try:
            system_prompt = self.prompt_manager.get_prompt("final_recommendation")

            completion = await self.client.beta.chat.completions.parse(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": aggregation_input},
                ],
                response_format=FinalRecommendation,
            )

            return completion.choices[0].message.parsed

        except Exception as e:
            logger.error(f"Final aggregation failed: {str(e)}")
            raise
