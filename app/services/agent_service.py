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
    StreamResults,
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

        self.ticker_extractor = TickerExtractor(
            openai_api_key=settings.openai_api_key,
            model=self.model,
            prompt_manager=self.prompt_manager,
        )

        self.document_retriever = DocumentRetriever(embedder, retriever)

        # Initialize analyzers
        self.fundamental_analyzer = FundamentalAnalyzer(
            openai_client=self.client,
            document_retriever=self.document_retriever,
            prompt_manager=self.prompt_manager,
            model=self.model,
        )

        self.momentum_analyzer = MomentumAnalyzer(
            openai_client=self.client,
            document_retriever=self.document_retriever,
            prompt_manager=self.prompt_manager,
            model=self.model,
        )

        self.sentiment_analyzer = SentimentAnalyzer(
            openai_client=self.client,
            document_retriever=self.document_retriever,
            prompt_manager=self.prompt_manager,
            model=self.model,
        )

    async def analyze_investment(
        self,
        ticker: Optional[str] = None,
        message: Optional[str] = None,
        include_details: bool = False,
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
            (
                (stream1_result, stream1_details),
                (stream2_result, stream2_details),
                stream3_result,
            ) = await asyncio.gather(
                self.fundamental_analyzer.analyze(ticker),
                self.momentum_analyzer.analyze(ticker),
                self.sentiment_analyzer.analyze(ticker),
            )

            # Run final aggregation
            final_recommendation = await self._aggregate_analyses(
                ticker, stream1_result, stream2_result, stream3_result
            )

            execution_time = time.time() - start_time

            # Prepare detailed results if requested
            detailed_results = None
            if include_details:
                detailed_results = StreamResults(
                    stream1_details=stream1_details,
                    stream2_details=stream2_details,
                    stream3_details=stream3_result.model_dump(),
                )

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
                detailed_results=detailed_results,
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
