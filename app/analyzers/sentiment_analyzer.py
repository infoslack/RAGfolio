from app.analyzers.base_analyzer import BaseAnalyzer
from app.models.agent import MarketSentiment
import logging

logger = logging.getLogger(__name__)


class SentimentAnalyzer(BaseAnalyzer[MarketSentiment]):
    """Analyzer for Stream 3: Market Sentiment Analysis (News)"""

    async def analyze(self, ticker: str) -> MarketSentiment:
        """Execute market sentiment analysis from news"""
        logger.info(f"Starting sentiment analysis for {ticker}")

        # Get config for sentiment analysis
        config = self.config_loader.get_analysis_config("sentiment", "market_news")

        # Replace {ticker} in query
        query = config["query"].format(ticker=ticker)

        # Query recent news
        documents = self.document_retriever.query_news(
            query=query,
            ticker=ticker,
            limit=self.document_limit,  # Uses news_search_limit from settings
        )

        # Convert to formatted context
        content = self.document_retriever.news_to_context(documents)

        # Analyze sentiment
        return await self._call_openai_structured(
            prompt_name=config["prompt_name"],
            user_content=f"{config.get('section_name', 'News')} content for {ticker}:\n{content}",
            response_model=MarketSentiment,
        )
