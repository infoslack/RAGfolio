from app.analyzers.base_analyzer import BaseAnalyzer
from app.models.agent import MarketSentiment
import logging

logger = logging.getLogger(__name__)


class SentimentAnalyzer(BaseAnalyzer[MarketSentiment]):
    """Analyzer for Stream 3: Market Sentiment Analysis (News)"""

    async def analyze(self, ticker: str) -> MarketSentiment:
        """Execute market sentiment analysis from news"""
        logger.info(f"Starting sentiment analysis for {ticker}")

        # Query recent news
        documents = self.document_retriever.query_news(
            query=f"{ticker} earnings revenue stock price",
            ticker=ticker,
            limit=10,
        )

        # Convert to formatted context
        content = self.document_retriever.news_to_context(documents)

        # Analyze sentiment
        return await self._call_openai_structured(
            prompt_name="sentiment_analysis",
            user_content=f"Recent news content for {ticker}:\n{content}",
            response_model=MarketSentiment,
        )
