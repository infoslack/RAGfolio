from app.analyzers.base_analyzer import BaseAnalyzer
from app.models.agent import FundamentalAnalysis
import logging

logger = logging.getLogger(__name__)


class FundamentalAnalyzer(BaseAnalyzer[FundamentalAnalysis]):
    """Analyzer for Stream 1: Fundamental Analysis (10-K documents)"""

    async def analyze(self, ticker: str) -> FundamentalAnalysis:
        """Execute complete fundamental analysis"""
        logger.info(f"Starting fundamental analysis for {ticker}")

        # Get all queries from config
        config = self.config_loader.get_analysis_config("fundamental", "all_sections")

        # Retrieve documents from all relevant sections
        all_documents = []

        for section_query in config["queries"]:
            documents = self.document_retriever.query_documents(
                query=section_query,
                ticker=ticker,
                form_type="10-K",
                limit=self.document_limit,
            )
            all_documents.extend(documents)

        # Convert to context
        content = self.document_retriever.documents_to_context(all_documents)

        # Generate consolidated analysis directly
        return await self._call_openai_structured(
            prompt_name="fundamental_analysis",
            user_content=f"Analyze this 10-K content for {ticker}:\n\n{content}",
            response_model=FundamentalAnalysis,
        )
