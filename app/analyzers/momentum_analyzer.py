from app.analyzers.base_analyzer import BaseAnalyzer
from app.models.agent import MomentumAnalysis
import logging

logger = logging.getLogger(__name__)


class MomentumAnalyzer(BaseAnalyzer[MomentumAnalysis]):
    """Analyzer for Stream 2: Momentum Analysis (10-Q documents)"""

    async def analyze(self, ticker: str) -> MomentumAnalysis:
        """Execute complete momentum analysis"""
        logger.info(f"Starting momentum analysis for {ticker}")

        # Get all queries from config
        config = self.config_loader.get_analysis_config("momentum", "all_sections")

        # Retrieve documents from all relevant sections
        all_documents = []

        for section_query in config["queries"]:
            documents = self.document_retriever.query_documents(
                query=section_query,
                ticker=ticker,
                form_type="10-Q",
                limit=self.document_limit,
            )
            all_documents.extend(documents)

        # Convert to context
        content = self.document_retriever.documents_to_context(all_documents)

        # Generate consolidated analysis directly
        return await self._call_openai_structured(
            prompt_name="momentum_analysis",
            user_content=f"Analyze this 10-Q content for {ticker}:\n\n{content}",
            response_model=MomentumAnalysis,
        )
