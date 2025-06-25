import asyncio
from typing import Tuple, Dict, Any
from app.analyzers.base_analyzer import BaseAnalyzer
from app.models.agent import (
    OperationalUpdate,
    QuarterlyPerformance,
    ShortTermRisks,
    MomentumAnalysis,
)
import logging

logger = logging.getLogger(__name__)


class MomentumAnalyzer(BaseAnalyzer[MomentumAnalysis]):
    """Analyzer for Stream 2: Momentum Analysis (10-Q documents)"""

    async def analyze(self, ticker: str) -> Tuple[MomentumAnalysis, Dict[str, Any]]:
        """Execute complete momentum analysis"""
        logger.info(f"Starting momentum analysis for {ticker}")

        # Run all analyses in parallel
        (
            operational_update,
            quarterly_performance,
            short_term_risks,
        ) = await asyncio.gather(
            self._analyze_operational_updates(ticker),
            self._analyze_quarterly_performance(ticker),
            self._analyze_short_term_risks(ticker),
        )

        # Prepare detailed results
        detailed_results = {
            "operational_update": operational_update.model_dump(),
            "quarterly_performance": quarterly_performance.model_dump(),
            "short_term_risks": short_term_risks.model_dump(),
        }

        # Consolidate analyses
        consolidated = await self._consolidate_analyses(
            ticker,
            operational_update,
            quarterly_performance,
            short_term_risks,
        )

        return consolidated, detailed_results

    async def _analyze_operational_updates(self, ticker: str) -> OperationalUpdate:
        """Analyze 10-Q Part 1 Item 1 - Business Operations"""
        documents = self.document_retriever.query_documents(
            query="business operations developments products services expansion",
            ticker=ticker,
            form_type="10-Q",
        )

        content = self.document_retriever.documents_to_context(documents)

        return await self._call_openai_structured(
            prompt_name="operational_updates",
            user_content=f"Operational content:\n{content}",
            response_model=OperationalUpdate,
        )

    async def _analyze_quarterly_performance(self, ticker: str) -> QuarterlyPerformance:
        """Analyze 10-Q Part 1 Item 2 - Financial Performance"""
        documents = self.document_retriever.query_documents(
            query="financial performance revenue margins liquidity costs quarterly",
            ticker=ticker,
            form_type="10-Q",
        )

        content = self.document_retriever.documents_to_context(documents)

        return await self._call_openai_structured(
            prompt_name="quarterly_performance",
            user_content=f"Financial performance content:\n{content}",
            response_model=QuarterlyPerformance,
        )

    async def _analyze_short_term_risks(self, ticker: str) -> ShortTermRisks:
        """Analyze 10-Q Part 2 Item 1A - Risk Factors"""
        documents = self.document_retriever.query_documents(
            query="risk factors emerging threats regulatory competitive short term",
            ticker=ticker,
            form_type="10-Q",
        )

        content = self.document_retriever.documents_to_context(documents)

        return await self._call_openai_structured(
            prompt_name="short_term_risks",
            user_content=f"Risk factors content:\n{content}",
            response_model=ShortTermRisks,
        )

    async def _consolidate_analyses(
        self,
        ticker: str,
        operational_update: OperationalUpdate,
        quarterly_performance: QuarterlyPerformance,
        short_term_risks: ShortTermRisks,
    ) -> MomentumAnalysis:
        """Consolidate all momentum analyses into a single assessment"""
        consolidation_prompt = f"""
        Based on the quarterly momentum analyses from the most recent 10-Q filing for {ticker}, 
        generate a consolidated momentum assessment:
        
        OPERATIONAL UPDATES: {operational_update.model_dump()}
        QUARTERLY PERFORMANCE: {quarterly_performance.model_dump()}
        SHORT-TERM RISKS: {short_term_risks.model_dump()}
        """

        return await self._call_openai_structured(
            prompt_name="momentum_consolidation",
            user_content=consolidation_prompt,
            response_model=MomentumAnalysis,
        )
