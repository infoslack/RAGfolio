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

        # Run all analyses in parallel using config
        (
            operational_update,
            quarterly_performance,
            short_term_risks,
        ) = await asyncio.gather(
            self._analyze_section_from_config(
                ticker=ticker,
                analysis_type="momentum",
                section_key="operational_updates",
                response_model=OperationalUpdate,
                form_type="10-Q",
            ),
            self._analyze_section_from_config(
                ticker=ticker,
                analysis_type="momentum",
                section_key="quarterly_performance",
                response_model=QuarterlyPerformance,
                form_type="10-Q",
            ),
            self._analyze_section_from_config(
                ticker=ticker,
                analysis_type="momentum",
                section_key="short_term_risks",
                response_model=ShortTermRisks,
                form_type="10-Q",
            ),
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
