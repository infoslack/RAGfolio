import asyncio
from typing import Tuple, Dict, Any
from app.analyzers.base_analyzer import BaseAnalyzer
from app.models.agent import (
    RiskAssessment,
    BusinessAnalysis,
    FinancialMetrics,
    ManagementInsights,
    FundamentalAnalysis,
)
import logging

logger = logging.getLogger(__name__)


class FundamentalAnalyzer(BaseAnalyzer[FundamentalAnalysis]):
    """Analyzer for Stream 1: Fundamental Analysis (10-K documents)"""

    async def analyze(self, ticker: str) -> Tuple[FundamentalAnalysis, Dict[str, Any]]:
        """Execute complete fundamental analysis"""
        logger.info(f"Starting fundamental analysis for {ticker}")

        # Run all analyses in parallel using config
        (
            risk_analysis,
            business_analysis,
            financial_analysis,
            management_analysis,
        ) = await asyncio.gather(
            self._analyze_section_from_config(
                ticker=ticker,
                analysis_type="fundamental",
                section_key="risk_factors",
                response_model=RiskAssessment,
                form_type="10-K",
            ),
            self._analyze_section_from_config(
                ticker=ticker,
                analysis_type="fundamental",
                section_key="business_model",
                response_model=BusinessAnalysis,
                form_type="10-K",
            ),
            self._analyze_section_from_config(
                ticker=ticker,
                analysis_type="fundamental",
                section_key="financial_statements",
                response_model=FinancialMetrics,
                form_type="10-K",
            ),
            self._analyze_section_from_config(
                ticker=ticker,
                analysis_type="fundamental",
                section_key="management_discussion",
                response_model=ManagementInsights,
                form_type="10-K",
            ),
        )

        # Prepare detailed results
        detailed_results = {
            "risk_assessment": risk_analysis.model_dump(),
            "business_analysis": business_analysis.model_dump(),
            "financial_metrics": financial_analysis.model_dump(),
            "management_insights": management_analysis.model_dump(),
        }

        # Consolidate analyses
        consolidated = await self._consolidate_analyses(
            ticker,
            risk_analysis,
            business_analysis,
            financial_analysis,
            management_analysis,
        )

        return consolidated, detailed_results

    async def _consolidate_analyses(
        self,
        ticker: str,
        risk_analysis: RiskAssessment,
        business_analysis: BusinessAnalysis,
        financial_analysis: FinancialMetrics,
        management_analysis: ManagementInsights,
    ) -> FundamentalAnalysis:
        """Consolidate all fundamental analyses into a single assessment"""
        consolidation_prompt = f"""
        Based on the analyses of the 4 sections from the 10-K filing for {ticker}, 
        generate a consolidated fundamental analysis:
        
        RISKS: {risk_analysis.model_dump()}
        BUSINESS: {business_analysis.model_dump()}
        FINANCIAL: {financial_analysis.model_dump()}
        MANAGEMENT: {management_analysis.model_dump()}
        """

        return await self._call_openai_structured(
            prompt_name="fundamental_consolidation",
            user_content=consolidation_prompt,
            response_model=FundamentalAnalysis,
        )
