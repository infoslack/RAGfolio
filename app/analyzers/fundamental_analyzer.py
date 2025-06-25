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

        # Run all analyses in parallel
        (
            risk_analysis,
            business_analysis,
            financial_analysis,
            management_analysis,
        ) = await asyncio.gather(
            self._analyze_risk_factors(ticker),
            self._analyze_business_model(ticker),
            self._analyze_financials(ticker),
            self._analyze_management_discussion(ticker),
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

    async def _analyze_risk_factors(self, ticker: str) -> RiskAssessment:
        """Analyze 10-K Section 1A - Risk Factors"""
        documents = self.document_retriever.query_documents(
            query="risk factors regulatory competitive operational threats",
            ticker=ticker,
            form_type="10-K",
        )

        content = self.document_retriever.documents_to_context(documents)

        return await self._call_openai_structured(
            prompt_name="risk_analysis",
            user_content=f"Risk factors content:\n{content}",
            response_model=RiskAssessment,
        )

    async def _analyze_business_model(self, ticker: str) -> BusinessAnalysis:
        """Analyze 10-K Section 1 - Business"""
        documents = self.document_retriever.query_documents(
            query="business operations revenue model competitive advantages",
            ticker=ticker,
            form_type="10-K",
        )

        content = self.document_retriever.documents_to_context(documents)

        return await self._call_openai_structured(
            prompt_name="business_analysis",
            user_content=f"Business content:\n{content}",
            response_model=BusinessAnalysis,
        )

    async def _analyze_financials(self, ticker: str) -> FinancialMetrics:
        """Analyze 10-K Section 8 - Financial Statements"""
        documents = self.document_retriever.query_documents(
            query="financial statements revenue income balance sheet cash flow",
            ticker=ticker,
            form_type="10-K",
        )

        content = self.document_retriever.documents_to_context(documents)

        return await self._call_openai_structured(
            prompt_name="financial_analysis",
            user_content=f"Financial statements content:\n{content}",
            response_model=FinancialMetrics,
        )

    async def _analyze_management_discussion(self, ticker: str) -> ManagementInsights:
        """Analyze 10-K Section 7 - MD&A"""
        documents = self.document_retriever.query_documents(
            query="management discussion analysis outlook guidance strategy",
            ticker=ticker,
            form_type="10-K",
        )

        content = self.document_retriever.documents_to_context(documents)

        return await self._call_openai_structured(
            prompt_name="management_analysis",
            user_content=f"Management discussion content:\n{content}",
            response_model=ManagementInsights,
        )

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
