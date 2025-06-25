from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


# STREAM 1 MODELS (10-K Fundamental Analysis)
class RiskAssessment(BaseModel):
    """Analysis of 10-K Section 1A - Risk Factors"""

    main_risks: List[str] = Field(description="Top 5 main risks identified")
    risk_categories: str = Field(
        description="Main categories: operational, regulatory, technological, macroeconomic"
    )
    emerging_risks: List[str] = Field(description="New or unusual risks mentioned")
    risk_profile: str = Field(description="low, moderate, or high")
    allocation_recommendation: str = Field(
        description="conservative, moderate, or aggressive"
    )
    risk_score: float = Field(description="Risk score from 0-10 (10 = highest risk)")


class BusinessAnalysis(BaseModel):
    """Analysis of 10-K Section 1 - Business"""

    business_model: str = Field(
        description="Summary of business model in 2-3 sentences"
    )
    revenue_streams: List[str] = Field(description="Main revenue sources")
    competitive_advantages: List[str] = Field(
        description="Competitive advantages identified"
    )
    market_position: str = Field(description="leader, challenger, niche, or follower")
    business_stability: str = Field(description="stable, growth, or volatile")


class FinancialMetrics(BaseModel):
    """Analysis of 10-K Section 8 - Financial Statements"""

    revenue_trend: str = Field(
        description="growing, stable, or declining over last 3 years"
    )
    profitability_health: str = Field(description="healthy, moderate, or concerning")
    debt_level: str = Field(description="low, moderate, or high debt")
    cash_position: str = Field(description="strong, adequate, or weak cash position")
    financial_quality_score: float = Field(description="Financial quality score 0-10")
    key_metrics: str = Field(description="Summary of key financial metrics")


class ManagementInsights(BaseModel):
    """Analysis of 10-K Section 7 - MD&A"""

    management_outlook: str = Field(description="optimistic, neutral, or pessimistic")
    strategic_initiatives: List[str] = Field(description="Main strategic initiatives")
    challenges_acknowledged: List[str] = Field(
        description="Challenges acknowledged by management"
    )
    guidance_quality: str = Field(description="clear, vague, or absent")
    management_credibility: str = Field(description="high, medium, or low")


class FundamentalAnalysis(BaseModel):
    """Consolidated Stream 1 Analysis"""

    overall_investment_thesis: str = Field(description="Consolidated investment thesis")
    investment_grade: str = Field(description="A, B, C, or D")
    confidence_score: float = Field(description="Analysis confidence 0-1")
    key_strengths: List[str] = Field(description="3 main strengths")
    key_concerns: List[str] = Field(description="3 main concerns")
    recommendation: str = Field(description="buy, hold, sell, or avoid")


# STREAM 2 MODELS (10-Q Momentum Analysis)
class OperationalUpdate(BaseModel):
    """Analysis of 10-Q Part 1 Item 1 - Business Operations"""

    operational_changes: List[str] = Field(
        description="Key operational changes in the quarter"
    )
    new_developments: List[str] = Field(
        description="New products, services, or business developments"
    )
    expansion_activities: List[str] = Field(
        description="Geographic or market expansion activities"
    )
    operational_momentum: str = Field(
        description="accelerating, stable, or decelerating"
    )
    operational_score: float = Field(
        description="Score from 0-10 for operational progress"
    )


class QuarterlyPerformance(BaseModel):
    """Analysis of 10-Q Part 1 Item 2 - Financial Performance"""

    revenue_performance: str = Field(
        description="strong, adequate, or weak quarterly revenue"
    )
    margin_trends: str = Field(description="improving, stable, or declining margins")
    liquidity_position: str = Field(
        description="strong, adequate, or concerning liquidity"
    )
    cost_management: str = Field(
        description="effective, moderate, or poor cost control"
    )
    financial_momentum: str = Field(description="positive, neutral, or negative")
    performance_score: float = Field(
        description="Score from 0-10 for quarterly performance"
    )


class ShortTermRisks(BaseModel):
    """Analysis of 10-Q Part 2 Item 1A - Risk Factors"""

    emerging_risks: List[str] = Field(description="New or escalating risks identified")
    risk_intensity: str = Field(
        description="increasing, stable, or decreasing risk levels"
    )
    immediate_concerns: List[str] = Field(description="Most pressing near-term risks")
    risk_mitigation: str = Field(
        description="strong, moderate, or weak risk management"
    )
    risk_outlook: str = Field(description="improving, stable, or deteriorating")
    risk_score: float = Field(description="Score from 0-10 for short-term risk level")


class MomentumAnalysis(BaseModel):
    """Consolidated Stream 2 Analysis"""

    overall_momentum: str = Field(description="positive, neutral, or negative")
    momentum_strength: str = Field(description="strong, moderate, or weak")
    key_momentum_drivers: List[str] = Field(description="3 main momentum drivers")
    momentum_risks: List[str] = Field(description="3 main momentum risks")
    short_term_outlook: str = Field(description="bullish, neutral, or bearish")
    momentum_score: float = Field(description="Score from 0-10 for overall momentum")


# STREAM 3 MODELS (News Sentiment Analysis)
class MarketSentiment(BaseModel):
    """Analysis of recent news sentiment"""

    sentiment_score: float = Field(
        description="Score from 1-10 (1=Very Negative, 5-6=Neutral, 10=Very Positive)"
    )
    sentiment_direction: str = Field(description="Positive, Neutral, or Negative")
    key_news_themes: List[str] = Field(
        description="Main topics being discussed (earnings, products, regulation, etc.)"
    )
    recent_catalysts: List[str] = Field(
        description="Specific events or announcements that could move the stock"
    )
    market_outlook: str = Field(
        description="Brief synthesis of current market perception"
    )


# AGGREGATOR MODELS (Final Output)
class FinalRecommendation(BaseModel):
    """Final consolidated investment recommendation"""

    action: str = Field(description="BUY, HOLD, or SELL")
    confidence: float = Field(description="Confidence in recommendation 0-1")
    rationale: str = Field(description="Rationale combining the 3 streams")
    key_risks: List[str] = Field(description="Top risks from all streams")
    key_opportunities: List[str] = Field(
        description="Top opportunities from all streams"
    )
    time_horizon: str = Field(description="Short-term, Medium-term, or Long-term")


# API REQUEST/RESPONSE MODELS
class AgentRequest(BaseModel):
    """Request model for agent analysis"""

    ticker: Optional[str] = Field(
        default=None, description="Stock ticker symbol (e.g., AAPL, MSFT)"
    )
    message: Optional[str] = Field(
        default=None, description="Message to extract ticker from"
    )
    include_details: Optional[bool] = Field(
        default=False, description="Include detailed analysis for each stream"
    )


class StreamResults(BaseModel):
    """Detailed results for each stream"""

    stream1_details: Optional[Dict[str, Any]] = Field(
        default=None, description="Detailed fundamental analysis"
    )
    stream2_details: Optional[Dict[str, Any]] = Field(
        default=None, description="Detailed momentum analysis"
    )
    stream3_details: Optional[Dict[str, Any]] = Field(
        default=None, description="Detailed sentiment analysis"
    )


class AgentResponse(BaseModel):
    """Response model for agent analysis"""

    ticker: str = Field(description="Stock ticker analyzed")
    execution_time: float = Field(description="Total execution time in seconds")

    # Main stream results
    fundamental_analysis: FundamentalAnalysis = Field(
        description="Stream 1 - Fundamental analysis results"
    )
    momentum_analysis: MomentumAnalysis = Field(
        description="Stream 2 - Momentum analysis results"
    )
    market_sentiment: MarketSentiment = Field(
        description="Stream 3 - Market sentiment results"
    )

    # Final recommendation
    final_recommendation: FinalRecommendation = Field(
        description="Aggregated final recommendation"
    )

    # Optional detailed results
    detailed_results: Optional[StreamResults] = Field(
        default=None, description="Detailed analysis if requested"
    )
