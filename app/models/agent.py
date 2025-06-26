from pydantic import BaseModel, Field
from typing import List, Optional


# MAIN OUTPUT MODELS (Keep only these 4)
class FundamentalAnalysis(BaseModel):
    """Consolidated Stream 1 Analysis from 10-K"""

    overall_investment_thesis: str = Field(description="Consolidated investment thesis")
    investment_grade: str = Field(description="A, B, C, or D")
    confidence_score: float = Field(description="Analysis confidence 0-1")
    key_strengths: List[str] = Field(description="3 main strengths")
    key_concerns: List[str] = Field(description="3 main concerns")
    recommendation: str = Field(description="buy, hold, sell, or avoid")


class MomentumAnalysis(BaseModel):
    """Consolidated Stream 2 Analysis from 10-Q"""

    overall_momentum: str = Field(description="positive, neutral, or negative")
    momentum_strength: str = Field(description="strong, moderate, or weak")
    key_momentum_drivers: List[str] = Field(description="3 main momentum drivers")
    momentum_risks: List[str] = Field(description="3 main momentum risks")
    short_term_outlook: str = Field(description="bullish, neutral, or bearish")
    momentum_score: float = Field(description="Score from 0-10 for overall momentum")


class MarketSentiment(BaseModel):
    """Stream 3 Analysis from News"""

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
