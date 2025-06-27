# RAGfolio

A multi-stream investment analysis system using RAG (Retrieval-Augmented Generation) to analyze stocks through fundamental, momentum, and sentiment analysis. Inspired by the [FinGPT paper's](https://arxiv.org/abs/2306.06031) approach to financial AI, RAGfolio combines structured data retrieval with LLM reasoning to provide comprehensive investment insights.

## Solution Overview

This project combines advanced vector search with LLM analysis to provide comprehensive investment insights through three parallel streams:

1. **Fundamental Analysis** - Analyzes 10-K SEC filings for long-term business health
2. **Momentum Analysis** - Processes 10-Q quarterly reports for short-term trends  
3. **Sentiment Analysis** - Evaluates recent news for market perception

### Hybrid Vector Search Architecture

The system uses **Qdrant vector database** with a sophisticated three-tier hybrid search approach:

#### Search Methods
- **Dense Vectors** (`sentence-transformers/all-MiniLM-L6-v2`) - Captures semantic meaning and context
- **Sparse Vectors** (`BM25`) - Provides precise keyword matching and term frequency relevance
- **Late Interaction** (`ColBERTv2`) - Enables token-level similarity for nuanced understanding

#### How It Works
The system employs a two-stage retrieval process: first, candidates are retrieved using both dense semantic search and sparse keyword matching to cast a wide net of potentially relevant documents. Then, [ColBERTv2's late interaction](https://jina.ai/news/what-is-colbert-and-late-interaction-and-why-they-matter-in-search/) model reranks these candidates using fine-grained token-level similarities, producing final results that combine semantic understanding, keyword precision, and nuanced token-level relevance.

The architecture works robustly across different query types, from broad investment concepts to specific financial terms, while maintaining scalable performance through its efficient two-stage design that can handle large document collections without sacrificing retrieval quality.

### Analysis Stream Details

#### Fundamental Analysis (Stream 1)
The fundamental analysis stream processes [10-K SEC filings](https://www.investor.gov/introduction-investing/investing-basics/glossary/form-10-k) through targeted RAG queries that retrieve specific sections of annual reports. The system executes four distinct searches: `"risk factors regulatory competitive operational threats challenges"` to identify business risks, `"business operations revenue model competitive advantages market position"` to understand the core business, `"financial statements revenue income balance sheet cash flow debt"` to assess financial health, and `"management discussion analysis outlook guidance strategy MD&A"` to capture management's perspective. The retrieved content is then analyzed using a specialized prompt that acts as a senior investment analyst, synthesizing all information into a consolidated assessment that includes an overall investment thesis, letter grade (A-D), confidence score, key strengths and concerns, and a clear `buy/hold/sell` recommendation based on financial health, competitive position, and risk profile.

#### Momentum Analysis (Stream 2)  
The momentum analysis focuses on [10-Q quarterly filings](https://www.investor.gov/introduction-investing/investing-basics/glossary/form-10-q) to assess short-term business trajectory and recent developments. Three targeted queries retrieve relevant sections: `"business operations developments products services expansion quarterly updates"` to capture operational changes, `"financial performance revenue margins liquidity costs quarterly results"` to evaluate recent financial trends, and `"risk factors emerging threats regulatory competitive short term challenges"` to identify near-term risks. The analysis prompt specializes in quarterly momentum assessment, determining whether momentum is positive, neutral, or negative while rating its strength and identifying key momentum drivers and risks. The output provides a short-term outlook for the next 3-6 months with a momentum score from 0-10, focusing specifically on quarter-over-quarter changes and emerging trends.

#### Sentiment Analysis (Stream 3)
The sentiment analysis stream processes recent news articles using the query `"{ticker} earnings revenue stock price announcement news"` to retrieve market-relevant content about the company. Unlike the SEC document streams, this analysis works with news data that includes titles, dates, and article content formatted for sentiment evaluation. The specialized prompt acts as a financial news analyst, assessing overall market sentiment on a `1-10 scale` where 1 represents very negative sentiment and 10 represents very positive sentiment. The analysis identifies key news themes such as earnings, product announcements, or regulatory developments, pinpoints recent catalysts that could move the stock price, and synthesizes current market perception into a brief outlook that captures how the financial community currently views the company.

#### Final Aggregation
After all three streams complete their parallel analysis, a final aggregation step synthesizes the results into a unified investment recommendation. The aggregation prompt receives the complete output from all three streams and acts as a senior investment analyst tasked with combining fundamental health, recent momentum, and market sentiment into a coherent investment thesis. The system applies a decision framework where recommendations with high confidence emerge when two or more streams align, while conflicting signals result in lower confidence recommendations with explicit explanations of the divergence. The final output provides a clear `BUY/HOLD/SELL` action with confidence score, comprehensive rationale explaining how the streams were weighted and combined, identification of key risks and opportunities across all analysis dimensions, and an appropriate time horizon for the recommendation.

### Architecture Inspiration

This multi-stream approach is based on [Anthropic's Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) methodology, specifically implementing **Workflow Parallelization**. Rather than processing information sequentially, the system runs independent analysis streams in parallel, each specialized for different data types and analysis perspectives. This parallel execution reduces latency while allowing each stream to focus deeply on its domain expertise, ultimately producing more comprehensive and reliable investment insights than any single analysis approach could achieve.

## Directory Structure

```
app/
├── analyzers/              # Analysis stream implementations
│   ├── __init__.py
│   ├── base_analyzer.py    # Base class for all analyzers
│   ├── fundamental_analyzer.py  # Stream 1: 10-K fundamental analysis
│   ├── momentum_analyzer.py     # Stream 2: 10-Q momentum analysis
│   └── sentiment_analyzer.py    # Stream 3: News sentiment analysis
│
├── config/                 # Configuration files
│   ├── queries.yaml        # Search queries for different analysis types
│   ├── settings.py         # Application settings and environment config
│   └── ticker_mappings.yaml # Company name to ticker symbol mappings
│
├── models/                 # Pydantic data models
│   ├── agent.py           # Analysis result models and API schemas
│   ├── api.py             # API request/response models
│   └── embeddings.py      # Embedding and document models
│
├── prompts/               # LLM system prompts
│   ├── final_recommendation.md  # Final aggregation prompt
│   ├── fundamental_analysis.md  # Stream 1 analysis prompt
│   ├── momentum_analysis.md     # Stream 2 analysis prompt
│   ├── sentiment_analysis.md    # Stream 3 analysis prompt
│   ├── ticker_extraction.md     # Ticker symbol extraction prompt
│   └── rag_response.md         # General RAG response prompt
│
├── routers/               # FastAPI route handlers
│   ├── agent.py          # Investment analysis endpoints
│   ├── llm.py            # LLM/RAG endpoints
│   └── search.py         # Document search endpoints
│
├── services/              # Business logic and external integrations
│   ├── agent_service.py       # Main orchestration service
│   ├── config_loader.py       # Configuration file loader
│   ├── document_retriever.py  # Document search and retrieval
│   ├── embedder.py           # Query embedding generation
│   ├── llm_service.py        # LLM interaction service
│   ├── prompt_manager.py     # Prompt loading and caching
│   ├── retriever.py          # Qdrant vector database client
│   └── ticker_extractor.py   # Company name to ticker extraction
│
├── utils/                 # Utility functions
│   ├── __init__.py
│   └── decorators.py     # Error handling and logging decorators
│
└── main.py               # FastAPI application entry point
```

## Key Components

- **Analyzers**: Three independent analysis streams that process different data sources
- **Services**: Core business logic for document retrieval, LLM interactions, and orchestration
- **Models**: Type-safe data structures using Pydantic for all API interactions and analysis results
- **Prompts**: Modular prompt templates for different analysis tasks
- **Config**: Centralized configuration management for queries, settings, and mappings

## Setup and Installation

### Prerequisites
- Python 3.8+
- [uv](https://github.com/astral-sh/uv) package manager
- Qdrant vector database
- Groq API key for LLM access

### Installation

#### Option 1: Install from requirements file
```bash
uv pip install -r requirements.txt
```

#### Option 2: Compile and install dependencies
```bash
# Compile requirements
uv pip compile requirements.in -o requirements.txt

# Then install
uv pip install -r requirements.txt
```

### Configuration
1. Copy the example environment file and configure your settings:
```bash
cp .env-example .env
```

2. Edit the `.env` file with your API keys and configuration

3. Ensure your Qdrant vector database is running and populated with financial documents

### Running the API
```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

### Example Usage
```bash
curl -X POST "http://localhost:8000/agent" -H "Content-Type: application/json" -d '{
    "message": "How is Apple doing today?"
  }'
```
This will trigger the complete three-stream analysis for Apple (AAPL), returning fundamental analysis, momentum assessment, market sentiment, and a final investment recommendation.

### Example Output
The API returns a comprehensive analysis structured across the three streams:
**Stream 1 - Fundamental Analysis**
```json
{
    "overall_investment_thesis": "Apple Inc. demonstrates strong financial health with consistent revenue generation and profitability, but faces significant risks from competition, regulatory challenges, and reliance on global supply chains.",
    "investment_grade": "B",
    "confidence_score": 0.85,
    "key_strengths": [
        "Strong brand loyalty and market presence",
        "Diverse product and service offerings",
        "Robust financial performance with consistent profitability"
    ],
    "recommendation": "hold"
}
```
**Stream 2 - Momentum Analysis**
```json
{
    "overall_momentum": "neutral",
    "momentum_strength": "moderate",
    "key_momentum_drivers": [
        "Strong product sales growth in services segment",
        "Stable gross margin performance despite cost pressures"
    ],
    "short_term_outlook": "neutral",
    "momentum_score": 5.0
}
```
**Stream 3 - Market Sentiment**
```json
{
    "sentiment_score": 5.0,
    "sentiment_direction": "Neutral",
    "key_news_themes": ["regulation", "competition", "product marketing"],
    "recent_catalysts": [
        "Apple in last-minute talks to avoid EU fines",
        "iPhone customers upset by Apple Wallet ad"
    ],
    "market_outlook": "The market sentiment around Apple is currently neutral, with regulatory pressures from the EU and customer dissatisfaction regarding marketing strategies impacting perception."
}
```
**Final Aggregated Recommendation**
```json
{
    "action": "HOLD",
    "confidence": 0.65,
    "rationale": "The analysis from the three streams presents a mixed picture for Apple Inc. Given the alignment of concerns across all streams and the lack of strong positive momentum, a 'HOLD' recommendation is warranted.",
    "key_risks": [
        "Intense competition in technology and consumer electronics",
        "Regulatory risks and potential legal challenges"
    ],
    "time_horizon": "Medium-term"
}
```