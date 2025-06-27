# RAGfolio - Investment Intelligence Frontend

A modern web interface for RAGfolio, a multi-stream investment analysis system using RAG (Retrieval-Augmented Generation) to analyze stocks through fundamental, momentum, and sentiment analysis.

## Overview

RAGfolio combines structured data retrieval with LLM reasoning to provide comprehensive investment insights. Inspired by the FinGPT paper's approach to financial AI, it offers two modes:

- **Standard RAG Mode**: Quick semantic search across financial documents
- **Multi-Stream Analysis**: Comprehensive investment analysis using three parallel streams

## Features

### Standard RAG Mode
- Semantic search across SEC filings and financial documents
- Real-time streaming responses
- Context-aware answers about companies and market trends

### Multi-Stream Analysis Mode
When enabled, provides comprehensive analysis including:

1. **Fundamental Analysis** (10-K filings)
   - Investment grade (A-D)
   - Key strengths and concerns
   - Long-term investment thesis

2. **Momentum Analysis** (10-Q filings)
   - Current momentum direction and strength
   - Key drivers and risks
   - Short-term outlook (3-6 months)

3. **Market Sentiment** (Recent news)
   - Sentiment score (1-10)
   - Key news themes
   - Recent catalysts

4. **Final Recommendation**
   - Clear BUY/HOLD/SELL action
   - Confidence level
   - Synthesized rationale
   - Key risks and opportunities

## Prerequisites

- Node.js 18+
- RAGfolio backend running at `http://localhost:8000`

## Installation

```bash
# Clone the repository
git clone https://github.com/infoslack/RAGfolio.git
cd RAGfolio/front

# Install dependencies
npm install
# or
yarn install
# or
pnpm install
```

## Configuration

The frontend automatically connects to:
- **Development**: `http://localhost:8000`

## Running the Application

```bash
# Development mode
npm run dev
# or
yarn dev
# or
pnpm dev
```

Visit `http://localhost:3000`

## Usage

### Standard Mode
Simply type your question about any company, stock, or market trend in the search box.

### Multi-Stream Analysis
1. Enable "Multi-Stream Analysis" toggle
2. Enter a company name or ticker symbol (e.g., "Apple" or "AAPL")
3. Click "Analyze" to receive comprehensive investment analysis

## Tech Stack

- **Framework**: Next.js 14 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **UI Components**: Custom React components
- **Streaming**: Server-Sent Events (SSE)

## Project Structure

```
src/
├── app/
│   ├── globals.css         # Global styles
│   ├── layout.tsx          # Root layout
│   └── page.tsx            # Main page
├── components/
│   ├── container.tsx       # Responsive container
│   ├── streaming-chat.tsx  # Main chat interface
│   └── upload-modal.tsx    # Document upload modal
└── lib/
    └── utils.ts            # Utility functions
```

## Troubleshooting

### Connection Issues
```bash
# Verify backend is running
curl http://localhost:8000/health
```

### Build Issues
```bash
# Clear cache
rm -rf .next
rm -rf node_modules/.cache
npm install
```

### Socket Timeout
- Multi-Stream Analysis can take up to 60 seconds
- Timeout is configured for 120 seconds

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

© 2024 TechLevel Pro