'use client';

import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { Send, TrendingUp } from 'lucide-react';

interface AgentAnalysis {
  ticker: string;
  execution_time: number;
  fundamental_analysis: {
    overall_investment_thesis: string;
    investment_grade: string;
    confidence_score: number;
    key_strengths: string[];
    key_concerns: string[];
    recommendation: string;
  };
  momentum_analysis: {
    overall_momentum: string;
    momentum_strength: string;
    key_momentum_drivers: string[];
    momentum_risks: string[];
    short_term_outlook: string;
    momentum_score: number;
  };
  market_sentiment: {
    sentiment_score: number;
    sentiment_direction: string;
    key_news_themes: string[];
    recent_catalysts: string[];
    market_outlook: string;
  };
  final_recommendation: {
    action: string;
    confidence: number;
    rationale: string;
    key_risks: string[];
    key_opportunities: string[];
    time_horizon: string;
  };
}

export function StreamingChat() {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState('');
  const [agentResponse, setAgentResponse] = useState<AgentAnalysis | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isAgentMode, setIsAgentMode] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleLLMStream = async () => {
    try {
      const response = await fetch('/api/llm/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query,
          model: 'llama3-8b-8192',
          temperature: 0.0,
          max_output_tokens: 4096,
          limit: 5,
          filters: {}
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error('No reader available');

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        buffer += chunk;

        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.trim() === '') continue;

          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') break;

            try {
              const parsed = JSON.parse(data);
              // Handle different event types
              if (parsed.type === 'text_delta') {
                setResponse(prev => prev + (parsed.delta || ''));
              } else if (parsed.type === 'error') {
                throw new Error(parsed.message);
              }
              // Ignore source_documents and stream_completed events
            } catch (e) {
              console.error('Error parsing JSON:', e, 'Line:', line);
            }
          }
        }
      }

      // Process any remaining data
      if (buffer.length > 0 && buffer.startsWith('data: ')) {
        const data = buffer.slice(6);
        if (data !== '[DONE]') {
          try {
            const parsed = JSON.parse(data);
            if (parsed.type === 'text_delta') {
              setResponse(prev => prev + (parsed.delta || ''));
            }
          } catch (e) {
            console.error('Error parsing final chunk:', e);
          }
        }
      }
    } catch (err) {
      console.error('LLM Stream Error:', err);
      throw err;
    }
  };

  const handleAgentAnalysis = async () => {
    try {
      const response = await fetch('/api/agent', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: query
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
      }

      const data: AgentAnalysis = await response.json();
      setAgentResponse(data);
    } catch (err) {
      console.error('Agent Analysis Error:', err);
      throw err;
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    
    setIsLoading(true);
    setResponse('');
    setAgentResponse(null);
    setError(null);

    try {
      if (isAgentMode) {
        await handleAgentAnalysis();
      } else {
        await handleLLMStream();
      }
    } catch (err) {
      console.error('Error:', err);
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      setError(errorMessage);
      setResponse('');
    } finally {
      setIsLoading(false);
    }
  };

  const renderAgentResponse = () => {
    if (!agentResponse) return null;

    return (
      <div className="space-y-6">
        {/* Header with ticker and execution time */}
        <div className="flex justify-between items-center border-b pb-2">
          <h3 className="text-xl font-bold">{agentResponse.ticker} - Complete Analysis</h3>
          <span className="text-sm text-muted-foreground">
            Execution time: {agentResponse.execution_time.toFixed(2)}s
          </span>
        </div>

        {/* Final Recommendation - Highlighted */}
        <div className={`p-4 rounded-lg border-2 ${
          agentResponse.final_recommendation.action === 'BUY' ? 'bg-green-50 border-green-500' :
          agentResponse.final_recommendation.action === 'SELL' ? 'bg-red-50 border-red-500' :
          'bg-yellow-50 border-yellow-500'
        }`}>
          <div className="flex justify-between items-center mb-2">
            <h4 className="text-lg font-semibold">Final Recommendation: {agentResponse.final_recommendation.action}</h4>
            <span className="text-sm">Confidence: {(agentResponse.final_recommendation.confidence * 100).toFixed(0)}%</span>
          </div>
          <p className="mb-2">{agentResponse.final_recommendation.rationale}</p>
          <p className="text-sm text-muted-foreground">Time Horizon: {agentResponse.final_recommendation.time_horizon}</p>
        </div>

        {/* Three Analysis Streams */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Fundamental Analysis */}
          <div className="bg-card p-4 rounded-lg border">
            <h4 className="font-semibold mb-2">Fundamental Analysis</h4>
            <div className="text-sm space-y-2">
              <p><strong>Grade:</strong> {agentResponse.fundamental_analysis.investment_grade}</p>
              <p><strong>Recommendation:</strong> {agentResponse.fundamental_analysis.recommendation}</p>
              <p className="text-xs">{agentResponse.fundamental_analysis.overall_investment_thesis}</p>
            </div>
          </div>

          {/* Momentum Analysis */}
          <div className="bg-card p-4 rounded-lg border">
            <h4 className="font-semibold mb-2">Momentum Analysis</h4>
            <div className="text-sm space-y-2">
              <p><strong>Momentum:</strong> {agentResponse.momentum_analysis.overall_momentum}</p>
              <p><strong>Score:</strong> {agentResponse.momentum_analysis.momentum_score}/10</p>
              <p><strong>Outlook:</strong> {agentResponse.momentum_analysis.short_term_outlook}</p>
            </div>
          </div>

          {/* Market Sentiment */}
          <div className="bg-card p-4 rounded-lg border">
            <h4 className="font-semibold mb-2">Market Sentiment</h4>
            <div className="text-sm space-y-2">
              <p><strong>Sentiment:</strong> {agentResponse.market_sentiment.sentiment_direction}</p>
              <p><strong>Score:</strong> {agentResponse.market_sentiment.sentiment_score}/10</p>
              <p className="text-xs">{agentResponse.market_sentiment.market_outlook}</p>
            </div>
          </div>
        </div>

        {/* Risks and Opportunities */}
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-red-50 p-4 rounded-lg">
            <h4 className="font-semibold mb-2 text-red-700">Key Risks</h4>
            <ul className="text-sm space-y-1">
              {agentResponse.final_recommendation.key_risks.map((risk, index) => (
                <li key={index} className="text-red-600">• {risk}</li>
              ))}
            </ul>
          </div>
          <div className="bg-green-50 p-4 rounded-lg">
            <h4 className="font-semibold mb-2 text-green-700">Key Opportunities</h4>
            <ul className="text-sm space-y-1">
              {agentResponse.final_recommendation.key_opportunities.map((opportunity, index) => (
                <li key={index} className="text-green-600">• {opportunity}</li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="w-full">
      <div className="bg-card rounded-lg shadow-sm border border-border">
        <form onSubmit={handleSubmit} className="p-4 space-y-4">
          {/* Agent Mode Toggle */}
          <div className="flex items-center gap-2 mb-4">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={isAgentMode}
                onChange={(e) => setIsAgentMode(e.target.checked)}
                className="w-4 h-4 text-primary bg-background border-border rounded focus:ring-primary"
                disabled={isLoading}
              />
              <span className="text-sm font-medium flex items-center gap-1">
                <TrendingUp className="h-4 w-4" />
                Multi-Stream Analysis (agent)
              </span>
            </label>
            {isAgentMode && (
              <span className="text-xs text-muted-foreground">
                (Comprehensive 3-stream analysis: Fundamental, Momentum, and Market Sentiment)
              </span>
            )}
          </div>

          <div className="flex gap-2">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={isAgentMode 
                ? "Enter company name or ticker symbol (e.g., Apple, AAPL)" 
                : "Ask about any company, stock, or market trend..."
              }
              className="flex-1 px-4 py-3 bg-background border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary text-foreground placeholder:text-muted-foreground"
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={isLoading || !query.trim()}
              className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:bg-muted disabled:text-muted-foreground disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
              aria-label="Send message"
            >
              {isLoading ? 'Processing...' : (
                <>
                  <span>{isAgentMode ? 'Analyze' : 'Search'}</span>
                  <Send className="h-4 w-4" />
                </>
              )}
            </button>
          </div>

          {/* Error Display */}
          {error && (
            <div className="p-3 bg-red-100 border border-red-400 text-red-700 rounded">
              <p className="text-sm font-medium">Error processing request:</p>
              <p className="text-sm">{error}</p>
            </div>
          )}

          {/* Response Display */}
          <div className="p-4 rounded-lg bg-accent min-h-48">
            {isAgentMode && agentResponse ? (
              renderAgentResponse()
            ) : (
              <div className="prose prose-sm max-w-none">
                <ReactMarkdown>
                  {response || 'Your response will appear here...'}
                </ReactMarkdown>
              </div>
            )}
          </div>
        </form>
      </div>
    </div>
  );
}