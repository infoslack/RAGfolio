from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # API Configuration
    api_title: str = "RAG API"
    api_description: str = "RAG API with Hybrid Search"
    api_version: str = "0.1.0"

    # Qdrant Configuration
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    collection_name: str = "documents"
    qdrant_timeout: float = 60.0
    prefetch_limit: int = 25

    # Model Configuration
    dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    bm25_model_name: str = "Qdrant/bm25"
    late_interaction_model_name: str = "colbert-ir/colbertv2.0"

    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.5
    openai_max_output_tokens: int = 4096
    openai_system_prompt: str = """Você é um assistente útil que responde a perguntas com base apenas no contexto fornecido.

    Contexto:
    {context}

    Pergunta: {query}

    Responda à pergunta acima usando apenas as informações do contexto fornecido."""

    # Document retrieval settings
    document_search_limit: int = 5
    news_search_limit: int = 10

    # OpenAI analysis settings
    analysis_temperature: float = 0.0
    analysis_max_tokens: Optional[int] = None  # Use OpenAI default

    # Ticker extraction settings
    ticker_extraction_temperature: float = 0.0
    ticker_extraction_max_tokens: int = 10

    # Config file paths
    queries_config_path: str = "app/config/queries.yaml"
    ticker_mappings_path: str = "app/config/ticker_mappings.yaml"

    model_config = {"env_file": ".env", "extra": "allow"}
