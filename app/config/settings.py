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

    # Models cache
    embedder_cache_dir: Optional[str] = "/tmp/vector"
    embedder_local_files_only: bool = True

    # LLM Configuration
    llm_api_key: Optional[str] = None
    llm_model: str = "llama3-8b-8192"
    llm_temperature: float = 0.0
    llm_max_output_tokens: int = 4096

    # Document retrieval settings
    document_search_limit: int = 3
    news_search_limit: int = 3

    # LLM analysis settings
    analysis_temperature: float = 0.0
    analysis_max_tokens: Optional[int] = None

    # Ticker extraction settings
    ticker_extraction_temperature: float = 0.0
    ticker_extraction_max_tokens: int = 5

    # Config file paths
    queries_config_path: str = "app/config/queries.yaml"
    ticker_mappings_path: str = "app/config/ticker_mappings.yaml"

    model_config = {"env_file": ".env", "extra": "allow"}
