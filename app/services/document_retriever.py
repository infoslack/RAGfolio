from typing import List
from app.models.embeddings import Document
from app.services.embedder import QueryEmbedder
from app.services.retriever import QdrantRetriever
import logging

logger = logging.getLogger(__name__)


class DocumentRetriever:
    """Service responsible for retrieving documents from Qdrant"""

    def __init__(self, embedder: QueryEmbedder, retriever: QdrantRetriever):
        self.embedder = embedder
        self.retriever = retriever

    def query_documents(
        self, query: str, ticker: str, form_type: str = "10-K", limit: int = 5
    ) -> List[Document]:
        """Query SEC documents using embeddings"""
        try:
            query_embeddings = self.embedder.embed_query(query)
            documents = self.retriever.search_documents(
                embeddings=query_embeddings,
                filters={"ticker": ticker, "formType": form_type},
                limit=limit,
            )

            logger.info(
                f"Retrieved {len(documents)} documents for {ticker} ({form_type})"
            )
            return documents

        except Exception as e:
            logger.error(f"Document query failed: {str(e)}")
            return []

    def query_news(self, query: str, ticker: str, limit: int = 10) -> List[Document]:
        """Query news articles using embeddings"""
        try:
            query_embeddings = self.embedder.embed_query(query)
            documents = self.retriever.search_documents(
                embeddings=query_embeddings,
                filters={"ticker": ticker, "chunk_type": "news"},
                limit=limit,
            )

            logger.info(f"Retrieved {len(documents)} news articles for {ticker}")
            return documents

        except Exception as e:
            logger.error(f"News query failed: {str(e)}")
            return []

    @staticmethod
    def documents_to_context(documents: List[Document], max_chars: int = 15000) -> str:
        """Convert documents to context string for LLM"""
        if not documents:
            return "No relevant content found"

        content_parts = [doc.page_content for doc in documents if doc.page_content]
        full_content = "\n\n".join(content_parts)

        # Truncate if too long
        if len(full_content) > max_chars:
            full_content = (
                full_content[:max_chars] + "\n\n[Content truncated due to length...]"
            )
        return full_content

    @staticmethod
    def news_to_context(documents: List[Document]) -> str:
        """Convert news documents to formatted context string"""
        if not documents:
            return "No news found"

        news_items = []
        for doc in documents:
            metadata = doc.metadata or {}
            title = metadata.get("title", "No title")
            date = metadata.get("date", "No date")
            content = doc.page_content or ""

            news_items.append(f"TITLE: {title}\nDATE: {date}\nCONTENT: {content}\n")

        return "\n" + "=" * 50 + "\n".join(news_items)
