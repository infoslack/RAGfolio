import os
import uuid
import tiktoken
from tqdm.auto import tqdm
from dotenv import load_dotenv

# News scraping
import yfinance as yf
import trafilatura

# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, List

# Embeddings
from fastembed.sparse.bm25 import Bm25
from fastembed.late_interaction import LateInteractionTextEmbedding
from fastembed import TextEmbedding

# Configuration
load_dotenv()
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL_ID = "Qdrant/bm25"
COLBERT_MODEL_ID = "colbert-ir/colbertv2.0"
MAX_TOKENS = 384


def setup_qdrant_client():
    """Initialize Qdrant client"""
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    return client


def setup_embedding_models():
    """Initialize all embedding models"""
    print("Loading embedding models...")

    # Dense embeddings
    dense_model = TextEmbedding(EMBED_MODEL_ID)

    # Sparse embeddings (BM25)
    bm25_model = Bm25(model_name=SPARSE_MODEL_ID)

    # ColBERT embeddings
    colbert_model = LateInteractionTextEmbedding(model_name=COLBERT_MODEL_ID)

    return dense_model, bm25_model, colbert_model


def fetch_news_data(ticker: str, max_stories: int = 10):
    """Fetch news data from Yahoo Finance"""
    print(f"Fetching news for {ticker}...")

    dat = yf.Ticker(ticker)
    news = dat.news

    news_data = []

    for item in news[:max_stories]:
        content = item.get("content", {})
        content_type = content.get("contentType")

        # Filter only stories
        if content_type != "STORY":
            continue

        canonical_url = content.get("canonicalUrl", {})
        title = content.get("title")
        date = content.get("pubDate")
        url = canonical_url.get("url")

        # Filter only Yahoo Finance links
        if "finance.yahoo.com" not in url:
            continue

        # Extract article text
        downloaded = trafilatura.fetch_url(url)
        text_content = trafilatura.extract(downloaded)

        if text_content:
            news_data.append(
                {"title": title, "url": url, "date": date, "text": text_content}
            )
            print(f"Extracted: {title}")

    return news_data


def create_text_chunks(text_content: str, max_tokens: int = MAX_TOKENS):
    """Create chunks based on token count only"""
    tokenizer = tiktoken.encoding_for_model("gpt-4o")

    # Split text into paragraphs
    paragraphs = [p.strip() for p in text_content.split("\n") if p.strip()]

    chunks = []
    current_chunk = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = len(tokenizer.encode(para))

        if current_tokens + para_tokens > max_tokens and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_tokens = para_tokens
        else:
            current_chunk.append(para)
            current_tokens += para_tokens

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def create_embeddings(chunk_text, dense_model, bm25_model, colbert_model):
    """Create the three types of embeddings for a text chunk"""
    dense_embedding = list(dense_model.passage_embed([chunk_text]))[0].tolist()
    sparse_embedding = list(bm25_model.passage_embed([chunk_text]))[0].as_object()
    colbert_embedding = list(colbert_model.passage_embed([chunk_text]))[0].tolist()

    return {
        "dense": dense_embedding,
        "sparse": sparse_embedding,
        "colbertv2.0": colbert_embedding,
    }


def prepare_news_point(chunk_text, news_metadata, ticker, embedding_models):
    """Prepare a single data point for Qdrant ingestion"""
    dense_model, bm25_model, colbert_model = embedding_models

    try:
        # Create embeddings
        embeddings = create_embeddings(
            chunk_text, dense_model, bm25_model, colbert_model
        )

        # Create point
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense": embeddings["dense"],
                "sparse": embeddings["sparse"],
                "colbertv2.0": embeddings["colbertv2.0"],
            },
            payload={
                "text": chunk_text,
                "metadata": {
                    "ticker": ticker,
                    "title": news_metadata["title"],
                    "url": news_metadata["url"],
                    "date": news_metadata["date"],
                    "chunk_type": "news",
                },
            },
        )

        return point

    except Exception as e:
        print(f"Error processing chunk: {e}")
        return None


def upload_in_batches(
    client: QdrantClient,
    collection_name: str,
    points: List[PointStruct],
    batch_size: int = 10,
):
    """
    Upload points to Qdrant in batches with progress tracking.
    """
    # Calculate number of batches
    n_batches = (len(points) + batch_size - 1) // batch_size

    print(
        f"Uploading {len(points)} points to collection '{collection_name}' in {n_batches} batches..."
    )

    # Process each batch with progress bar
    for i in tqdm(range(0, len(points), batch_size), total=n_batches):
        batch = points[i : i + batch_size]
        client.upload_points(collection_name=collection_name, points=batch)

    print(
        f"Successfully uploaded {len(points)} points to collection '{collection_name}'"
    )


def process_and_ingest_news(ticker: str, max_stories: int = 10):
    """Complete pipeline to fetch, process and ingest news"""
    print(f"Starting news ingestion pipeline for {ticker}")

    # Step 1: Fetch news data
    news_data = fetch_news_data(ticker, max_stories)
    if not news_data:
        print("No news data found")
        return False

    # Step 2: Setup models and Qdrant
    embedding_models = setup_embedding_models()
    qdrant_client = setup_qdrant_client()

    # Step 3: Process each news article
    all_points = []

    for news_item in news_data:
        print(f"Processing: {news_item['title']}")

        # Create chunks for this article
        chunks = create_text_chunks(news_item["text"])
        print(f"Created {len(chunks)} chunks")

        # Process each chunk
        for chunk_text in chunks:
            point = prepare_news_point(chunk_text, news_item, ticker, embedding_models)
            if point:
                all_points.append(point)

    # Step 4: Ingest to Qdrant
    if all_points:
        collection_name = os.getenv("COLLECTION_NAME")
        upload_in_batches(
            client=qdrant_client,
            collection_name=collection_name,
            points=all_points,
            batch_size=4,
        )
        print(
            f"Successfully ingested {len(all_points)} chunks from {len(news_data)} articles"
        )
        return True
    else:
        print("No valid points to ingest")
        return False


# Configuration and execution
ticker = "AAPL"
max_stories = 10

# Run the pipeline
process_and_ingest_news(ticker=ticker, max_stories=max_stories)
