import os
import html
import uuid
from tqdm.auto import tqdm
from typing import Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# SEC API
from sec_api import QueryApi, ExtractorApi

# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, List

# Embeddings
from fastembed.sparse.bm25 import Bm25
from fastembed.late_interaction import LateInteractionTextEmbedding
from fastembed import TextEmbedding

# Semantic chunking
from sentence_transformers import SentenceTransformer
import hdbscan
import tiktoken

# Configuration
load_dotenv()
SEC_API_KEY = os.getenv("SEC_API_KEY")
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL_ID = "Qdrant/bm25"
COLBERT_MODEL_ID = "colbert-ir/colbertv2.0"
MAX_TOKENS = 384


@dataclass
class ProcessingConfig:
    """Configuration for document processing"""

    max_tokens: int = MAX_TOKENS
    embed_model_id: str = EMBED_MODEL_ID
    sparse_model_id: str = SPARSE_MODEL_ID
    colbert_model_id: str = COLBERT_MODEL_ID


def setup_sec_apis():
    """Initialize SEC API clients"""
    query_api = QueryApi(api_key=SEC_API_KEY)
    extractor_api = ExtractorApi(api_key=SEC_API_KEY)
    return query_api, extractor_api


def setup_qdrant_client():
    """Initialize Qdrant client (assumes collection already exists)"""
    # Load environment variables
    load_dotenv()

    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )

    return client


def setup_embedding_models(config: ProcessingConfig):
    """Initialize all embedding models"""
    print("Loading embedding models...")

    # Dense embeddings
    dense_model = TextEmbedding(config.embed_model_id)

    # Sparse embeddings (BM25)
    bm25_model = Bm25(model_name=config.sparse_model_id)

    # ColBERT embeddings
    colbert_model = LateInteractionTextEmbedding(model_name=config.colbert_model_id)

    return dense_model, bm25_model, colbert_model


def fetch_sec_filing_text(
    ticker: str, form_type: str = "10-K", section: str = "1A"
) -> Optional[Dict[str, Any]]:
    """
    Fetch SEC filing text for a specific company and section.

    Args:
        ticker: Company ticker symbol (e.g., "AAPL")
        form_type: SEC form type (e.g., "10-K", "10-Q")
        section: Section to extract (e.g., "1A" for Risk Factors)

    Returns:
        Dictionary with text content and filing metadata
    """
    query_api, extractor_api = setup_sec_apis()

    try:
        print(f"Fetching {form_type} filing for {ticker}...")

        # Query for the latest filing
        filings = query_api.get_filings(
            {
                "query": f'ticker:{ticker} AND formType:"{form_type}"',
                "from": "0",
                "size": "1",
                "sort": [{"filedAt": {"order": "desc"}}],
            }
        )

        if not filings.get("filings"):
            print(f"No {form_type} filings found for {ticker}")
            return None

        filing = filings["filings"][0]
        filing_url = filing["linkToFilingDetails"]

        print(f"Extracting section {section} from filing...")

        # Extract specific section
        section_text = extractor_api.get_section(filing_url, section, "text")

        if not section_text:
            print(f"Section {section} not found in filing")
            return None

        # Clean up HTML entities
        text_content = html.unescape(section_text)

        # Extract filing metadata
        filing_metadata = {
            "ticker": filing.get("ticker", ticker),
            "companyName": filing.get("companyName", ""),
            "periodOfReport": filing.get("periodOfReport", ""),
            "formType": filing.get("formType", form_type),
        }

        print(f"Successfully extracted {len(text_content)} characters")
        return {"text": text_content, "metadata": filing_metadata}

    except Exception as e:
        print(f"Error fetching SEC filing: {e}")
        return None


def create_semantic_chunks(text_content: str, max_tokens: int = 384):
    """
    Create semantic chunks using clustering approach.
    """
    # Divide em parágrafos
    paragraphs = [
        p.strip() for p in text_content.split("\n") if len(p.strip().split()) > 10
    ]

    # Gera embeddings para clustering
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(paragraphs)

    # Clustering principal
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, metric="euclidean")
    labels = clusterer.fit_predict(embeddings)

    # Agrupa clusters principais
    from collections import defaultdict

    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        if label != -1:
            clusters[f"cluster_{label}"].append(paragraphs[i])

    cluster_chunks = {
        cluster_id: "\n\n".join(pars) for cluster_id, pars in clusters.items()
    }

    # Processa órfãos
    orphan_indices = [i for i, label in enumerate(labels) if label == -1]
    if len(orphan_indices) > 1:
        orphan_embeddings = embeddings[orphan_indices]
        orphan_clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric="euclidean")
        orphan_labels = orphan_clusterer.fit_predict(orphan_embeddings)

        orphan_clusters = defaultdict(list)
        for i, sublabel in enumerate(orphan_labels):
            original_idx = orphan_indices[i]
            if sublabel != -1:
                orphan_clusters[f"orphan_cluster_{sublabel}"].append(
                    paragraphs[original_idx]
                )
            else:
                cluster_chunks[f"single_orphan_{i}"] = paragraphs[original_idx]

        for cluster_id, pars in orphan_clusters.items():
            cluster_chunks[cluster_id] = "\n\n".join(pars)

    # Chunking com tiktoken
    tokenizer = tiktoken.encoding_for_model("gpt-4o")
    final_chunks = []

    for cluster_id, text in cluster_chunks.items():
        tokens = tokenizer.encode(text)

        if len(tokens) <= max_tokens:
            final_chunks.append({"text": text})
        else:
            # Divide o cluster
            paragraphs_list = text.split("\n\n")
            current_chunk = []
            current_tokens = 0

            for para in paragraphs_list:
                para_tokens = len(tokenizer.encode(para))

                if current_tokens + para_tokens > max_tokens and current_chunk:
                    final_chunks.append({"text": "\n\n".join(current_chunk)})
                    current_chunk = [para]
                    current_tokens = para_tokens
                else:
                    current_chunk.append(para)
                    current_tokens += para_tokens

            if current_chunk:
                final_chunks.append({"text": "\n\n".join(current_chunk)})

    return final_chunks


def create_embeddings(chunk_text, dense_model, bm25_model, colbert_model):
    """
    Create the three types of embeddings for a text chunk.
    """
    # Generate embeddings for each model
    dense_embedding = list(dense_model.passage_embed([chunk_text]))[0].tolist()
    sparse_embedding = list(bm25_model.passage_embed([chunk_text]))[0].as_object()
    colbert_embedding = list(colbert_model.passage_embed([chunk_text]))[0].tolist()

    return {
        "dense": dense_embedding,
        "sparse": sparse_embedding,
        "colbertv2.0": colbert_embedding,
    }


def prepare_point(chunk, filing_metadata, embedding_models):
    """
    Prepare a single data point for Qdrant ingestion.
    """
    dense_model, bm25_model, colbert_model = embedding_models

    # Extract text from chunk
    text = chunk.get("text", "")

    try:
        # Create embeddings
        embeddings = create_embeddings(text, dense_model, bm25_model, colbert_model)

        # Create point with your requested metadata
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense": embeddings["dense"],
                "sparse": embeddings["sparse"],
                "colbertv2.0": embeddings["colbertv2.0"],
            },
            payload={
                "text": text,
                "metadata": {
                    "ticker": filing_metadata["ticker"],
                    "companyName": filing_metadata["companyName"],
                    "periodOfReport": filing_metadata["periodOfReport"],
                    "formType": filing_metadata["formType"],
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


def process_and_ingest_filing(
    ticker: str,
    form_type: str = "10-K",
    section: str = "1A",
    config: ProcessingConfig = None,
):
    """
    Complete pipeline to fetch, process and ingest SEC filing.

    Args:
        ticker: Company ticker symbol
        form_type: SEC form type
        section: Section to extract
        config: Processing configuration
    """
    if config is None:
        config = ProcessingConfig()

    print(f"Starting SEC ingestion pipeline for {ticker} {form_type} section {section}")

    # Step 1: Fetch SEC filing text
    filing_data = fetch_sec_filing_text(ticker, form_type, section)
    if not filing_data:
        print("Failed to fetch SEC filing text")
        return False

    text_content = filing_data["text"]
    filing_metadata = filing_data["metadata"]

    # Step 2: Create semantic chunks
    print("Creating semantic chunks...")
    chunks = create_semantic_chunks(text_content, config.max_tokens)

    # Step 3: Setup models and Qdrant
    print("Setting up embedding models...")
    embedding_models = setup_embedding_models(config)

    print("Setting up Qdrant client...")
    qdrant_client = setup_qdrant_client()

    # Step 5: Process and ingest chunks
    print("Processing and ingesting chunks...")
    points = []

    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i + 1}/{len(chunks)}")

        # Convert chunk to dict format if needed
        if hasattr(chunk, "text"):
            chunk_dict = {"text": chunk.text}
        else:
            chunk_dict = chunk

        point = prepare_point(chunk_dict, filing_metadata, embedding_models)
        if point:
            points.append(point)

    if not points:
        print("No valid points to ingest")
        return False

    # Step 6: Ingest to Qdrant
    print(f"Ingesting {len(points)} points to Qdrant...")
    try:
        # Get collection name from environment
        load_dotenv()
        collection_name = os.getenv("COLLECTION_NAME")

        upload_in_batches(
            client=qdrant_client,
            collection_name=collection_name,
            points=points,
            batch_size=4,  # Adjust based on your document size and memory constraints
        )

        # qdrant_client.upsert(collection_name=collection_name, points=points)
        print(f"Successfully ingested {len(points)} chunks to Qdrant")
        return True

    except Exception as e:
        print(f"Error ingesting to Qdrant: {e}")
        return False


# Configuration
ticker = "AAPL"
form_type = "10-Q"
section = "part2item1a"  # Risk Factors

config = ProcessingConfig()

# Run the pipeline
process_and_ingest_filing(
    ticker=ticker, form_type=form_type, section=section, config=config
)
