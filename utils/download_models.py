#!/usr/bin/env python3
"""
Script to pre-download embedding models.
Used during Docker build to cache models.
"""

import os
import sys
from pathlib import Path


def download_models():
    """Download required embedding models"""

    # Disable tokenizer parallelism to prevent issues
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Cache directory (can be overridden by environment variable)
    cache_dir = os.getenv("EMBEDDER_CACHE_DIR", "/tmp/vector")

    print("Starting embedding models download...")
    print(f"Cache directory: {cache_dir}")

    # Create cache directory if it doesn't exist
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    try:
        # Import libraries
        print("Importing libraries...")
        from fastembed import TextEmbedding
        from fastembed.sparse.bm25 import Bm25
        from fastembed.late_interaction import LateInteractionTextEmbedding

        # Download models
        print("Downloading dense embedding model...")
        TextEmbedding("sentence-transformers/all-MiniLM-L6-v2", cache_dir=cache_dir)

        print("Downloading BM25 sparse model...")
        Bm25("Qdrant/bm25", cache_dir=cache_dir)

        print("Downloading late interaction model...")
        LateInteractionTextEmbedding("colbert-ir/colbertv2.0", cache_dir=cache_dir)

        print("All models downloaded successfully!")
        return True

    except ImportError as e:
        print(f"Error importing libraries: {e}")
        return False
    except Exception as e:
        print(f"Error during download: {e}")
        return False


if __name__ == "__main__":
    success = download_models()
    sys.exit(0 if success else 1)
