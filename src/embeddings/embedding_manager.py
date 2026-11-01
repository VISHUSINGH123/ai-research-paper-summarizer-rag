"""
Hugging Face Embedding Manager.
Provides high-performance, normalized local embeddings for scientific literature.
"""

import os
import torch
import logging
from typing import Optional, Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# Recommended embedding models for scientific and academic text
SUPPORTED_EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": 384,
        "description": "Fast, lightweight 384-dim embeddings. Gold standard for real-time RAG."
    },
    "bge-small-en-v1.5": {
        "name": "BAAI/bge-small-en-v1.5",
        "dim": 384,
        "description": "State-of-the-art small embedding model with exceptional retrieval accuracy."
    },
    "all-mpnet-base-v2": {
        "name": "sentence-transformers/all-mpnet-base-v2",
        "dim": 768,
        "description": "Deep 768-dim embeddings offering maximum semantic fidelity."
    }
}


class EmbeddingManager:
    """Manages Hugging Face embedding model lifecycle and caching."""

    _instances: Dict[str, HuggingFaceEmbeddings] = {}

    @classmethod
    def get_embeddings(
        cls,
        model_key: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None
    ) -> HuggingFaceEmbeddings:
        """
        Get or initialize a cached HuggingFaceEmbeddings instance.
        """
        model_info = SUPPORTED_EMBEDDING_MODELS.get(model_key)
        model_name = model_info["name"] if model_info else model_key

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        cache_key = f"{model_name}:{device}"
        if cache_key not in cls._instances:
            logger.info(f"Initializing embedding model '{model_name}' on device '{device}'...")
            
            # Disable HF symlinks warning on Windows
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True}
            )
            cls._instances[cache_key] = embeddings
            logger.info(f"Embedding model '{model_name}' ready.")

        return cls._instances[cache_key]

    @classmethod
    def get_model_info(cls, model_key: str = "all-MiniLM-L6-v2") -> Dict[str, Any]:
        return SUPPORTED_EMBEDDING_MODELS.get(
            model_key,
            {"name": model_key, "dim": 384, "description": "Custom Hugging Face model"}
        )


def get_embedding_model(model_key: str = "all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """Convenience helper to obtain default embeddings."""
    return EmbeddingManager.get_embeddings(model_key)
