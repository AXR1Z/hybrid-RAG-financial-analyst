"""Embedding generation using sentence-transformers."""

import logging
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

logger = logging.getLogger(__name__)


class Embedder:
    """Generate embeddings for text using sentence-transformers."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedder.
        
        Args:
            model_name: HuggingFace model name for embeddings
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading embedding model: {model_name} on device: {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def embed_text(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Embed a single text string.
        
        Args:
            text: Text to embed
            normalize: Whether to normalize embedding
            
        Returns:
            Embedding array of shape (embedding_dim,)
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        
        if normalize:
            norm = np.linalg.norm(embedding)
            embedding = embedding / norm if norm > 0 else embedding
        
        return embedding
    
    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> List[np.ndarray]:
        """
        Embed a batch of texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            normalize: Whether to normalize embeddings
            
        Returns:
            List of embedding arrays
        """
        logger.info(f"Embedding {len(texts)} texts with batch size {batch_size}")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        
        if normalize:
            # Normalize embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        return [embeddings[i] for i in range(len(embeddings))]


def embed_text(text: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    """Convenience function to embed a single text."""
    embedder = Embedder(model_name)
    return embedder.embed_text(text)


def embed_texts(texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> List[np.ndarray]:
    """Convenience function to embed multiple texts."""
    embedder = Embedder(model_name)
    return embedder.embed_texts(texts)
