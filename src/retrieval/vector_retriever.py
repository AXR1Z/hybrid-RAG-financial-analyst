"""Vector similarity retrieval using ChromaDB."""

import logging
from typing import List, Dict, Optional, Tuple
import numpy as np

from src.indexing.chroma_store import ChromaDBStore
from src.indexing.embedder import Embedder

logger = logging.getLogger(__name__)


class VectorRetriever:
    """Retrieve documents using vector similarity search."""
    
    def __init__(
        self,
        chroma_store: ChromaDBStore,
        embedder: Embedder,
    ):
        """
        Initialize retriever.
        
        Args:
            chroma_store: ChromaDB store instance
            embedder: Embedder instance
        """
        self.chroma_store = chroma_store
        self.embedder = embedder
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        where_filter: Optional[Dict] = None,
    ) -> List[Tuple[str, str, float, Dict]]:
        """
        Retrieve top-k similar chunks for query.
        
        Args:
            query: Query text
            top_k: Number of results to return
            where_filter: Optional ChromaDB where filter for metadata
            
        Returns:
            List of (chunk_id, text, distance_score, metadata) tuples
        """
        logger.info(f"Vector retrieval for query: {query[:50]}...")
        
        # Embed query
        query_embedding = self.embedder.embed_text(query, normalize=True)
        
        # Query ChromaDB
        results = self.chroma_store.query(
            query_embedding=query_embedding,
            n_results=top_k,
            where_filter=where_filter,
        )
        
        # Convert distances to similarity scores (ChromaDB returns distances, higher = less similar)
        # Convert cosine distance to similarity: 1 - distance
        retrieved = []
        for chunk_id, text, metadata, distance in zip(
            results["ids"],
            results["documents"],
            results["metadatas"],
            results["distances"],
        ):
            # Convert distance to similarity score (0-1 range, higher = more similar)
            similarity = 1 - distance
            retrieved.append((chunk_id, text, similarity, metadata))
        
        logger.info(f"✓ Retrieved {len(retrieved)} chunks")
        return retrieved
