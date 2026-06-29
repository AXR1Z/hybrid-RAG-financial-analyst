"""BM25 sparse retrieval."""

import logging
from typing import List, Dict, Optional, Tuple

from src.indexing.bm25_index import BM25Index

logger = logging.getLogger(__name__)


class BM25Retriever:
    """Retrieve documents using BM25 sparse retrieval."""
    
    def __init__(self, bm25_index: BM25Index):
        """
        Initialize retriever.
        
        Args:
            bm25_index: BM25Index instance
        """
        self.bm25_index = bm25_index
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        where_filter: Optional[Dict] = None,
    ) -> List[Tuple[str, str, float, Dict]]:
        """
        Retrieve top-k results using BM25.
        
        Args:
            query: Query text
            top_k: Number of results to return
            where_filter: Optional metadata filter {key: value}
            
        Returns:
            List of (chunk_id, text, score, metadata) tuples
        """
        logger.info(f"BM25 retrieval for query: {query[:50]}...")
        
        # Search BM25
        ranked = self.bm25_index.search(
            query=query,
            top_k=top_k * 2,  # Get more results to account for filtering
            where_filter=where_filter,
        )
        
        # Retrieve chunk details
        retrieved = []
        for chunk_id, score in ranked[:top_k]:
            chunk_info = self.bm25_index.get_chunk(chunk_id)
            if chunk_info:
                metadata = chunk_info.get("metadata", {})
                text = chunk_info.get("text", "")
                
                retrieved.append((chunk_id, text, score, metadata))
        
        logger.info(f"✓ Retrieved {len(retrieved)} chunks")
        return retrieved
