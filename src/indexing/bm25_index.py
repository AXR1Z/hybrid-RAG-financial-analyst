"""BM25 indexing and retrieval."""

import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25Index:
    """Build and query BM25 index over document chunks."""
    
    def __init__(self):
        """Initialize BM25 index."""
        self.bm25 = None
        self.chunk_registry = {}  # chunk_id -> chunk info
        self.chunk_ids = []  # ordered list of chunk_ids for ranking
    
    def build(self, chunks: List[Dict[str, str]]) -> None:
        """
        Build BM25 index from chunks.
        
        Args:
            chunks: List of dicts with keys: chunk_id, text
        """
        logger.info(f"Building BM25 index from {len(chunks)} chunks")
        
        # Tokenize documents (simple whitespace + lowercase)
        corpus = []
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            text = chunk["text"]
            
            # Store chunk info
            self.chunk_registry[chunk_id] = {
                "text": text,
                "metadata": chunk.get("metadata", {}),
            }
            
            # Tokenize
            tokens = text.lower().split()
            corpus.append(tokens)
            self.chunk_ids.append(chunk_id)
        
        # Build BM25
        self.bm25 = BM25Okapi(corpus)
        logger.info(f"BM25 index built. Registry size: {len(self.chunk_registry)}")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        where_filter: Optional[Dict] = None,
    ) -> List[Tuple[str, float]]:
        """
        Search the BM25 index.
        
        Args:
            query: Query text
            top_k: Number of top results to return
            where_filter: Optional metadata filter {key: value}
            
        Returns:
            List of (chunk_id, score) tuples, sorted by score descending
        """
        if self.bm25 is None:
            logger.warning("BM25 index not built")
            return []
        
        # Tokenize query
        query_tokens = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Rank by score
        ranked = sorted(
            zip(self.chunk_ids, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        
        # Apply metadata filter if provided
        if where_filter:
            ranked = self._filter_by_metadata(ranked, where_filter)
        
        return ranked[:top_k]
    
    def _filter_by_metadata(
        self,
        ranked_results: List[Tuple[str, float]],
        where_filter: Dict,
    ) -> List[Tuple[str, float]]:
        """
        Filter ranked results by metadata.
        
        Args:
            ranked_results: List of (chunk_id, score)
            where_filter: Metadata filter {key: value or [values]}
            
        Returns:
            Filtered list
        """
        filtered = []
        
        for chunk_id, score in ranked_results:
            metadata = self.chunk_registry[chunk_id].get("metadata", {})
            
            # Check all filter conditions
            matches = True
            for key, value in where_filter.items():
                if isinstance(value, list):
                    if metadata.get(key) not in value:
                        matches = False
                        break
                else:
                    if metadata.get(key) != value:
                        matches = False
                        break
            
            if matches:
                filtered.append((chunk_id, score))
        
        return filtered
    
    def save(self, bm25_path: Path, registry_path: Path) -> None:
        """
        Save BM25 index to disk.
        
        Args:
            bm25_path: Path to save BM25 pickle
            registry_path: Path to save chunk registry JSON
        """
        if self.bm25 is None:
            logger.warning("BM25 index not built, skipping save")
            return
        
        logger.info(f"Saving BM25 index to {bm25_path}")
        
        # Save BM25
        with open(bm25_path, "wb") as f:
            pickle.dump(self.bm25, f)
        
        # Save registry and chunk_ids
        registry_data = {
            "chunk_ids": self.chunk_ids,
            "chunk_registry": self.chunk_registry,
        }
        
        with open(registry_path, "w") as f:
            json.dump(registry_data, f, indent=2)
        
        logger.info("BM25 index saved")
    
    def load(self, bm25_path: Path, registry_path: Path) -> None:
        """
        Load BM25 index from disk.
        
        Args:
            bm25_path: Path to BM25 pickle
            registry_path: Path to chunk registry JSON
        """
        logger.info(f"Loading BM25 index from {bm25_path}")
        
        # Load BM25
        with open(bm25_path, "rb") as f:
            self.bm25 = pickle.load(f)
        
        # Load registry
        with open(registry_path, "r") as f:
            registry_data = json.load(f)
        
        self.chunk_ids = registry_data["chunk_ids"]
        self.chunk_registry = registry_data["chunk_registry"]
        
        logger.info(f"BM25 index loaded. Chunks: {len(self.chunk_ids)}")
    
    def get_chunk(self, chunk_id: str) -> Optional[Dict]:
        """Get chunk info by ID."""
        return self.chunk_registry.get(chunk_id)
