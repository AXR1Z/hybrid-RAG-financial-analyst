"""Hybrid retrieval ranking using RRF (Reciprocal Rank Fusion)."""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class RankedChunk:
    """Represents a ranked chunk with combined scores."""
    
    chunk_id: str
    text: str
    rrf_score: float
    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "rrf_score": self.rrf_score,
            "vector_score": self.vector_score,
            "bm25_score": self.bm25_score,
            "metadata": self.metadata or {},
        }


class HybridRanker:
    """Combine vector and BM25 retrieval using Reciprocal Rank Fusion."""
    
    def __init__(self, k: int = 60):
        """
        Initialize ranker.
        
        Args:
            k: RRF constant parameter (standard value: 60)
        """
        self.k = k
    
    def _compute_rrf_score(self, rank: int) -> float:
        """
        Compute RRF score for a rank.
        
        Formula: 1 / (k + rank)
        
        Args:
            rank: 0-based rank
            
        Returns:
            RRF score
        """
        return 1.0 / (self.k + rank + 1)
    
    def fuse(
        self,
        vector_results: List[Tuple[str, str, float, Dict]],
        bm25_results: List[Tuple[str, str, float, Dict]],
        top_k: int = 10,
    ) -> List[RankedChunk]:
        """
        Fuse vector and BM25 results using RRF.
        
        Args:
            vector_results: List of (chunk_id, text, score, metadata) from vector retriever
            bm25_results: List of (chunk_id, text, score, metadata) from BM25 retriever
            top_k: Number of top results to return
            
        Returns:
            List of RankedChunk objects, sorted by RRF score descending
        """
        logger.info("Fusing vector and BM25 results using RRF...")
        
        # Create score map: chunk_id -> {rrf_score, vector_score, bm25_score, text, metadata}
        chunk_scores = defaultdict(lambda: {
            "rrf_score": 0,
            "vector_score": None,
            "bm25_score": None,
            "text": "",
            "metadata": {},
        })
        
        # Process vector results
        for rank, (chunk_id, text, score, metadata) in enumerate(vector_results):
            rrf = self._compute_rrf_score(rank)
            chunk_scores[chunk_id]["rrf_score"] += rrf
            chunk_scores[chunk_id]["vector_score"] = score
            chunk_scores[chunk_id]["text"] = text
            chunk_scores[chunk_id]["metadata"] = metadata
        
        # Process BM25 results
        for rank, (chunk_id, text, score, metadata) in enumerate(bm25_results):
            rrf = self._compute_rrf_score(rank)
            chunk_scores[chunk_id]["rrf_score"] += rrf
            chunk_scores[chunk_id]["bm25_score"] = score
            
            # Update text/metadata if not already set
            if not chunk_scores[chunk_id]["text"]:
                chunk_scores[chunk_id]["text"] = text
                chunk_scores[chunk_id]["metadata"] = metadata
        
        # Deduplicate and sort by RRF score
        ranked = []
        for chunk_id, scores in chunk_scores.items():
            ranked.append(RankedChunk(
                chunk_id=chunk_id,
                text=scores["text"],
                rrf_score=scores["rrf_score"],
                vector_score=scores["vector_score"],
                bm25_score=scores["bm25_score"],
                metadata=scores["metadata"],
            ))
        
        ranked.sort(key=lambda x: x.rrf_score, reverse=True)
        
        # Return top-k
        result = ranked[:top_k]
        logger.info(f"✓ Fused {len(chunk_scores)} unique chunks, returning {len(result)} top results")
        
        return result
