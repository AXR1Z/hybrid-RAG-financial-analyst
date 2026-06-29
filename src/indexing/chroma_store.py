"""ChromaDB vector store wrapper."""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
import chromadb
from chromadb.config import Settings
import numpy as np

logger = logging.getLogger(__name__)


class ChromaDBStore:
    """Wrapper around ChromaDB for persistent vector storage."""
    
    COLLECTION_NAME = "sec_10q"
    
    def __init__(self, db_path: Path, collection_name: str = COLLECTION_NAME):
        """
        Initialize ChromaDB store.
        
        Args:
            db_path: Path to persistent ChromaDB directory
            collection_name: Name of the collection
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        
        logger.info(f"Initializing ChromaDB at {self.db_path}")
        
        # Initialize persistent client
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        
        logger.info(f"Collection '{collection_name}' ready")
    
    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[np.ndarray],
    ) -> None:
        """
        Add chunks and embeddings to ChromaDB.
        
        Args:
            chunks: List of dicts with keys: chunk_id, text, section_name, content_type, metadata
            embeddings: List of numpy arrays (embeddings)
        """
        if not chunks or not embeddings:
            logger.warning("No chunks or embeddings to add")
            return
        
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunk count ({len(chunks)}) != embedding count ({len(embeddings)})")
        
        logger.info(f"Adding {len(chunks)} chunks to ChromaDB collection")
        
        # Prepare data for ChromaDB
        ids = [chunk["chunk_id"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = []
        embeddings_list = []
        
        for chunk, embedding in zip(chunks, embeddings):
            metadata = chunk.get("metadata", {})
            metadata["section_name"] = chunk.get("section_name", "Unknown")
            metadata["content_type"] = chunk.get("content_type", "text")
            
            metadatas.append(metadata)
            embeddings_list.append(embedding.tolist())
        
        # Add to collection
        try:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings_list,
                metadatas=metadatas,
            )
            logger.info(f"Successfully added {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Error adding chunks to ChromaDB: {e}")
            raise
    
    def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = 10,
        where_filter: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Query ChromaDB for similar chunks.
        
        Args:
            query_embedding: Query embedding (numpy array)
            n_results: Number of results to return
            where_filter: Optional metadata filter dict
            
        Returns:
            Dict with keys: ids, documents, metadatas, distances
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
            
            # Flatten results (query returns lists with one element for single query)
            if results["ids"]:
                return {
                    "ids": results["ids"][0],
                    "documents": results["documents"][0],
                    "metadatas": results["metadatas"][0],
                    "distances": results["distances"][0],
                }
            else:
                return {
                    "ids": [],
                    "documents": [],
                    "metadatas": [],
                    "distances": [],
                }
        except Exception as e:
            logger.error(f"Error querying ChromaDB: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "chunk_count": count,
        }
    
    def reset_collection(self) -> None:
        """Delete and recreate collection."""
        logger.warning("Resetting ChromaDB collection")
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection reset complete")
