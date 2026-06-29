"""Main ingestion pipeline orchestration."""

import logging
from pathlib import Path
from typing import List

from src.config import (
    DATA_DIR, DOCS_DIR, CHROMA_DB_DIR, BM25_INDEX_PATH, BM25_REGISTRY_PATH,
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, GITHUB_RAW_BASE,
)
from src.ingestion.metadata_extractor import extract_metadata_from_path
from src.ingestion.pdf_parser import parse_pdf_to_pages
from src.ingestion.chunker import chunk_document, Chunk
from src.indexing.embedder import Embedder
from src.indexing.chroma_store import ChromaDBStore
from src.indexing.bm25_index import BM25Index
from scripts.download_data import download_sec_10q_data

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def ingest_pipeline():
    """Run complete ingestion pipeline."""
    
    logger.info("=" * 80)
    logger.info("STARTING INGESTION PIPELINE")
    logger.info("=" * 80)
    
    # Step 1: Download data
    logger.info("\n[STEP 1] Downloading SEC 10-Q data from GitHub...")
    download_sec_10q_data(DATA_DIR, GITHUB_RAW_BASE)
    
    # Verify PDFs exist
    pdf_files = list(DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.error("No PDFs found in docs directory. Exiting.")
        return False
    
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    # Step 2: Initialize components
    logger.info("\n[STEP 2] Initializing indexing components...")
    embedder = Embedder(model_name=EMBEDDING_MODEL)
    chroma_store = ChromaDBStore(db_path=CHROMA_DB_DIR)
    bm25_index = BM25Index()
    
    all_chunks: List[Chunk] = []
    chunk_dicts: List[dict] = []
    embeddings_list = []
    
    # Step 3: Process PDFs
    logger.info(f"\n[STEP 3] Processing {len(pdf_files)} PDFs...")
    
    for pdf_path in sorted(pdf_files):
        logger.info(f"\n  Processing: {pdf_path.name}")
        
        # Extract metadata
        metadata = extract_metadata_from_path(pdf_path)
        if not metadata:
            logger.warning(f"  ✗ Could not extract metadata from {pdf_path.name}")
            continue
        
        logger.info(f"    Company: {metadata['company']}, Quarter: {metadata['filing_period']}")
        
        try:
            # Parse PDF
            pages = parse_pdf_to_pages(pdf_path)
            
            # Extract text and detect content type
            full_text = "\n\n--- PAGE BREAK ---\n\n".join(page.text for page in pages)
            
            # Determine content type (mixed if any page is table-heavy)
            content_types = [page.content_type for page in pages]
            if "table" in content_types:
                content_type = "mixed" if "text" in content_types else "table"
            else:
                content_type = "text"
            
            logger.info(f"    Content type: {content_type}, Pages: {len(pages)}")
            
            # Chunk document
            chunks = chunk_document(
                text=full_text,
                doc_metadata=metadata,
                content_type=content_type,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )
            
            all_chunks.extend(chunks)
            
            logger.info(f"    ✓ Created {len(chunks)} chunks")
        
        except Exception as e:
            logger.error(f"    ✗ Error processing {pdf_path.name}: {e}")
            continue
    
    if not all_chunks:
        logger.error("No chunks created. Exiting.")
        return False
    
    logger.info(f"\n✓ Total chunks created: {len(all_chunks)}")
    
    # Step 4: Generate embeddings
    logger.info("\n[STEP 4] Generating embeddings...")
    
    chunk_texts = [chunk.text for chunk in all_chunks]
    embeddings = embedder.embed_texts(chunk_texts, batch_size=32)
    
    logger.info(f"✓ Generated {len(embeddings)} embeddings")
    
    # Step 5: Index chunks
    logger.info("\n[STEP 5] Indexing chunks...")
    
    # Prepare chunk dicts for ChromaDB
    for chunk in all_chunks:
        chunk_dicts.append(chunk.to_dict())
    
    # Add to ChromaDB
    chroma_store.add_chunks(chunk_dicts, embeddings)
    
    # Add to BM25
    bm25_index.build(chunk_dicts)
    
    logger.info("✓ Chunks indexed in ChromaDB and BM25")
    
    # Step 6: Save indices
    logger.info("\n[STEP 6] Saving indices...")
    
    bm25_index.save(BM25_INDEX_PATH, BM25_REGISTRY_PATH)
    
    logger.info("✓ Indices saved")
    
    # Step 7: Print stats
    logger.info("\n[STEP 7] Collection statistics:")
    stats = chroma_store.get_collection_stats()
    logger.info(f"  ChromaDB chunks: {stats['chunk_count']}")
    logger.info(f"  BM25 chunks: {len(bm25_index.chunk_ids)}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ INGESTION PIPELINE COMPLETE")
    logger.info("=" * 80)
    
    return True


if __name__ == "__main__":
    success = ingest_pipeline()
    exit(0 if success else 1)
