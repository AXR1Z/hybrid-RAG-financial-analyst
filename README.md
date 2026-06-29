# Hybrid RAG Financial Analyst

A production-ready hybrid Retrieval-Augmented Generation (RAG) system for analyzing SEC 10-Q financial filings using advanced retrieval techniques, LangGraph orchestration, and Groq's fast LLMs.

## Features

- **Hybrid Retrieval**: Combines vector similarity (ChromaDB) and sparse BM25 search with Reciprocal Rank Fusion (RRF)
- **Multi-Agent Orchestration**: LangGraph-based pipeline with specialized agents for query analysis, retrieval, web search, and answer generation
- **Section-Aware Chunking**: Respects SEC 10-Q document boundaries (ITEM sections, notes, financial statements)
- **Metadata Filtering**: Filter results by company, quarter, year
- **Web Search Integration**: Conditional web search for questions requiring current information
- **RAGAS Evaluation**: Comprehensive evaluation metrics (faithfulness, answer relevancy, context precision/recall)
- **CLI Interface**: Both interactive and batch query modes

## Architecture

```
Query
  ↓
[Query Analyzer] → Extract metadata, determine web search need
  ↓
[RAG Agent] → Vector + BM25 hybrid retrieval with RRF fusion
  ↓
[Web Search] ← Conditional: if needs_web_search=True
  ↓
[Context Assembler] → Format retrieved chunks and web results
  ↓
[Answer Agent] → Generate answer using Groq LLM
  ↓
Answer + Sources
```

## Setup

### 1. Prerequisites

- Python 3.10+
- Groq API key (free from [console.groq.com](https://console.groq.com))

### 2. Installation

```bash
git clone <repo-url>
cd hybrid-RAG-financial-analyst

# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows
# or
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Create `.env` file in project root:

```env
GROQ_API_KEY=your_api_key_here

# Optional: Customize models and parameters
QUERY_ANALYZER_MODEL=llama-3.1-8b-instant
ANSWER_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Optional: Retrieval parameters
VECTOR_RETRIEVAL_TOP_K=20
BM25_RETRIEVAL_TOP_K=20
HYBRID_FUSION_TOP_K=10

# Optional: Chunking
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Optional: Logging
LOG_LEVEL=INFO
```

### 4. Data Ingestion

Download SEC 10-Q PDFs and build indexes:

```bash
python scripts/ingest.py
```

This will:
1. Download 20 SEC 10-Q PDFs (5 companies × 2022 Q3 + 2023 Q1-Q3)
2. Parse PDFs and extract text with content type detection
3. Perform section-aware chunking
4. Generate embeddings using sentence-transformers
5. Index in ChromaDB (vector) and BM25 (sparse)
6. Save indexes for reuse

Expected output:
```
✓ Downloaded 20 PDFs + 2 CSV files
✓ Processed 20 documents
✓ Created XXX chunks
✓ Indexed in ChromaDB and BM25
```

## Usage

### 1. Interactive Query

```bash
python scripts/query.py
```

Then type questions:
```
Q: What were Apple's total revenues in Q3 2023?
A: [answer with sources]

Q: Compare revenue growth across semiconductor companies
A: [answer with sources]
```

### 2. Single Query

```bash
python scripts/query.py "What were Apple's operating expenses in Q1 2023?"

# With filters
python scripts/query.py \
  "What were operating expenses?" \
  --company AAPL \
  --year 2023

# Save results
python scripts/query.py "Your question?" --output results.json
```

### 3. Programmatic Usage

```python
from src.app import get_rag_system

rag = get_rag_system()

result = rag.query(
    question="What was Apple's net income in Q3 2023?",
    company="AAPL",
    year=2023,
)

print(result["answer"])
print("Sources:", result["sources"])
```

### 4. Evaluation

Evaluate on QnA dataset:

```bash
# Evaluate on mini dataset (5 samples)
python scripts/evaluate.py --dataset mini

# Evaluate on full dataset (195 samples)
python scripts/evaluate.py --dataset full --output eval_results.json

# Save detailed results
python scripts/evaluate.py --dataset mini --output results.json --verbose
```

## Project Structure

```
hybrid-RAG-financial-analyst/
├── data/
│   ├── docs/                    # Downloaded SEC 10-Q PDFs
│   ├── qna_data.csv            # Full Q&A dataset (195 pairs)
│   ├── qna_data_mini.csv       # Mini Q&A dataset (5 pairs)
│   ├── bm25_index.pkl          # Serialized BM25 index
│   └── bm25_registry.json       # BM25 chunk registry
├── chroma_db/                   # ChromaDB persistent storage
├── src/
│   ├── config.py               # Centralized configuration
│   ├── app.py                  # Main RAG interface
│   ├── ingestion/
│   │   ├── metadata_extractor.py
│   │   ├── pdf_parser.py
│   │   └── chunker.py
│   ├── indexing/
│   │   ├── embedder.py
│   │   ├── chroma_store.py
│   │   └── bm25_index.py
│   ├── retrieval/
│   │   ├── vector_retriever.py
│   │   ├── bm25_retriever.py
│   │   └── hybrid_ranker.py
│   ├── agents/
│   │   ├── state.py            # LangGraph state definition
│   │   ├── query_analyzer.py   # Query analysis node
│   │   ├── rag_agent.py        # Retrieval node
│   │   ├── web_search_agent.py # Web search node
│   │   ├── context_assembler.py # Context formatting node
│   │   ├── answer_agent.py     # Answer generation node
│   │   └── orchestrator.py     # LangGraph orchestrator
│   └── evaluation/
│       └── evaluator.py        # RAGAS-based evaluation
├── scripts/
│   ├── download_data.py        # Data download utility
│   ├── ingest.py               # Full ingestion pipeline
│   ├── query.py                # CLI query interface
│   └── evaluate.py             # Evaluation script
├── requirements.txt
├── .env.example
└── README.md
```

## Key Components

### Retrieval Pipeline

**Vector Retrieval**
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- Database: ChromaDB with HNSW index
- Top-K: 20 results

**BM25 Retrieval**
- Tokenization: Whitespace + lowercase
- Algorithm: BM25Okapi
- Top-K: 20 results

**Hybrid Fusion (RRF)**
- Combines both retrievers using Reciprocal Rank Fusion
- Formula: `score(d) = Σ 1/(k + rank_i)` where k=60
- Final Top-K: 10 results

### Chunking Strategy

1. **Section Splitting**: Regex-based split on SEC section patterns (11 patterns)
   - ITEM 1-6
   - Management's Discussion & Analysis
   - Financial Statements
   - Notes to Financials
   - Risk Factors

2. **Recursive Chunking**: Within sections, split using:
   - Chunk size: 1000 characters
   - Overlap: 200 characters
   - Min chunk size: 100 characters

3. **Content Type Detection**: Heuristic-based
   - >30% table-like content → "table"
   - 10-30% → "mixed"
   - <10% → "text"

### LangGraph Orchestration

5-node pipeline:
1. **Query Analyzer**: Extract metadata, determine web search need
2. **RAG Agent**: Perform hybrid retrieval
3. **Web Search**: Conditional web search (if needed)
4. **Context Assembler**: Format retrieved context
5. **Answer Agent**: Generate answer using Groq

Conditional routing: Web search only if `needs_web_search=True`

## Model Selection

| Component | Model | Rationale |
|-----------|-------|-----------|
| Query Analysis | llama-3.1-8b-instant | Fast, low-latency for metadata extraction |
| Answer Generation | llama-3.3-70b-versatile | Large context window, excellent reasoning |
| Embeddings | all-MiniLM-L6-v2 | Lightweight (80MB), 384-dim, good semantic understanding |

## Performance

- **Indexing**: ~2-3 minutes for 20 PDFs (1000+ chunks)
- **Query Latency**: ~3-5 seconds per query (Groq API + retrieval)
- **Retrieval Quality**: Tuned for financial domain with section-aware chunking
- **Memory**: ~500MB ChromaDB + 100MB BM25 index

## Evaluation Metrics (RAGAS)

- **Faithfulness**: Does the answer rely on retrieved context?
- **Answer Relevancy**: Is the answer relevant to the question?
- **Context Precision**: Is retrieved context noise-free?
- **Context Recall**: Was all relevant context retrieved?

## Advanced Usage

### Custom Metadata Filtering

```python
from src.retrieval.vector_retriever import VectorRetriever
from src.indexing.chroma_store import ChromaDBStore
from src.indexing.embedder import Embedder

chroma = ChromaDBStore(db_path="./chroma_db")
embedder = Embedder()
retriever = VectorRetriever(chroma, embedder)

# Filter by company and year
results = retriever.retrieve(
    query="operating expenses",
    where_filter={"company": "AAPL", "year": 2023},
    top_k=10,
)
```

### Batch Processing

```python
from src.app import get_rag_system

rag = get_rag_system()

questions = [
    "What were total revenues?",
    "What were operating expenses?",
    "What were net earnings?",
]

results = rag.batch_query(questions, company="AAPL", year=2023)
```

### Direct Orchestrator Access

```python
from src.agents.orchestrator import get_orchestrator

orchestrator = get_orchestrator()

state = orchestrator.invoke(
    query="Your question?",
    metadata_filters={"company": "MSFT"},
)

print("Answer:", state["answer"])
print("RAG Results:", len(state["rag_results"]))
print("Web Results:", len(state["web_results"]))
```

## Troubleshooting

### API Key Error
```
ValueError: GROQ_API_KEY not set
```
→ Add `GROQ_API_KEY=...` to `.env` file

### ChromaDB Connection Error
```
chromadb.errors.InvalidCollectionError
```
→ Delete `chroma_db/` directory and re-run `scripts/ingest.py`

### Out of Memory
- Reduce `CHUNK_SIZE` (e.g., 500)
- Reduce `VECTOR_RETRIEVAL_TOP_K` (e.g., 10)
- Run on GPU: Set `CUDA_VISIBLE_DEVICES=0`

### Slow Queries
- Check Groq API status at console.groq.com
- Use smaller model: `llama-3.1-8b-instant`
- Reduce context size: Lower `HYBRID_FUSION_TOP_K`

## Contributing

1. Fork repository
2. Create feature branch
3. Submit pull request

## License

MIT License

## Citations

- SEC 10-Q Dataset: [docugami/KG-RAG-datasets](https://github.com/docugami/KG-RAG-datasets)
- LangGraph: LangChain
- RAGAS: Evaluation framework
- Groq: LLM inference
- ChromaDB: Vector database