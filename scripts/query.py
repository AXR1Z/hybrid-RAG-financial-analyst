#!/usr/bin/env python3
"""CLI interface for the hybrid RAG financial analyst."""

import sys
import argparse
import logging
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.orchestrator import get_orchestrator
from src.config import LOG_LEVEL

# Setup logging
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Hybrid RAG Financial Analyst - Query SEC 10-Q filings"
    )
    
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Query to ask (if not provided, enters interactive mode)"
    )
    
    parser.add_argument(
        "--company",
        type=str,
        help="Filter by company ticker (AAPL, AMZN, INTC, MSFT, NVDA)"
    )
    
    parser.add_argument(
        "--quarter",
        type=str,
        help="Filter by quarter (Q1, Q2, Q3, Q4)"
    )
    
    parser.add_argument(
        "--year",
        type=int,
        help="Filter by year (e.g., 2023)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Save results to JSON file"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Get orchestrator
    orchestrator = get_orchestrator()
    
    # Prepare metadata filters
    metadata_filters = {}
    if args.company:
        metadata_filters["company"] = args.company
    if args.quarter:
        metadata_filters["quarter"] = args.quarter
    if args.year:
        metadata_filters["year"] = args.year
    
    metadata_filters = metadata_filters if metadata_filters else None
    
    if args.query:
        # Single query mode
        process_query(orchestrator, args.query, metadata_filters, args.output, args.verbose)
    else:
        # Interactive mode
        interactive_mode(orchestrator, metadata_filters, args.verbose)


def process_query(orchestrator, query, metadata_filters, output_file, verbose):
    """Process a single query."""
    try:
        logger.info(f"Query: {query}")
        
        result = orchestrator.invoke(query, metadata_filters=metadata_filters)
        
        # Display result
        print("\n" + "=" * 80)
        print("ANSWER")
        print("=" * 80)
        print(result["answer"])
        
        print("\n" + "=" * 80)
        print("SOURCES")
        print("=" * 80)
        for i, source in enumerate(result["sources"], 1):
            print(f"{i}. {source}")
        
        if verbose:
            print("\n" + "=" * 80)
            print("RETRIEVED CHUNKS")
            print("=" * 80)
            if result.get("rag_results"):
                for i, chunk in enumerate(result["rag_results"], 1):
                    print(f"\n[{i}] Score: {chunk.get('rrf_score', 0):.3f}")
                    print(f"Text: {chunk.get('text', '')[:200]}...")
                    print(f"Section: {chunk.get('section_name', 'Unknown')}")
        
        # Save to file if requested
        if output_file:
            output_data = {
                "query": query,
                "answer": result["answer"],
                "sources": result["sources"],
                "num_chunks": len(result.get("rag_results", [])),
                "num_web_results": len(result.get("web_results", [])),
            }
            
            with open(output_file, "w") as f:
                json.dump(output_data, f, indent=2)
            
            print(f"\n✓ Results saved to {output_file}")
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise


def interactive_mode(orchestrator, metadata_filters, verbose):
    """Interactive query mode."""
    print("\n" + "=" * 80)
    print("HYBRID RAG FINANCIAL ANALYST")
    print("=" * 80)
    print("Ask questions about SEC 10-Q filings (type 'quit' to exit)\n")
    
    while True:
        try:
            query = input("Q: ").strip()
            
            if query.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            result = orchestrator.invoke(query, metadata_filters=metadata_filters)
            
            print("\nA:", result["answer"])
            
            if result["sources"]:
                print("\nSources:")
                for i, source in enumerate(result["sources"], 1):
                    print(f"  {i}. {source}")
            
            print()
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")


if __name__ == "__main__":
    main()
