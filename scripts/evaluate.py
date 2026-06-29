#!/usr/bin/env python3
"""Evaluation script for the hybrid RAG system."""

import sys
import argparse
import logging
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.orchestrator import get_orchestrator
from src.evaluation.evaluator import evaluate_on_qna_dataset
from src.config import LOG_LEVEL, QNA_DATA_PATH, QNA_DATA_MINI_PATH

# Setup logging
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate hybrid RAG system on QnA dataset"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["full", "mini"],
        default="mini",
        help="Which dataset to evaluate on"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Save evaluation results to JSON"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Select dataset
    if args.dataset == "full":
        qna_path = QNA_DATA_PATH
    else:
        qna_path = QNA_DATA_MINI_PATH
    
    if not Path(qna_path).exists():
        logger.error(f"Dataset not found: {qna_path}")
        logger.info("Run scripts/ingest.py first to download data")
        return
    
    # Get orchestrator
    logger.info("Initializing orchestrator...")
    orchestrator = get_orchestrator()
    
    # Run evaluation
    logger.info(f"Evaluating on {args.dataset} dataset: {qna_path}")
    metrics = evaluate_on_qna_dataset(orchestrator, qna_path)
    
    # Display results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    for metric_name, score in metrics.items():
        print(f"{metric_name:30s}: {score:.4f}")
    
    # Save if requested
    if args.output:
        output_data = {
            "dataset": args.dataset,
            "metrics": metrics,
        }
        
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✓ Results saved to {args.output}")


if __name__ == "__main__":
    main()
