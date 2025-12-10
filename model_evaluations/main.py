#!/usr/bin/env python3
"""
CLI entry point for model evaluation.

Evaluates benchmark models on the multi-turn EMGSD dataset.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model_evaluations.config import EvaluationConfig, BENCHMARK_MODELS
from model_evaluations.evaluation_runner import EvaluationRunner


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description='Evaluate benchmark models on multi-turn EMGSD dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all models on full dataset
  python -m model_evaluations.main

  # Evaluate specific models
  python -m model_evaluations.main --models us.anthropic.claude-3-5-sonnet-20241022-v2:0

  # Limit to 100 entries
  python -m model_evaluations.main --samples 100

  # Evaluate specific bias types
  python -m model_evaluations.main --bias-types confirmation_bias availability_bias

  # Resume from latest checkpoint
  python -m model_evaluations.main --resume-from-checkpoint
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='dataset_generation/data',
        help='Path to dataset CSV file or directory (default: dataset_generation/data - auto-finds latest)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='model_evaluations/results',
        help='Output directory for results (default: model_evaluations/results)'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=None,
        help='Models to evaluate (default: all benchmark models)'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=None,
        help='Limit number of entries to evaluate (default: all)'
    )
    parser.add_argument(
        '--bias-types',
        type=str,
        nargs='+',
        default=None,
        help='Bias types to evaluate (default: all 8 types)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=5,
        help='Number of parallel workers (default: 5)'
    )
    parser.add_argument(
        '--rate-limit',
        type=float,
        default=3.0,
        help='Maximum requests per second (default: 3.0)'
    )
    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help='Disable parallel processing'
    )
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=10,
        help='Save checkpoint every N entries (default: 10)'
    )
    parser.add_argument(
        '--resume-from-checkpoint',
        action='store_true',
        help='Resume evaluation from latest checkpoint if available (default: False)'
    )
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Create configuration
    config = EvaluationConfig(
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output_dir),
        models=args.models,
        sample_limit=args.samples,
        bias_types=args.bias_types,
        max_workers=args.workers,
        max_requests_per_second=args.rate_limit,
        use_parallel=not args.no_parallel,
        checkpoint_interval=args.checkpoint_interval,
        resume_from_checkpoint=args.resume_from_checkpoint
    )
    
    # Print configuration
    print("="*70)
    print("MODEL EVALUATION CONFIGURATION")
    print("="*70)
    print(f"Dataset:              {config.dataset_path}")
    print(f"Output Directory:     {config.output_dir}")
    print(f"Models:               {len(config.models)}")
    for model in config.models:
        print(f"  • {model}")
    print(f"Bias Types:           {len(config.bias_types)}")
    print(f"  - {', '.join(config.bias_types)}")
    if config.sample_limit:
        print(f"Sample Limit:         {config.sample_limit}")
    print(f"Parallel Processing:  {config.use_parallel}")
    if config.use_parallel:
        print(f"  Workers:            {config.max_workers}")
        print(f"  Rate Limit:         {config.max_requests_per_second} req/s")
    print(f"Checkpoint Interval:  {config.checkpoint_interval}")
    print(f"Resume from Checkpoint: {config.resume_from_checkpoint}")
    print()
    
    # Run evaluation
    runner = EvaluationRunner(config)
    runner.run_evaluation()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

