#!/usr/bin/env python3
"""
CLI entry point for dataset generation.

Provides command-line interface for generating multi-turn EMGSD datasets.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.bedrock_llm_service import BedrockLLMService
from dataset_generation.config import DatasetConfig
from dataset_generation.dataset_builder import DatasetBuilder
from dataset_generation.bias_generator import BiasGenerator
from dataset_generation.emgsd_processor import EMGSDProcessor
from dataset_generation.logger import set_logger_verbose


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description='Generate multi-turn EMGSD dataset with cognitive bias injection questions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate dataset for all stereotypes
  python -m dataset_generation.main

  # Limit to 100 samples
  python -m dataset_generation.main --samples 100

  # Filter by type
  python -m dataset_generation.main --stereotype-type profession

  # Use all 8 bias types
  python -m dataset_generation.main --all-biases

  # Resume from checkpoint
  python -m dataset_generation.main --resume data/checkpoint_multiturn_emgsd_*.csv
        """
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=None,
        help='Number of samples to process (default: all entries)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='Output directory for dataset files (default: data/)'
    )
    parser.add_argument(
        '--stereotype-type',
        type=str,
        choices=['profession', 'nationality', 'gender', 'religion'],
        default=None,
        help='Filter by stereotype type (default: all types)'
    )
    parser.add_argument(
        '--category',
        type=str,
        choices=['stereotype', 'neutral', 'unrelated'],
        default='stereotype',
        help='Filter by category (default: stereotype)'
    )
    parser.add_argument(
        '--all-biases',
        action='store_true',
        help='Use all available bias types (default: 4 most common)'
    )
    parser.add_argument(
        '--bias-types',
        type=str,
        nargs='+',
        help='Specify custom bias types (e.g., --bias-types availability_bias confirmation_bias)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume from saved checkpoint (path to CSV file)'
    )
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=50,
        help='Save checkpoint every N entries (default: 50)'
    )
    parser.add_argument(
        '--no-persona',
        action='store_true',
        help='Use psycholinguistic prompts instead of persona-based'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model ID for bias generation (default: Claude 3.5 Sonnet V2). '
             'Example: us.anthropic.claude-3-5-sonnet-20241022-v2:0'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of parallel workers (default: 8). Higher = faster but may hit rate limits'
    )
    parser.add_argument(
        '--rate-limit',
        type=float,
        default=5.0,
        help='Maximum requests per second (default: 5.0). Lower if hitting rate limits'
    )
    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help='Disable parallel processing (use sequential processing)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed debug output (default: quiet mode for long-running scripts)'
    )
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Create configuration
    config = DatasetConfig(
        output_dir=Path(args.output_dir),
        category_filter=args.category,
        stereotype_type_filter=args.stereotype_type,
        sample_limit=args.samples,
        use_persona_prompts=not args.no_persona,
        checkpoint_interval=args.checkpoint_interval,
        max_workers=args.workers,
        max_requests_per_second=args.rate_limit,
        use_parallel=not args.no_parallel
    )
    
    # Handle bias types
    if args.all_biases:
        config.bias_types = config.all_available_bias_types
    elif args.bias_types:
        config.bias_types = args.bias_types
    
    # Override model if specified
    if args.model:
        config.bias_generator_model = args.model
    
    # Set verbosity
    set_logger_verbose(args.verbose)
    
    # Validate configuration
    errors = config.validate()
    if errors:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        return 1
    
    # Print configuration
    print("="*70)
    print("MULTI-TURN EMGSD DATASET GENERATION")
    print("="*70)
    print(f"Bias Generator Model: {config.bias_generator_model}")
    print(f"Prompt Approach:      {'Persona-based' if config.use_persona_prompts else 'Psycholinguistic'}")
    print(f"Bias Types ({len(config.bias_types)}):     {', '.join(config.bias_types)}")
    print(f"Category Filter:      {config.category_filter}")
    if config.stereotype_type_filter:
        print(f"Stereotype Type:      {config.stereotype_type_filter}")
    if config.sample_limit:
        print(f"Sample Limit:         {config.sample_limit}")
    print(f"Parallel Processing:  {'Enabled' if config.use_parallel else 'Disabled'}")
    if config.use_parallel:
        print(f"  Workers:            {config.max_workers}")
        print(f"  Rate Limit:         {config.max_requests_per_second} req/s")
    print()
    print(f"Note: {len(config.all_available_bias_types)} total bias types available:")
    print(f"      {', '.join(config.all_available_bias_types)}")
    print("="*70)
    print()
    
    # Initialize services
    print("Initializing services...")
    try:
        llm_service = BedrockLLMService()
        print("  ✓ Bedrock service ready")
    except Exception as e:
        print(f"  ✗ Bedrock service failed: {e}")
        return 1
    
    # Initialize components
    bias_generator = BiasGenerator(
        llm_service=llm_service,
        bias_generator_model=config.bias_generator_model,
        use_persona_prompts=config.use_persona_prompts
    )
    
    emgsd_processor = EMGSDProcessor(
        transformed_path=config.emgsd_transformed_path,
        default_path=config.emgsd_default_path
    )
    
    dataset_builder = DatasetBuilder(
        config=config,
        bias_generator=bias_generator,
        emgsd_processor=emgsd_processor
    )
    
    # Build dataset
    resume_path = Path(args.resume) if args.resume else None
    summary = dataset_builder.build_dataset(resume_from=resume_path)
    
    # Print final statistics
    print("\n" + "="*70)
    print("DATASET GENERATION COMPLETE")
    print("="*70)
    stats = summary['statistics']
    print(f"Total entries:       {stats['total_entries']}")
    print(f"With target question: {stats['entries_with_target_question']}/{stats['total_entries']}")
    print(f"Duration:            {stats['duration_minutes']:.1f} minutes")
    
    print(f"\nBias Injection Questions Generated:")
    for bias_type in config.bias_types:
        generated = stats.get(f'{bias_type}_questions_generated', 0)
        refusals = stats.get(f'{bias_type}_refusals', 0)
        errors = stats.get(f'{bias_type}_errors', 0)
        print(f"  {bias_type:25} → {generated}/{stats['total_entries']} generated, "
              f"{refusals} refusals, {errors} errors")
    
    print("\n" + "="*70)
    print(f"Dataset saved to: {config.output_dir}")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

