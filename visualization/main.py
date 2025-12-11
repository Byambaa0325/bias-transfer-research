"""
Main entry point for drift analysis visualization.

This script processes drift analysis CSV files and generates visualizations
for each model. It operates as a separate pipeline step.
"""

import argparse
from pathlib import Path
from visualization.config import VisualizationConfig
from visualization.visualizer import DriftVisualizer


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate visualizations from drift analysis results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate visualizations for all models (latest date for each)
  python -m visualization.main

  # Generate visualizations for a specific model
  python -m visualization.main --model gemma2_9b

  # Generate visualizations for a specific model and date
  python -m visualization.main --model gemma2_9b --date 20251210

  # Use custom input/output directories
  python -m visualization.main --input-dir ../drift_analysis/results --output-dir ./output
        """
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        default='drift_analysis/results',
        help='Directory containing drift analysis results (default: drift_analysis/results)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for visualizations (default: same as input, per model/date)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Specific model to process (default: all models)'
    )
    
    parser.add_argument(
        '--date',
        type=str,
        default=None,
        help='Specific date to process, YYYYMMDD format (default: latest for each model)'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for saved figures (default: 300)'
    )
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Create configuration
    config = VisualizationConfig(
        drift_results_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        model_name=args.model,
        date=args.date,
        dpi=args.dpi
    )
    
    # Validate input directory
    if not config.drift_results_dir.exists():
        print(f"WARNING: Input directory does not exist: {config.drift_results_dir}")
        return
    
    # Create visualizer
    visualizer = DriftVisualizer(config)
    
    # Process models
    if config.model_name:
        # Process specific model
        visualizer.generate_all_visualizations(config.model_name, config.date)
    else:
        # Process all models
        visualizer.process_all_models()
    
    print("\n" + "="*70)
    print("Visualization generation complete!")
    print("="*70)


if __name__ == '__main__':
    main()

