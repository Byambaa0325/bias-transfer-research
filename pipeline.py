"""
Main pipeline script for end-to-end bias transfer research.

This script orchestrates the complete pipeline:
1. Dataset Generation
2. Model Evaluation
3. Drift Analysis
4. Visualization

Usage:
    python pipeline.py --models gemma2:9b llama3.1:8b --samples 100
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"ERROR: Error in {description}")
        sys.exit(1)
    
    print(f"OK: Completed: {description}")
    return result


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="End-to-end bias transfer research pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Steps:
  1. Dataset Generation: Creates bias-injected conversation dataset
  2. Model Evaluation: Evaluates models on the dataset
  3. Drift Analysis: Calculates semantic drift and stereotype leakage
  4. Visualization: Generates visualizations from results

Examples:
  # Run full pipeline with default settings
  python pipeline.py

  # Run with specific models and sample size
  python pipeline.py --models gemma2:9b llama3.1:8b --samples 100

  # Start from drift analysis and run visualization
  python pipeline.py --steps drift visualization

  # Start from drift analysis (skip dataset and evaluation)
  python pipeline.py --skip-steps dataset evaluation

  # Run only drift analysis
  python pipeline.py --steps drift

  # Skip visualization step
  python pipeline.py --skip-steps visualization
        """
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=None,
        help='Number of samples to process (default: all)'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=None,
        help='Specific models to evaluate (default: all configured models)'
    )
    
    parser.add_argument(
        '--steps',
        type=str,
        nargs='+',
        choices=['dataset', 'evaluation', 'drift', 'visualization'],
        default=['dataset', 'evaluation', 'drift', 'visualization'],
        help='Pipeline steps to run (default: all steps)'
    )
    
    parser.add_argument(
        '--skip-steps',
        type=str,
        nargs='+',
        choices=['dataset', 'evaluation', 'drift', 'visualization'],
        default=[],
        help='Pipeline steps to skip'
    )
    
    parser.add_argument(
        '--dataset-input',
        type=str,
        default=None,
        help='Input dataset CSV file (default: use EMGSD)'
    )
    
    parser.add_argument(
        '--bias-types',
        type=str,
        nargs='+',
        default=None,
        help='Specific bias types to use (default: all 8 types)'
    )
    
    return parser


def main():
    """Main pipeline execution."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Determine which steps to run
    steps_to_run = [s for s in args.steps if s not in args.skip_steps]
    
    if not steps_to_run:
        print("⚠️ No steps to run!")
        return
    
    print("\n" + "="*70)
    print("BIAS TRANSFER RESEARCH PIPELINE")
    print("="*70)
    print(f"Steps to run: {', '.join(steps_to_run)}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Step 1: Dataset Generation
    if 'dataset' in steps_to_run:
        cmd = ['python', '-m', 'dataset_generation.main']
        
        if args.samples:
            cmd.extend(['--samples', str(args.samples)])
        
        if args.bias_types:
            cmd.extend(['--bias-types'] + args.bias_types)
        
        if args.dataset_input:
            cmd.extend(['--input', args.dataset_input])
        
        run_command(cmd, "Step 1: Dataset Generation")
    
    # Step 2: Model Evaluation
    if 'evaluation' in steps_to_run:
        cmd = ['python', '-m', 'model_evaluations.main']
        
        if args.models:
            cmd.extend(['--models'] + args.models)
        
        if args.samples:
            cmd.extend(['--samples', str(args.samples)])
        
        run_command(cmd, "Step 2: Model Evaluation")
    
    # Step 3: Drift Analysis
    if 'drift' in steps_to_run:
        cmd = ['python', '-m', 'drift_analysis.main']
        
        if args.models:
            # Convert model IDs to format expected by drift analysis
            model_names = [m.replace(':', '_').replace('.', '_') for m in args.models]
            cmd.extend(['--models'] + model_names)
        
        run_command(cmd, "Step 3: Drift Analysis")
    
    # Step 4: Visualization
    if 'visualization' in steps_to_run:
        cmd = ['python', '-m', 'visualization.main']
        
        if args.models:
            # Convert to format expected by visualization
            model_names = [m.replace(':', '_').replace('.', '_') for m in args.models]
            if len(model_names) == 1:
                cmd.extend(['--model', model_names[0]])
        
        run_command(cmd, "Step 4: Visualization")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print("\nResults saved to:")
    print("  - Dataset: dataset_generation/data/")
    print("  - Evaluations: model_evaluations/results/")
    print("  - Drift Analysis: drift_analysis/results/")
    print("  - Visualizations: drift_analysis/results/{model}/{date}/")
    print("="*70)


if __name__ == '__main__':
    main()

