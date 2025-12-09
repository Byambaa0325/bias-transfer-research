"""
Transform EMGSD Dataset with High-Quality Prompts

This script runs the Prompt Transformation & Validation Pipeline on the EMGSD dataset.

Usage:
    python transform_emgsd_prompts.py [--no-validation] [--sample-size 50]

Options:
    --no-validation: Skip LLM validation (faster, deterministic only)
    --sample-size N: Number of samples to validate (default: 50)
    --threshold P: Pass rate threshold for skipping full validation (default: 0.95)
    --output PATH: Output path for transformed CSV (default: data/emgsd_with_prompts.csv)
"""

import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.emgsd_loader import load_emgsd
from data.prompt_transformer import PromptTransformer


def main():
    parser = argparse.ArgumentParser(
        description="Transform EMGSD dataset with high-quality prompts"
    )
    parser.add_argument(
        '--no-validation',
        action='store_true',
        help='Skip LLM validation (deterministic only)'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=50,
        help='Number of samples to validate (default: 50)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.95,
        help='Pass rate threshold for skipping full validation (default: 0.95)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/emgsd_with_prompts.csv',
        help='Output path for transformed CSV'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("EMGSD PROMPT TRANSFORMATION SCRIPT")
    print("="*70)
    print()
    print("Configuration:")
    print(f"  LLM Validation:  {'Disabled' if args.no_validation else 'Enabled'}")
    print(f"  Sample Size:     {args.sample_size}")
    print(f"  Pass Threshold:  {args.threshold:.0%}")
    print(f"  Output Path:     {args.output}")
    print()
    
    # Step 1: Load EMGSD
    print("Step 1: Loading EMGSD dataset...")
    try:
        emgsd = load_emgsd()
        df = emgsd.df
        print(f"✓ Loaded {len(df)} entries")
        print()
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("\nPlease update the EMGSD path in data/emgsd_loader.py")
        return 1
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return 1
    
    # Step 2: Create transformer
    print("Step 2: Initializing transformer...")
    use_llm = not args.no_validation
    transformer = PromptTransformer(use_llm_validation=use_llm)
    print()
    
    # Step 3: Transform dataset
    print("Step 3: Transforming dataset...")
    try:
        df_transformed = transformer.transform_dataset(
            df,
            sample_validate=args.sample_size,
            full_validate_threshold=args.threshold
        )
    except Exception as e:
        print(f"✗ Transformation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 4: Save results
    print("\nStep 4: Saving results...")
    try:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        transformer.save_transformed_dataset(df_transformed, str(output_path))
        print()
    except Exception as e:
        print(f"✗ Save failed: {e}")
        return 1
    
    # Step 5: Summary & usage instructions
    print("="*70)
    print("✓ TRANSFORMATION COMPLETE")
    print("="*70)
    print()
    print("Next Steps:")
    print()
    print("1. Update your experiment to use the transformed prompts:")
    print()
    print("   from data.emgsd_loader import load_emgsd")
    print()
    print("   # Load with transformed prompts")
    print(f"   emgsd = load_emgsd(dataset_path='{args.output}')")
    print()
    print("   # Use the final_prompt instead of generic questions")
    print("   for entry in emgsd.get_stereotypes(limit=10):")
    print("       if entry.final_prompt:")
    print("           print(f'Prompt: {entry.final_prompt}')")
    print("           print(f'Expected: {entry.target_word}')")
    print()
    print("2. Run the demonstration notebook with transformed prompts")
    print()
    print("3. Compare results with generic questions to measure improvement")
    print()
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

