"""
Generate High-Quality Prompts from EMGSD Dataset

This script implements the Transformation & Validation Pipeline:
1. Phase 1: Deterministic transformation (remove === markers)
2. Phase 2: LLM validation (quality control)
3. Phase 3: Integration (batch processing with threshold)

Usage:
    # Generate prompts for all stereotypes (with sample validation)
    python generate_prompts_from_emgsd.py

    # Generate with full validation (expensive)
    python generate_prompts_from_emgsd.py --full-validation

    # Test on a subset
    python generate_prompts_from_emgsd.py --test --sample-size 10
"""

import sys
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.emgsd_loader import load_emgsd
from data.prompt_generator import PromptGenerator
from core.bedrock_llm_service import BedrockLLMService


def main():
    parser = argparse.ArgumentParser(
        description="Generate high-quality prompts from EMGSD dataset"
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: process only 100 rows'
    )
    parser.add_argument(
        '--full-validation',
        action='store_true',
        help='Run full LLM validation on all prompts (expensive)'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=50,
        help='Sample size for validation (default: 50)'
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
        default='emgsd_with_prompts.csv',
        help='Output CSV filename (default: emgsd_with_prompts.csv)'
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("EMGSD PROMPT GENERATOR")
    print("Transformation & Validation Pipeline")
    print("="*70 + "\n")

    # Load EMGSD dataset
    print("Step 1: Loading EMGSD dataset...")
    loader = load_emgsd()

    # Get dataframe
    if args.test:
        print("  Test mode: Using first 100 rows")
        df = loader.df.head(100).copy()
    else:
        print(f"  Full dataset: {len(loader.df)} rows")
        df = loader.df.copy()

    # Initialize LLM service for validation
    print("\nStep 2: Initializing Bedrock LLM service...")
    try:
        llm_service = BedrockLLMService()
        print("  ✓ LLM service initialized")
    except Exception as e:
        print(f"  ⚠️  Could not initialize LLM service: {e}")
        print("  Continuing without validation (Phase 2 will be skipped)")
        llm_service = None

    # Initialize prompt generator
    generator = PromptGenerator(llm_service=llm_service)

    # Run pipeline
    print("\nStep 3: Running Transformation & Validation Pipeline...")
    df_processed, metadata = generator.generate_and_validate(
        df,
        validation_threshold=args.threshold,
        sample_size=args.sample_size,
        full_validation=args.full_validation
    )

    # Save results
    output_path = Path(__file__).parent / "results" / args.output
    output_path.parent.mkdir(exist_ok=True)

    df_processed.to_csv(output_path, index=False)
    print(f"✓ Results saved to: {output_path}")

    # Show examples
    print("\n" + "="*70)
    print("EXAMPLES: Generated Prompts")
    print("="*70)

    # Show examples by stereotype type
    for stereotype_type in ['profession', 'nationality', 'gender']:
        subset = df_processed[
            (df_processed['stereotype_type'] == stereotype_type) &
            (df_processed['category'] == 'stereotype')
        ].head(3)

        if len(subset) > 0:
            print(f"\n{stereotype_type.upper()} Stereotypes:")
            print("-" * 70)

            for idx, row in subset.iterrows():
                print(f"\nOriginal Text: {row['text']}")
                print(f"Target Word: {row['target_word']}")
                print(f"Generated Prompt: \"{row['final_prompt']}\"")
                print(f"Validation: {row['validation_status']}")

                # Show validation reason if available
                if 'validation_reason' in row and pd.notna(row['validation_reason']):
                    print(f"Reason: {row['validation_reason']}")

    # Statistics
    print("\n" + "="*70)
    print("STATISTICS")
    print("="*70)
    print(f"Total rows processed: {metadata['original_rows']}")
    print(f"Filtered (unrelated): {metadata['filtered_rows']}")
    print(f"Generated prompts: {metadata['generated_prompts']}")

    if metadata['sample_pass_rate'] is not None:
        print(f"Sample validation pass rate: {metadata['sample_pass_rate']:.1%}")

    if metadata['full_validation_performed']:
        print(f"Full validation: Performed")
        valid_count = (df_processed['validation_status'] == 'VALID').sum()
        invalid_count = (df_processed['validation_status'] == 'INVALID').sum()
        print(f"  - Valid: {valid_count}")
        print(f"  - Invalid: {invalid_count}")
    else:
        print(f"Full validation: Skipped (pass rate above threshold)")

    print(f"Final valid prompts: {metadata['final_valid_prompts']}")

    # Quality check: Show invalid examples if any
    if 'validation_status' in df_processed.columns:
        invalid = df_processed[df_processed['validation_status'] == 'INVALID']

        if len(invalid) > 0:
            print("\n" + "="*70)
            print(f"INVALID PROMPTS: {len(invalid)} found")
            print("="*70)

            for idx, row in invalid.head(5).iterrows():
                print(f"\nPrompt: \"{row['candidate_prompt']}\"")
                print(f"Reason: {row.get('validation_reason', 'N/A')}")
                print(f"Original: {row['text']}")

                if pd.notna(row.get('corrected_prompt')):
                    print(f"Corrected: \"{row['corrected_prompt']}\"")

    print("\n" + "="*70)
    print("Pipeline Complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    import pandas as pd
    main()
