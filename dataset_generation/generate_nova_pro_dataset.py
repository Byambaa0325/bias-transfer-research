#!/usr/bin/env python3
"""
Generate Multi-Turn EMGSD Dataset with Cognitive Bias Injection Questions

This script creates an enhanced version of the EMGSD dataset where each row includes:
- Original EMGSD fields
- Target question (extracted from === markers)
- Turn 1 bias injection questions for each cognitive bias type
- Metadata (bias type, definition source, model used, etc.)

The dataset is structured for multi-turn bias transfer experiments.
"""

import sys
from pathlib import Path
import json
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import argparse

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import project modules
from core.bedrock_llm_service import BedrockLLMService
from core.bedrock_client import BedrockModels
from core.bias_instructions import get_bias_instruction, get_all_bias_types, get_all_bias_types
from data.emgsd_loader import load_emgsd, EMGSDEntry


# Configuration
BIAS_GENERATOR_MODEL = BedrockModels.CLAUDE_3_5_SONNET_V2  # Always Claude for consistency
USE_PERSONA_PROMPTS = True  # Most effective approach

# Get all available bias types from bias_instructions
from core.bias_instructions import get_all_bias_types
ALL_BIAS_TYPES = get_all_bias_types()  # All 8 bias types available

# Default to the 4 most commonly used cognitive biases
# Set to ALL_BIAS_TYPES to use all 8 bias types
BIAS_TYPES = ["availability_bias", "confirmation_bias", "anchoring_bias", "framing_bias"]
# Or use all: BIAS_TYPES = ALL_BIAS_TYPES


def extract_target_question(stereotype_entry: EMGSDEntry) -> str:
    """
    Extract the target question from EMGSD entry.
    
    Priority:
    1. Use pre-transformed final_prompt if available
    2. Extract from text_with_marker using === markers
    
    Returns:
        Target question string
    """
    # Option 1: Use pre-transformed prompt
    if stereotype_entry.final_prompt:
        return stereotype_entry.final_prompt
    
    # Option 2: Extract from === markers
    if stereotype_entry.text_with_marker:
        parts = stereotype_entry.text_with_marker.split('===')
        if len(parts) >= 2:
            candidate = parts[0]
            if candidate.strip():
                return candidate
    
    # Fallback: return empty string (will be marked as error)
    return ""


def generate_bias_injection_question(
    llm_service: BedrockLLMService,
    target_question: str,
    bias_type: str,
    stereotype_entry: EMGSDEntry
) -> dict:
    """
    Generate a Turn 1 bias injection question for a given cognitive bias type.
    
    Args:
        llm_service: Bedrock LLM service
        target_question: The target question to prime for
        bias_type: Cognitive bias type (availability, confirmation, etc.)
        stereotype_entry: EMGSD entry with target trait
    
    Returns:
        Dictionary with:
        - question: The generated bias injection question
        - model: Model used for generation
        - bias_type: Cognitive bias type
        - bias_name: Name of the bias
        - bias_description: Description of the bias
        - bias_source: Source of bias definition
        - refusal_detected: Whether model refused
        - refusal_reason: Reason for refusal (if any)
        - error: Error message (if any)
    """
    # Get bias instruction
    instruction = get_bias_instruction(bias_type)
    if not instruction:
        return {
            'question': None,
            'model': BIAS_GENERATOR_MODEL,
            'bias_type': bias_type,
            'bias_name': bias_type,
            'bias_description': f'{bias_type} bias',
            'bias_source': 'unknown',
            'refusal_detected': False,
            'refusal_reason': None,
            'error': f'Bias instruction not found for {bias_type}'
        }
    
    bias_name = instruction.get('name', bias_type)
    bias_description = instruction.get('description', '')
    bias_source = instruction.get('source', 'bias_instructions.py')
    
    try:
        # Generate bias injection question
        bias_result = llm_service.inject_bias_llm(
            prompt=target_question,
            bias_type=bias_type,
            target_model_id=None,  # Not needed for question generation only
            stereotype_entry=stereotype_entry,
            existing_conversation=None,
            use_persona_prompts=USE_PERSONA_PROMPTS
        )
        
        # Check for refusals
        if bias_result.get('refusal_detected', False):
            return {
                'question': None,
                'model': BIAS_GENERATOR_MODEL,
                'bias_type': bias_type,
                'bias_name': bias_name,
                'bias_description': bias_description,
                'bias_source': bias_source,
                'refusal_detected': True,
                'refusal_reason': bias_result.get('refusal_reason', 'Unknown'),
                'error': 'Model refused to generate question'
            }
        
        # Extract the bias injection question
        bias_question = bias_result['conversation']['turn1_question']
        
        return {
            'question': bias_question,
            'model': BIAS_GENERATOR_MODEL,
            'bias_type': bias_type,
            'bias_name': bias_name,
            'bias_description': bias_description,
            'bias_source': bias_source,
            'refusal_detected': False,
            'refusal_reason': None,
            'error': None
        }
    
    except Exception as e:
        return {
            'question': None,
            'model': BIAS_GENERATOR_MODEL,
            'bias_type': bias_type,
            'bias_name': bias_name,
            'bias_description': bias_description,
            'bias_source': bias_source,
            'refusal_detected': False,
            'refusal_reason': None,
            'error': str(e)
        }


def process_entry(
    llm_service: BedrockLLMService,
    entry: EMGSDEntry,
    entry_index: int
) -> dict:
    """
    Process a single EMGSD entry and generate all bias injection questions.
    
    Returns:
        Dictionary with all fields for the dataset row
    """
    # Extract target question
    target_question = extract_target_question(entry)
    
    # Base row with original EMGSD fields
    row = {
        # Original EMGSD fields
        'emgsd_text': entry.text,
        'emgsd_text_with_marker': entry.text_with_marker,
        'emgsd_stereotype_type': entry.stereotype_type,
        'emgsd_category': entry.category,
        'emgsd_data_source': entry.data_source,
        'emgsd_label': entry.label,
        'emgsd_target_group': entry.target_group,
        'emgsd_trait': entry.trait,
        'emgsd_target_word': entry.target_word,
        
        # Target question
        'target_question': target_question,
        'target_question_source': 'final_prompt' if entry.final_prompt else 'extracted_from_markers',
        
        # Metadata
        'bias_generator_model': BIAS_GENERATOR_MODEL,
        'prompt_approach': 'persona-based' if USE_PERSONA_PROMPTS else 'psycholinguistic',
        'generation_timestamp': datetime.now().isoformat(),
    }
    
    # Generate bias injection questions for each bias type
    for bias_type in BIAS_TYPES:
        result = generate_bias_injection_question(
            llm_service=llm_service,
            target_question=target_question,
            bias_type=bias_type,
            stereotype_entry=entry
        )
        
        # Add columns for this bias type
        row[f'turn1_question_{bias_type}'] = result['question']
        row[f'bias_name_{bias_type}'] = result['bias_name']
        row[f'bias_description_{bias_type}'] = result['bias_description']
        row[f'bias_source_{bias_type}'] = result['bias_source']
        row[f'refusal_detected_{bias_type}'] = result['refusal_detected']
        row[f'refusal_reason_{bias_type}'] = result['refusal_reason']
        row[f'error_{bias_type}'] = result['error']
    
    return row


def main():
    """Main function to generate the dataset"""
    parser = argparse.ArgumentParser(
        description='Generate multi-turn EMGSD dataset with cognitive bias injection questions'
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
        '--resume',
        type=str,
        default=None,
        help='Resume from saved checkpoint (path to CSV file)'
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print("="*70)
    print("MULTI-TURN EMGSD DATASET GENERATION")
    print("="*70)
    print(f"Bias Generator Model: {BIAS_GENERATOR_MODEL}")
    print(f"Prompt Approach:      {'Persona-based' if USE_PERSONA_PROMPTS else 'Psycholinguistic'}")
    print(f"Bias Types ({len(BIAS_TYPES)}):     {', '.join(BIAS_TYPES)}")
    print(f"Category Filter:      {args.category}")
    if args.stereotype_type:
        print(f"Stereotype Type:      {args.stereotype_type}")
    if args.samples:
        print(f"Sample Limit:         {args.samples}")
    print()
    print(f"Note: {len(ALL_BIAS_TYPES)} total bias types available:")
    print(f"      {', '.join(ALL_BIAS_TYPES)}")
    print("="*70)
    print()
    
    # Initialize LLM service
    print("Initializing services...")
    try:
        llm_service = BedrockLLMService()
        print("  ✓ Bedrock service ready")
    except Exception as e:
        print(f"  ✗ Bedrock service failed: {e}")
        return 1
    
    # Load EMGSD dataset
    print("Loading EMGSD dataset...")
    transformed_path = Path('data/emgsd_with_prompts.csv')
    if transformed_path.exists():
        emgsd = load_emgsd(dataset_path=str(transformed_path))
        print("  ✓ Using transformed EMGSD prompts")
    else:
        emgsd = load_emgsd()
        print("  ✓ EMGSD dataset loaded")
        print("  ⚠️  Prompts will be extracted from === markers")
    
    # Get entries
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        existing_df = pd.read_csv(args.resume)
        processed_texts = set(existing_df['emgsd_text'].tolist())
        print(f"  Found {len(processed_texts)} existing entries")
        
        # Get all entries
        if args.category == 'stereotype':
            entries = emgsd.get_stereotypes(stereotype_type=args.stereotype_type)
        elif args.category == 'neutral':
            entries = emgsd.get_neutral(stereotype_type=args.stereotype_type)
        else:
            entries = emgsd.get_unrelated(stereotype_type=args.stereotype_type)
        
        # Filter out already processed
        entries = [e for e in entries if e.text not in processed_texts]
        results = existing_df.to_dict('records')
        start_index = len(results)
    else:
        # Get entries
        if args.category == 'stereotype':
            entries = emgsd.get_stereotypes(stereotype_type=args.stereotype_type)
        elif args.category == 'neutral':
            entries = emgsd.get_neutral(stereotype_type=args.stereotype_type)
        else:
            entries = emgsd.get_unrelated(stereotype_type=args.stereotype_type)
        
        results = []
        start_index = 0
    
    # Limit samples if specified
    if args.samples:
        entries = entries[:args.samples]
    
    print(f"\nProcessing {len(entries)} entries...")
    print(f"Estimated time: {len(entries) * len(BIAS_TYPES) * 0.5} - {len(entries) * len(BIAS_TYPES) * 1} minutes")
    print(f"Estimated API calls: {len(entries) * len(BIAS_TYPES)} (one per bias type per entry)\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Process entries
    start_time = datetime.now()
    
    for i, entry in enumerate(tqdm(entries, desc="Processing"), start=start_index):
        try:
            row = process_entry(
                llm_service=llm_service,
                entry=entry,
                entry_index=i
            )
            results.append(row)
        except Exception as e:
            print(f"\n  ❌ Error processing entry {i}: {str(e)[:100]}")
            # Add error row
            error_row = {
                'emgsd_text': entry.text,
                'emgsd_text_with_marker': entry.text_with_marker,
                'emgsd_stereotype_type': entry.stereotype_type,
                'emgsd_category': entry.category,
                'emgsd_data_source': entry.data_source,
                'emgsd_label': entry.label,
                'target_question': None,
                'error': str(e)
            }
            results.append(error_row)
        
        # Save checkpoint every 50 entries
        if (i + 1) % 50 == 0:
            checkpoint_df = pd.DataFrame(results)
            checkpoint_path = output_dir / f'checkpoint_multiturn_emgsd_{timestamp}.csv'
            checkpoint_df.to_csv(checkpoint_path, index=False)
            
            # Show progress
            successful = sum(1 for r in results if r.get('target_question'))
            print(f"\n  Progress: {i+1}/{len(entries)} ({((i+1)/len(entries)*100):.1f}%)")
            print(f"  Successful: {successful}/{i+1} ({successful/(i+1)*100:.1f}%)")
            
            # Count refusals
            total_refusals = 0
            for bias_type in BIAS_TYPES:
                refusals = sum(1 for r in results if r.get(f'refusal_detected_{bias_type}', False))
                total_refusals += refusals
            if total_refusals > 0:
                print(f"  Refusals: {total_refusals} across all bias types")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60
    
    # Save final dataset
    print("\n" + "="*70)
    print("Saving dataset...")
    print("="*70)
    
    results_df = pd.DataFrame(results)
    
    # Save CSV
    csv_path = output_dir / f'multiturn_emgsd_dataset_{timestamp}.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"  ✓ CSV: {csv_path}")
    
    # Save JSON (full data)
    json_path = output_dir / f'multiturn_emgsd_dataset_{timestamp}.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  ✓ JSON: {json_path}")
    
    # Save summary
    summary = {
        'configuration': {
            'bias_generator_model': BIAS_GENERATOR_MODEL,
            'prompt_approach': 'persona-based' if USE_PERSONA_PROMPTS else 'psycholinguistic',
            'bias_types': BIAS_TYPES,
            'category_filter': args.category,
            'stereotype_type_filter': args.stereotype_type
        },
        'statistics': {
            'total_entries': len(results),
            'entries_with_target_question': sum(1 for r in results if r.get('target_question')),
            'duration_minutes': round(duration, 2)
        }
    }
    
    # Count refusals and errors per bias type
    for bias_type in BIAS_TYPES:
        refusals = sum(1 for r in results if r.get(f'refusal_detected_{bias_type}', False))
        errors = sum(1 for r in results if r.get(f'error_{bias_type}') and not r.get(f'refusal_detected_{bias_type}', False))
        questions_generated = sum(1 for r in results if r.get(f'turn1_question_{bias_type}'))
        
        summary['statistics'][f'{bias_type}_refusals'] = refusals
        summary['statistics'][f'{bias_type}_errors'] = errors
        summary['statistics'][f'{bias_type}_questions_generated'] = questions_generated
    
    summary_path = output_dir / f'multiturn_emgsd_dataset_{timestamp}_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Summary: {summary_path}")
    
    # Print final statistics
    print("\n" + "="*70)
    print("DATASET GENERATION COMPLETE")
    print("="*70)
    print(f"Total entries:       {len(results)}")
    print(f"With target question: {summary['statistics']['entries_with_target_question']}/{len(results)}")
    print(f"Duration:            {duration:.1f} minutes")
    
    print(f"\nBias Injection Questions Generated:")
    for bias_type in BIAS_TYPES:
        generated = summary['statistics'][f'{bias_type}_questions_generated']
        refusals = summary['statistics'][f'{bias_type}_refusals']
        errors = summary['statistics'][f'{bias_type}_errors']
        print(f"  {bias_type:15} → {generated}/{len(results)} generated, {refusals} refusals, {errors} errors")
    
    print("\n" + "="*70)
    print(f"Dataset saved to: {output_dir}")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
