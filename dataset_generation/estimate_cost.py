#!/usr/bin/env python3
"""
Estimate cost for dataset generation.

Calculates API calls and estimates cost based on AWS Bedrock pricing.
"""

# Configuration
VALID_ENTRIES = 17_128
BIAS_TYPES = 8
BIAS_GENERATOR_MODEL = "Nova Pro"  # Current model in config

# AWS Bedrock Pricing (as of 2024, approximate - verify with AWS pricing page)
# Prices per 1M tokens
PRICING = {
    "Claude 3.5 Sonnet V2": {
        "input": 3.00,   # $3.00 per 1M input tokens
        "output": 15.00  # $15.00 per 1M output tokens
    },
    "Claude 3.5 Haiku": {
        "input": 0.25,   # $0.25 per 1M input tokens
        "output": 1.25   # $1.25 per 1M output tokens
    },
    "Nova Pro": {
        "input": 0.80,   # $0.80 per 1M input tokens
        "output": 3.20   # $3.20 per 1M output tokens
    }
}

# Estimated tokens per API call
# Persona-based prompts (current default - use_persona_prompts=True):
# - Input: Persona template only, NO system prompt (~150 tokens)
# - Output: Generated bias question (~300 tokens)
# Psycholinguistic prompts (if use_persona_prompts=False):
# - Input: System prompt + detailed user prompt (~1400 tokens)
# - Output: Generated bias question (~300 tokens)
ESTIMATED_TOKENS_PER_CALL = {
    "input": 150,   # Persona-based: much shorter! (~594 chars / 4 = ~148 tokens)
    "output": 300
}

def estimate_cost(
    entries: int,
    bias_types: int,
    model: str = "Claude 3.5 Sonnet V2"
):
    """Estimate cost for dataset generation."""
    
    # Calculate API calls
    total_api_calls = entries * bias_types
    
    # Calculate tokens
    total_input_tokens = total_api_calls * ESTIMATED_TOKENS_PER_CALL["input"]
    total_output_tokens = total_api_calls * ESTIMATED_TOKENS_PER_CALL["output"]
    
    # Get pricing
    if model not in PRICING:
        print(f"⚠️  Warning: Pricing not available for {model}")
        print(f"   Using Claude 3.5 Sonnet V2 pricing as estimate")
        model = "Claude 3.5 Sonnet V2"
    
    input_price_per_million = PRICING[model]["input"]
    output_price_per_million = PRICING[model]["output"]
    
    # Calculate costs
    input_cost = (total_input_tokens / 1_000_000) * input_price_per_million
    output_cost = (total_output_tokens / 1_000_000) * output_price_per_million
    total_cost = input_cost + output_cost
    
    # Format numbers
    def format_num(n):
        return f"{n:,.0f}"
    
    def format_cost(n):
        return f"${n:,.2f}"
    
    print("="*70)
    print("DATASET GENERATION COST ESTIMATE")
    print("="*70)
    print()
    print(f"Configuration:")
    print(f"  Entries to process:     {format_num(entries)}")
    print(f"  Bias types per entry:   {bias_types}")
    print(f"  Model:                  {model}")
    print()
    print(f"API Calls:")
    print(f"  Total API calls:        {format_num(total_api_calls)}")
    print(f"  Calls per entry:        {bias_types}")
    print()
    print(f"Token Usage (estimated):")
    print(f"  Input tokens per call:  {format_num(ESTIMATED_TOKENS_PER_CALL['input'])}")
    print(f"  Output tokens per call: {format_num(ESTIMATED_TOKENS_PER_CALL['output'])}")
    print(f"  Total input tokens:     {format_num(total_input_tokens)} ({total_input_tokens/1_000_000:.2f}M)")
    print(f"  Total output tokens:    {format_num(total_output_tokens)} ({total_output_tokens/1_000_000:.2f}M)")
    print()
    print(f"Cost Estimate ({model}):")
    print(f"  Input cost:             {format_cost(input_cost)}")
    print(f"    ({total_input_tokens/1_000_000:.2f}M tokens × ${input_price_per_million}/M)")
    print(f"  Output cost:            {format_cost(output_cost)}")
    print(f"    ({total_output_tokens/1_000_000:.2f}M tokens × ${output_price_per_million}/M)")
    print(f"  {'─'*70}")
    print(f"  TOTAL ESTIMATED COST:   {format_cost(total_cost)}")
    print()
    print("="*70)
    print()
    print("Alternative Models (for comparison):")
    print()
    
    # Compare with other models
    for alt_model, prices in PRICING.items():
        if alt_model == model:
            continue
        alt_input_cost = (total_input_tokens / 1_000_000) * prices["input"]
        alt_output_cost = (total_output_tokens / 1_000_000) * prices["output"]
        alt_total = alt_input_cost + alt_output_cost
        savings = total_cost - alt_total
        print(f"  {alt_model:25} → {format_cost(alt_total)} "
              f"({'Save ' + format_cost(savings) if savings > 0 else 'More expensive'})")
    
    print()
    print("="*70)
    print("Notes:")
    print("  • Prices are approximate - verify with AWS Bedrock pricing page")
    print("  • Token estimates are based on typical bias injection prompts")
    print("  • Actual costs may vary based on:")
    print("    - Actual prompt lengths")
    print("    - Model response lengths")
    print("    - Rate limiting and retries")
    print("  • Consider using Claude 3.5 Haiku for ~90% cost savings")
    print("="*70)

if __name__ == "__main__":
    estimate_cost(VALID_ENTRIES, BIAS_TYPES, BIAS_GENERATOR_MODEL)

