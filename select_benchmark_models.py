#!/usr/bin/env python3
"""
Select representative models for benchmarking.

Includes:
1. The 4 models from your CSV files that are available on Bedrock
2. The 2 most representative models from each Bedrock model family
"""

from core.bedrock_client import BedrockModels
from collections import defaultdict

# Required models from your CSV files
REQUIRED_MODELS = {
    'Claude-3.5-Sonnet': {
        'name': 'Claude 3.5 Sonnet V2',
        'model_id': BedrockModels.CLAUDE_3_5_SONNET_V2,
        'family': 'Anthropic Claude',
        'reason': 'From your CSV files'
    },
    'Claude-3-Sonnet': {
        'name': 'Claude 3 Sonnet',
        'model_id': BedrockModels.CLAUDE_3_SONNET,
        'family': 'Anthropic Claude',
        'reason': 'From your CSV files'
    },
    'Llama-3-70B-T': {
        'name': 'Llama 3.1 70B',
        'model_id': BedrockModels.LLAMA_3_1_70B,
        'family': 'Meta Llama',
        'reason': 'From your CSV files (closest match)'
    },
    'Mistral-Large-2': {
        'name': 'Mistral Large',
        'model_id': BedrockModels.MISTRAL_LARGE,
        'family': 'Mistral',
        'reason': 'From your CSV files (closest match)'
    }
}

# All available models organized by family
ALL_MODELS_BY_FAMILY = {
    'Anthropic Claude': [
        {
            'name': 'Claude 3.5 Sonnet V2',
            'model_id': BedrockModels.CLAUDE_3_5_SONNET_V2,
            'size': 'large',
            'version': '3.5',
            'priority': 1  # Latest and best
        },
        {
            'name': 'Claude 3.5 Haiku',
            'model_id': BedrockModels.CLAUDE_3_5_HAIKU,
            'size': 'small',
            'version': '3.5',
            'priority': 2
        },
        {
            'name': 'Claude Sonnet 4.5',
            'model_id': BedrockModels.CLAUDE_SONNET_4_5,
            'size': 'large',
            'version': '4.5',
            'priority': 1  # Latest 4.x
        },
        {
            'name': 'Claude Opus 4',
            'model_id': BedrockModels.CLAUDE_OPUS_4,
            'size': 'largest',
            'version': '4',
            'priority': 2
        },
        {
            'name': 'Claude 3 Opus',
            'model_id': BedrockModels.CLAUDE_3_OPUS,
            'size': 'largest',
            'version': '3',
            'priority': 3
        },
        {
            'name': 'Claude 3 Sonnet',
            'model_id': BedrockModels.CLAUDE_3_SONNET,
            'size': 'large',
            'version': '3',
            'priority': 2
        },
        {
            'name': 'Claude 3 Haiku',
            'model_id': BedrockModels.CLAUDE_3_HAIKU,
            'size': 'small',
            'version': '3',
            'priority': 3
        },
        {
            'name': 'Claude Sonnet 4',
            'model_id': BedrockModels.CLAUDE_SONNET_4,
            'size': 'large',
            'version': '4',
            'priority': 3
        },
        {
            'name': 'Claude Haiku 4.5',
            'model_id': BedrockModels.CLAUDE_HAIKU_4_5,
            'size': 'small',
            'version': '4.5',
            'priority': 3
        },
    ],
    'Meta Llama': [
        {
            'name': 'Llama 3.3 70B',
            'model_id': BedrockModels.LLAMA_3_3_70B,
            'size': '70B',
            'version': '3.3',
            'priority': 1  # Latest large
        },
        {
            'name': 'Llama 3.2 90B',
            'model_id': BedrockModels.LLAMA_3_2_90B,
            'size': '90B',
            'version': '3.2',
            'priority': 1  # Largest
        },
        {
            'name': 'Llama 3.1 70B',
            'model_id': BedrockModels.LLAMA_3_1_70B,
            'size': '70B',
            'version': '3.1',
            'priority': 2
        },
        {
            'name': 'Llama 4 Scout',
            'model_id': BedrockModels.LLAMA_4_SCOUT,
            'size': '17B',
            'version': '4',
            'priority': 1  # Latest version
        },
        {
            'name': 'Llama 4 Maverick',
            'model_id': BedrockModels.LLAMA_4_MAVERICK,
            'size': '17B',
            'version': '4',
            'priority': 2
        },
        {
            'name': 'Llama 3.2 11B',
            'model_id': BedrockModels.LLAMA_3_2_11B,
            'size': '11B',
            'version': '3.2',
            'priority': 2
        },
        {
            'name': 'Llama 3.1 8B',
            'model_id': BedrockModels.LLAMA_3_1_8B,
            'size': '8B',
            'version': '3.1',
            'priority': 3
        },
        {
            'name': 'Llama 3.2 3B',
            'model_id': BedrockModels.LLAMA_3_2_3B,
            'size': '3B',
            'version': '3.2',
            'priority': 3
        },
        {
            'name': 'Llama 3.2 1B',
            'model_id': BedrockModels.LLAMA_3_2_1B,
            'size': '1B',
            'version': '3.2',
            'priority': 3
        },
    ],
    'Amazon Nova': [
        {
            'name': 'Nova Premier',
            'model_id': BedrockModels.NOVA_PREMIER,
            'size': 'largest',
            'version': 'v1',
            'priority': 1
        },
        {
            'name': 'Nova Pro',
            'model_id': BedrockModels.NOVA_PRO,
            'size': 'large',
            'version': 'v1',
            'priority': 1
        },
        {
            'name': 'Nova Lite',
            'model_id': BedrockModels.NOVA_LITE,
            'size': 'small',
            'version': 'v1',
            'priority': 2
        },
        {
            'name': 'Nova Micro',
            'model_id': BedrockModels.NOVA_MICRO,
            'size': 'smallest',
            'version': 'v1',
            'priority': 2
        },
    ],
    'Mistral': [
        {
            'name': 'Mistral Large',
            'model_id': BedrockModels.MISTRAL_LARGE,
            'size': 'large',
            'version': '2402',
            'priority': 1
        },
        {
            'name': 'Pixtral Large',
            'model_id': BedrockModels.PIXTRAL_LARGE,
            'size': 'large',
            'version': '2502',
            'priority': 1  # Latest and multimodal
        },
        {
            'name': 'Mixtral 8x7B',
            'model_id': BedrockModels.MIXTRAL_8X7B,
            'size': '56B',
            'version': 'v0',
            'priority': 2
        },
        {
            'name': 'Mistral Small',
            'model_id': BedrockModels.MISTRAL_SMALL,
            'size': 'small',
            'version': '2402',
            'priority': 2
        },
        {
            'name': 'Mistral 7B',
            'model_id': BedrockModels.MISTRAL_7B,
            'size': '7B',
            'version': 'v0',
            'priority': 3
        },
    ],
    'DeepSeek': [
        {
            'name': 'DeepSeek R1',
            'model_id': BedrockModels.DEEPSEEK_R1,
            'size': 'large',
            'version': 'v1',
            'priority': 1
        },
    ],
}


def select_representative_models(family_models, exclude_model_ids=None):
    """
    Select 2 most representative models from a family.
    
    Strategy:
    1. Prefer latest versions
    2. Prefer different sizes (large + small/medium)
    3. Exclude models already in required set
    """
    if exclude_model_ids is None:
        exclude_model_ids = set()
    
    # Filter out excluded models
    available = [m for m in family_models if m['model_id'] not in exclude_model_ids]
    
    if len(available) == 0:
        return []
    
    if len(available) <= 2:
        return available
    
    # Sort by priority (lower is better), then by version
    available.sort(key=lambda x: (x['priority'], x.get('version', '')))
    
    # Strategy: Pick one large and one small/medium if possible
    large_models = [m for m in available if 'large' in m['size'].lower() or 
                    any(x in m['size'] for x in ['70B', '90B', 'largest'])]
    small_models = [m for m in available if 'small' in m['size'].lower() or 
                    any(x in m['size'] for x in ['8B', '11B', '17B', 'smallest'])]
    
    selected = []
    
    # Pick best large model
    if large_models:
        selected.append(large_models[0])
        available = [m for m in available if m['model_id'] != large_models[0]['model_id']]
    
    # Pick best small/medium model (or another large if no small available)
    if small_models:
        selected.append(small_models[0])
    elif available:
        # If no small models, pick second best overall
        selected.append(available[0])
    
    return selected[:2]


def main():
    print("="*70)
    print("BENCHMARK MODEL SELECTION")
    print("="*70)
    print()
    
    # Collect required model IDs to exclude from family selection
    required_model_ids = {m['model_id'] for m in REQUIRED_MODELS.values()}
    
    # Select representative models from each family
    benchmark_models = {}
    
    # Add required models
    print("📋 Required Models (from your CSV files):")
    print("-"*70)
    for key, model in REQUIRED_MODELS.items():
        benchmark_models[model['model_id']] = {
            'name': model['name'],
            'family': model['family'],
            'reason': model['reason'],
            'source': 'required'
        }
        print(f"✓ {model['name']:30s} ({model['family']}) - {model['reason']}")
    
    print()
    print("="*70)
    print("Representative Models from Each Family")
    print("="*70)
    
    # Select from each family
    for family_name, family_models in ALL_MODELS_BY_FAMILY.items():
        selected = select_representative_models(family_models, required_model_ids)
        
        print(f"\n{family_name}:")
        print("-"*70)
        
        if not selected:
            print("  (All models already in required set)")
        else:
            for model in selected:
                if model['model_id'] not in benchmark_models:
                    benchmark_models[model['model_id']] = {
                        'name': model['name'],
                        'family': family_name,
                        'reason': f"Representative {model['size']} model",
                        'source': 'representative'
                    }
                    print(f"  ✓ {model['name']:30s} ({model['size']})")
                else:
                    print(f"  → {model['name']:30s} (already selected as required)")
    
    print()
    print("="*70)
    print("FINAL BENCHMARK SET")
    print("="*70)
    print()
    
    # Organize by family
    by_family = defaultdict(list)
    for model_id, info in benchmark_models.items():
        by_family[info['family']].append((model_id, info))
    
    total_count = 0
    for family_name in sorted(by_family.keys()):
        models = by_family[family_name]
        print(f"{family_name} ({len(models)} models):")
        for model_id, info in sorted(models, key=lambda x: x[1]['name']):
            total_count += 1
            source_marker = "⭐" if info['source'] == 'required' else "  "
            print(f"  {source_marker} {info['name']:35s} → {model_id}")
        print()
    
    print("="*70)
    print(f"Total: {total_count} models selected for benchmarking")
    print("="*70)
    
    # Generate Python code for easy use
    print()
    print("="*70)
    print("PYTHON CODE FOR BENCHMARK")
    print("="*70)
    print()
    print("# Benchmark models")
    print("BENCHMARK_MODELS = {")
    for model_id, info in sorted(benchmark_models.items(), key=lambda x: (x[1]['family'], x[1]['name'])):
        var_name = info['name'].upper().replace(' ', '_').replace('-', '_').replace('.', '_')
        print(f"    '{var_name}': '{model_id}',  # {info['name']} ({info['family']})")
    print("}")
    print()
    
    # Generate list format
    print("="*70)
    print("MODEL LIST (for easy copy-paste)")
    print("="*70)
    print()
    for model_id, info in sorted(benchmark_models.items(), key=lambda x: (x[1]['family'], x[1]['name'])):
        print(f"'{model_id}',  # {info['name']}")
    print()


if __name__ == "__main__":
    main()

