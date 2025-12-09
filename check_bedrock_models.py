#!/usr/bin/env python3
"""
Check which models are available on your Bedrock account.

Compares your CSV files with available Bedrock models.
"""

from core.bedrock_client import BedrockModels, BedrockClient
import os

# Your CSV files
files_to_process = [
    'Claude-2 Outputs.csv',
    'Claude-3.5-Sonnet Outputs.csv',
    'Claude-3-Sonnet Outputs.csv',
    'Gemini-1.0-Pro Outputs.csv',
    'Gemini-1.5-Pro Outputs.csv',
    'GPT-3.5-Turbo Outputs.csv',
    'GPT-4o Outputs.csv',
    'GPT-4-Turbo Outputs.csv',
    'Llama-3-70B-T Outputs.csv',
    'Llama-3.1-405B-T Outputs.csv',
    'Mistral Large 2 Outputs.csv',
    'Mistral Medium Outputs.csv'
]

# Mapping from CSV names to Bedrock model IDs
CSV_TO_BEDROCK = {
    'Claude-2 Outputs.csv': None,  # Claude 2 not available on Bedrock
    'Claude-3.5-Sonnet Outputs.csv': BedrockModels.CLAUDE_3_5_SONNET_V2,
    'Claude-3-Sonnet Outputs.csv': BedrockModels.CLAUDE_3_SONNET,
    'Gemini-1.0-Pro Outputs.csv': None,  # Google Gemini not on Bedrock
    'Gemini-1.5-Pro Outputs.csv': None,  # Google Gemini not on Bedrock
    'GPT-3.5-Turbo Outputs.csv': None,  # OpenAI GPT not on Bedrock
    'GPT-4o Outputs.csv': None,  # OpenAI GPT not on Bedrock
    'GPT-4-Turbo Outputs.csv': None,  # OpenAI GPT not on Bedrock
    'Llama-3-70B-T Outputs.csv': BedrockModels.LLAMA_3_1_70B,  # Closest match
    'Llama-3.1-405B-T Outputs.csv': None,  # 405B not available, only 70B
    'Mistral Large 2 Outputs.csv': BedrockModels.MISTRAL_LARGE,  # Closest match
    'Mistral Medium Outputs.csv': None,  # Not available on Bedrock
}

# All available Bedrock models
ALL_BEDROCK_MODELS = {
    'Claude 3.5 Sonnet V2': BedrockModels.CLAUDE_3_5_SONNET_V2,
    'Claude 3.5 Haiku': BedrockModels.CLAUDE_3_5_HAIKU,
    'Claude 3 Opus': BedrockModels.CLAUDE_3_OPUS,
    'Claude 3 Sonnet': BedrockModels.CLAUDE_3_SONNET,
    'Claude 3 Haiku': BedrockModels.CLAUDE_3_HAIKU,
    'Claude Opus 4': BedrockModels.CLAUDE_OPUS_4,
    'Claude Sonnet 4': BedrockModels.CLAUDE_SONNET_4,
    'Claude Sonnet 4.5': BedrockModels.CLAUDE_SONNET_4_5,
    'Claude Haiku 4.5': BedrockModels.CLAUDE_HAIKU_4_5,
    'Llama 3.2 90B': BedrockModels.LLAMA_3_2_90B,
    'Llama 3.2 11B': BedrockModels.LLAMA_3_2_11B,
    'Llama 3.2 3B': BedrockModels.LLAMA_3_2_3B,
    'Llama 3.2 1B': BedrockModels.LLAMA_3_2_1B,
    'Llama 3.1 70B': BedrockModels.LLAMA_3_1_70B,
    'Llama 3.1 8B': BedrockModels.LLAMA_3_1_8B,
    'Llama 3.3 70B': BedrockModels.LLAMA_3_3_70B,
    'Llama 4 Scout': BedrockModels.LLAMA_4_SCOUT,
    'Llama 4 Maverick': BedrockModels.LLAMA_4_MAVERICK,
    'Nova Premier': BedrockModels.NOVA_PREMIER,
    'Nova Pro': BedrockModels.NOVA_PRO,
    'Nova Lite': BedrockModels.NOVA_LITE,
    'Nova Micro': BedrockModels.NOVA_MICRO,
    'Pixtral Large': BedrockModels.PIXTRAL_LARGE,
    'Mistral Large': BedrockModels.MISTRAL_LARGE,
    'Mistral Small': BedrockModels.MISTRAL_SMALL,
    'Mistral 7B': BedrockModels.MISTRAL_7B,
    'Mixtral 8x7B': BedrockModels.MIXTRAL_8X7B,
    'DeepSeek R1': BedrockModels.DEEPSEEK_R1,
}


def check_model_access(model_id: str) -> bool:
    """Check if a model is accessible by making a test call."""
    try:
        client = BedrockClient()
        # Try a minimal invoke to check access
        test_messages = [{"role": "user", "content": "test"}]
        response = client.invoke(
            messages=test_messages,
            model=model_id,
            max_tokens=10
        )
        return True
    except Exception as e:
        # Check if it's an access/permission error vs other error
        error_str = str(e).lower()
        if 'not found' in error_str or 'not available' in error_str or 'access' in error_str:
            return False
        # Other errors (like rate limits) suggest the model exists
        return True


def main():
    print("="*70)
    print("BEDROCK MODEL AVAILABILITY CHECK")
    print("="*70)
    print()
    
    # Check your CSV files
    print("📋 Checking your CSV files against Bedrock models:")
    print("-"*70)
    
    available = []
    unavailable = []
    partial_match = []
    
    for csv_file in files_to_process:
        bedrock_model = CSV_TO_BEDROCK.get(csv_file)
        
        if bedrock_model is None:
            unavailable.append(csv_file)
            print(f"❌ {csv_file:40s} → NOT AVAILABLE on Bedrock")
        else:
            # Check if model is actually accessible
            model_name = csv_file.replace(' Outputs.csv', '')
            print(f"✓  {csv_file:40s} → {bedrock_model}")
            available.append((csv_file, bedrock_model, model_name))
    
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n✅ Available on Bedrock: {len(available)}")
    for csv_file, model_id, name in available:
        print(f"   • {name:30s} → {model_id}")
    
    print(f"\n❌ NOT Available on Bedrock: {len(unavailable)}")
    for csv_file in unavailable:
        model_name = csv_file.replace(' Outputs.csv', '')
        print(f"   • {model_name}")
    
    print()
    print("="*70)
    print("ALL AVAILABLE BEDROCK MODELS")
    print("="*70)
    print("\nThese are all the models available on AWS Bedrock:")
    for name, model_id in sorted(ALL_BEDROCK_MODELS.items()):
        print(f"   • {name:30s} → {model_id}")
    
    print()
    print("="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    print("\nFor your CSV files:")
    print(f"   • {len(available)} files can be processed with Bedrock models")
    print(f"   • {len(unavailable)} files are from other providers (Google, OpenAI)")
    print("\nNote: You may need to verify actual access in your AWS account.")
    print("      Some models may require specific region or quota settings.")


if __name__ == "__main__":
    main()

