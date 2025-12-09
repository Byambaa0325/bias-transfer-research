"""
Model Source Mapper

Maps model IDs to their inference source (Bedrock or Ollama).
"""

from typing import Dict, Literal
from enum import Enum


class ModelSource(Enum):
    """Model inference source"""
    BEDROCK = "bedrock"
    OLLAMA = "ollama"


# Mapping of model IDs to their source
# Bedrock models (default - anything not in Ollama mapping)
BEDROCK_MODELS = {
    # Anthropic Claude
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "us.anthropic.claude-3-opus-20240229-v1:0",
    "us.anthropic.claude-3-sonnet-20240229-v1:0",
    "us.anthropic.claude-3-haiku-20240307-v1:0",
    "us.anthropic.claude-opus-4-20250514-v1:0",
    "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    
    # Meta Llama (Bedrock)
    "us.meta.llama3-2-90b-instruct-v1:0",
    "us.meta.llama3-2-11b-instruct-v1:0",
    "us.meta.llama3-2-3b-instruct-v1:0",
    "us.meta.llama3-2-1b-instruct-v1:0",
    "us.meta.llama3-1-70b-instruct-v1:0",
    "us.meta.llama3-1-8b-instruct-v1:0",
    "us.meta.llama3-3-70b-instruct-v1:0",
    "us.meta.llama4-scout-17b-instruct-v1:0",
    "us.meta.llama4-maverick-17b-instruct-v1:0",
    
    # Amazon Nova
    "us.amazon.nova-premier-v1:0",
    "us.amazon.nova-pro-v1:0",
    "us.amazon.nova-lite-v1:0",
    "us.amazon.nova-micro-v1:0",
    
    # Mistral (Bedrock)
    "us.mistral.pixtral-large-2502-v1:0",
    "mistral.mistral-large-2402-v1:0",
    "mistral.mistral-small-2402-v1:0",
    "mistral.mistral-7b-instruct-v0:2",
    "mistral.mixtral-8x7b-instruct-v0:1",
    
    # DeepSeek
    "us.deepseek.r1-v1:0",
}

# Ollama models (self-hosted)
# These are the Ollama model names that can be used
OLLAMA_MODELS = {
    # Llama models
    "llama3.1:8b",
    "llama3.1:70b",
    "llama3.1:405b",
    "llama3.2:1b",
    "llama3.2:3b",
    "llama3:8b",
    "llama3:70b",
    
    # Mistral models
    "mistral:7b",
    "mistral:8x7b",
    "mistral-nemo:12b",
    "mistral-small:latest",
    "mistral-medium:latest",
    "mistral-large:latest",
    "mistral-large-2:latest",  # Mistral Large 2
    "mistral-tiny:latest",
    
    # DeepSeek models
    "deepseek-r1:7b",
    "deepseek-r1:14b",  # DeepSeek R1 14B
    "deepseek-r1:32b",
    "deepseek-r1:70b",
    "deepseek-coder:6.7b",
    "deepseek-coder:33b",
    
    # Gemma models
    "gemma2:9b",  # Gemma 2 9B
    "gemma2:27b",
    "gemma:2b",
    "gemma:7b",
    
    # Qwen models
    "qwen2.5:7b",  # Qwen 2.5 7B
    "qwen2.5:14b",
    "qwen2.5:32b",
    "qwen2.5:72b",
    "qwen:7b",
    "qwen:14b",
    "qwen:32b",
    
    # GPT OSS models
    "gpt-oss:20b-cloud",  # GPT OSS 20B
    "gpt-oss:20b",
    
    # Other models
    "phi3:mini",
    "phi3:medium",
    "codellama:7b",
    "codellama:13b",
    "codellama:34b",
    "neural-chat:7b",
    "starling-lm:7b",
    "solar:10.7b",
}


def get_model_source(model_id: str) -> ModelSource:
    """
    Determine the inference source for a given model ID.
    
    Args:
        model_id: Model identifier
        
    Returns:
        ModelSource enum (BEDROCK or OLLAMA)
    """
    # Check if it's an Ollama model (simpler format, no "us." prefix)
    if model_id in OLLAMA_MODELS:
        return ModelSource.OLLAMA
    
    # Check if it's a Bedrock model (has "us." prefix or known Bedrock format)
    if model_id in BEDROCK_MODELS:
        return ModelSource.BEDROCK
    
    # Heuristic: Ollama models typically don't have "us." prefix
    # Bedrock models typically have "us." prefix
    if model_id.startswith("us."):
        return ModelSource.BEDROCK
    
    # Default: assume Ollama for simple model names (e.g., "llama3.1:8b")
    # This allows flexibility for custom Ollama models
    return ModelSource.OLLAMA


def is_bedrock_model(model_id: str) -> bool:
    """Check if a model is from Bedrock."""
    return get_model_source(model_id) == ModelSource.BEDROCK


def is_ollama_model(model_id: str) -> bool:
    """Check if a model is from Ollama."""
    return get_model_source(model_id) == ModelSource.OLLAMA


def get_model_source_dict(models: list[str]) -> Dict[str, ModelSource]:
    """
    Get source mapping for a list of models.
    
    Args:
        models: List of model IDs
        
    Returns:
        Dictionary mapping model_id -> ModelSource
    """
    return {model_id: get_model_source(model_id) for model_id in models}

