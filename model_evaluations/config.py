"""
Configuration for model evaluation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict
import os

# Benchmark models with source specification
# Format: {model_id: source} where source is "bedrock" or "ollama"
# The system automatically detects the source based on model ID format
BENCHMARK_MODELS = [
    # Bedrock Models (cloud-based)
    'us.anthropic.claude-3-5-sonnet-20241022-v2:0',  # Claude 3.5 Sonnet V2 (Bedrock)
    'us.anthropic.claude-3-sonnet-20240229-v1:0',    # Claude 3 Sonnet (Bedrock)
    'us.anthropic.claude-3-5-haiku-20241022-v1:0',  # Claude 3.5 Haiku (Bedrock)
    'us.anthropic.claude-3-haiku-20240307-v1:0',    # Claude 3 Haiku (Bedrock)
    'us.meta.llama3-1-70b-instruct-v1:0',            # Llama 3.1 70B (Bedrock)
    'us.amazon.nova-pro-v1:0',                      # Nova Pro (Bedrock)
    'us.amazon.nova-lite-v1:0',                     # Nova Lite (Bedrock)
    'us.amazon.nova-micro-v1:0',                    # Nova Micro (Bedrock)
    
    # Ollama Models (self-hosted)
    # Note: Make sure Ollama is running and models are pulled before evaluation
    'llama3.1:8b',      # Ollama - Llama 3.1 8B (~4.7GB)
    'llama3.2:3b',
    'llama3.2:1b',
    'mistral:7b',       # Ollama - Mistral 7B (~4.1GB)
    'gemma2:9b',        # Ollama - Gemma 2 9B (~5.4GB)
    'qwen2.5:7b',       # Ollama - Qwen 2.5 7B (~4.1GB)
    'deepseek-llm:7b',  # Ollama - DeepSeek LLM 7B (~4GB)
    'gpt-oss:20b-cloud', # Ollama - GPT OSS 20B
]


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    
    # Dataset paths (relative to project root)
    dataset_path: Path = Path('dataset_generation/data')  # Will auto-find latest dataset
    output_dir: Path = Path('model_evaluations/results')
    
    # Models to evaluate
    models: List[str] = None
    
    # Processing options
    sample_limit: int = None  # Limit number of entries to evaluate
    bias_types: List[str] = None  # Specific bias types to evaluate (None = all)
    
    # Parallel processing
    max_workers: int = 5  # Conservative for evaluation
    max_requests_per_second: float = 3.0  # Conservative rate limit
    use_parallel: bool = True
    
    # Checkpointing
    checkpoint_interval: int = 10  # Save checkpoint every N entries
    resume_from_checkpoint: bool = False  # Resume from latest checkpoint if available
    
    def __post_init__(self):
        """Initialize default values."""
        if self.models is None:
            self.models = BENCHMARK_MODELS
        
        if self.bias_types is None:
            # Default to all 8 bias types
            self.bias_types = [
                'confirmation_bias',
                'anchoring_bias',
                'demographic_bias',
                'availability_bias',
                'framing_bias',
                'leading_question',
                'stereotypical_assumption',
                'negativity_bias'
            ]
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def get_model_sources(models: List[str]) -> Dict[str, str]:
        """
        Get source mapping for a list of models.
        
        Args:
            models: List of model IDs
            
        Returns:
            Dictionary mapping model_id -> source ("bedrock" or "ollama")
        """
        try:
            from core.model_source_mapper import get_model_source, ModelSource
            return {
                model_id: get_model_source(model_id).value 
                for model_id in models
            }
        except ImportError:
            # Fallback: assume all are Bedrock if mapper not available
            return {model_id: "bedrock" for model_id in models}

