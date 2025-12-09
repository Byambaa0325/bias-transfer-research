"""
Configuration for model evaluation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

# Benchmark models (10 models - Mistral and DeepSeek removed)
BENCHMARK_MODELS = [
    # Required (3)
    'us.anthropic.claude-3-5-sonnet-20241022-v2:0',  # Claude 3.5 Sonnet V2
    'us.anthropic.claude-3-sonnet-20240229-v1:0',    # Claude 3 Sonnet
    'us.meta.llama3-1-70b-instruct-v1:0',            # Llama 3.1 70B
    
    # Representative (7)
    'us.anthropic.claude-3-5-haiku-20241022-v1:0',  # Claude 3.5 Haiku
    'us.anthropic.claude-3-haiku-20240307-v1:0',    # Claude 3 Haiku
    'us.meta.llama3-1-8b-instruct-v1:0',            # Llama 3.1 8B
    'us.meta.llama3-2-11b-instruct-v1:0',          # Llama 3.2 11B
    'us.amazon.nova-pro-v1:0',                      # Nova Pro
    'us.amazon.nova-lite-v1:0',                     # Nova Lite
    'us.amazon.nova-micro-v1:0',                    # Nova Micro
]


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    
    # Dataset paths
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

