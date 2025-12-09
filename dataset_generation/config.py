"""
Configuration management for dataset generation.

Centralizes all configuration settings and provides type-safe access.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from core.bedrock_client import BedrockModels
from core.bias_instructions import get_all_bias_types


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    
    # Model configuration
    # Change this to use a different model for generating bias injection questions
    # Available models: BedrockModels.CLAUDE_3_5_SONNET_V2, BedrockModels.CLAUDE_3_5_HAIKU, etc.
    bias_generator_model: str = BedrockModels.NOVA_PRO#CLAUDE_3_5_SONNET_V2
    use_persona_prompts: bool = True
    
    # Bias types to generate
    bias_types: Optional[List[str]] = None
    
    # Dataset paths
    emgsd_transformed_path: Path = Path('data/emgsd_with_prompts.csv')
    emgsd_default_path: Optional[Path] = None
    output_dir: Path = Path('data')
    
    # Processing options
    category_filter: str = 'stereotype'  # 'stereotype', 'neutral', 'unrelated'
    stereotype_type_filter: Optional[str] = None
    sample_limit: Optional[int] = None
    
    # Checkpointing
    checkpoint_interval: int = 50  # Save checkpoint every N entries
    
    # Parallel processing
    max_workers: int = 10  # Number of concurrent workers
    max_requests_per_second: float = 5.0  # Rate limit (requests per second)
    use_parallel: bool = True  # Enable parallel processing
    
    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.bias_types is None:
            # Default to all available bias types
            self.bias_types = self.all_available_bias_types
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def all_available_bias_types(self) -> List[str]:
        """Get all available bias types from bias_instructions."""
        return get_all_bias_types()
    
    def validate(self) -> List[str]:
        """
        Validate configuration and return list of errors.
        
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        # Validate bias types
        available = set(self.all_available_bias_types)
        invalid = set(self.bias_types) - available
        if invalid:
            errors.append(
                f"Invalid bias types: {invalid}. "
                f"Available: {available}"
            )
        
        # Validate category filter
        valid_categories = {'stereotype', 'neutral', 'unrelated'}
        if self.category_filter not in valid_categories:
            errors.append(
                f"Invalid category_filter: {self.category_filter}. "
                f"Valid: {valid_categories}"
            )
        
        # Validate stereotype type filter
        if self.stereotype_type_filter:
            valid_types = {'profession', 'nationality', 'gender', 'religion'}
            if self.stereotype_type_filter not in valid_types:
                errors.append(
                    f"Invalid stereotype_type_filter: {self.stereotype_type_filter}. "
                    f"Valid: {valid_types}"
                )
        
        # Validate sample limit
        if self.sample_limit is not None and self.sample_limit <= 0:
            errors.append(f"sample_limit must be positive, got: {self.sample_limit}")
        
        # Validate checkpoint interval
        if self.checkpoint_interval <= 0:
            errors.append(
                f"checkpoint_interval must be positive, got: {self.checkpoint_interval}"
            )
        
        return errors
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary for serialization."""
        return {
            'bias_generator_model': self.bias_generator_model,
            'use_persona_prompts': self.use_persona_prompts,
            'bias_types': self.bias_types,
            'category_filter': self.category_filter,
            'max_workers': self.max_workers,
            'max_requests_per_second': self.max_requests_per_second,
            'use_parallel': self.use_parallel,
            'stereotype_type_filter': self.stereotype_type_filter,
            'sample_limit': self.sample_limit,
            'checkpoint_interval': self.checkpoint_interval,
        }

