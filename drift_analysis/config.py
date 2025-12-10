"""
Configuration for drift analysis.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class DriftAnalysisConfig:
    """Configuration for drift analysis."""

    # Input/Output paths
    evaluation_results_dir: Path = Path('model_evaluations/results')
    output_dir: Path = Path('drift_analysis/results')

    # Models to analyze (None = analyze all available results)
    models: Optional[List[str]] = None

    # Bias types to analyze (None = all)
    bias_types: Optional[List[str]] = None

    # Similarity metrics configuration
    similarity_metrics: List[str] = None

    # Processing options
    batch_size: int = 32  # Batch size for embedding generation

    # Embedding model configuration
    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'  # Fast, lightweight model
    # Alternative models:
    # - 'sentence-transformers/all-mpnet-base-v2' (higher quality, slower)
    # - 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2' (multilingual)

    # HEARTS configuration (primary drift detection method)
    enable_hearts: bool = True  # Use HEARTS for stereotype detection (recommended)
    enable_shap: bool = False  # Enable SHAP explanations (memory-intensive)
    enable_lime: bool = False  # Enable LIME explanations (very memory-intensive)

    def __post_init__(self):
        """Initialize default values."""
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

        if self.similarity_metrics is None:
            # Default similarity metrics to compute
            self.similarity_metrics = [
                'cosine',           # Cosine similarity (semantic)
                'euclidean',        # Euclidean distance
                'bleu',             # BLEU score (n-gram overlap)
                'rouge',            # ROUGE score (recall-oriented)
            ]

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> List[str]:
        """
        Validate configuration and return list of errors.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Check if evaluation results directory exists
        if not self.evaluation_results_dir.exists():
            errors.append(
                f"Evaluation results directory not found: {self.evaluation_results_dir}"
            )

        # Validate similarity metrics
        valid_metrics = {'cosine', 'euclidean', 'bleu', 'rouge'}
        invalid = set(self.similarity_metrics) - valid_metrics
        if invalid:
            errors.append(
                f"Invalid similarity metrics: {invalid}. "
                f"Valid: {valid_metrics}"
            )

        # Validate batch size
        if self.batch_size <= 0:
            errors.append(f"batch_size must be positive, got: {self.batch_size}")

        return errors

    def to_dict(self) -> dict:
        """Convert configuration to dictionary for serialization."""
        return {
            'evaluation_results_dir': str(self.evaluation_results_dir),
            'output_dir': str(self.output_dir),
            'models': self.models,
            'bias_types': self.bias_types,
            'similarity_metrics': self.similarity_metrics,
            'batch_size': self.batch_size,
            'embedding_model': self.embedding_model,
            'enable_hearts': self.enable_hearts,
            'enable_shap': self.enable_shap,
            'enable_lime': self.enable_lime,
        }
