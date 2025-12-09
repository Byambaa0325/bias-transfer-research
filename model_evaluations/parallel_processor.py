"""
Parallel processing utilities for model evaluation.

Reuses the parallel processor from dataset generation.
"""

import sys
from pathlib import Path

# Import from dataset_generation
sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset_generation.parallel_processor import ParallelProcessor, RateLimiter

__all__ = ['ParallelProcessor', 'RateLimiter']

