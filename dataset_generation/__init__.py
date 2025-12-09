"""
Multi-Turn EMGSD Dataset Generation Package

A maintainable, modular system for generating enhanced EMGSD datasets
with cognitive bias injection questions.
"""

__version__ = "1.0.0"

from .config import DatasetConfig
from .dataset_builder import DatasetBuilder
from .bias_generator import BiasGenerator
from .emgsd_processor import EMGSDProcessor

__all__ = [
    'DatasetConfig',
    'DatasetBuilder',
    'BiasGenerator',
    'EMGSDProcessor',
]

