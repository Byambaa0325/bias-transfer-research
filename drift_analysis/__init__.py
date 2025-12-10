"""
Drift Analysis Module

This module provides tools for analyzing semantic drift and stereotype similarity
in LLM responses. It processes evaluation results to measure:
1. Semantic drift between control and bias-injected responses
2. Similarity of responses to original stereotype sentences
"""

from drift_analysis.config import DriftAnalysisConfig
from drift_analysis.drift_calculator import DriftCalculator
from drift_analysis.similarity_analyzer import SimilarityAnalyzer
from drift_analysis.main import DriftAnalyzer

__all__ = [
    'DriftAnalysisConfig',
    'DriftCalculator',
    'SimilarityAnalyzer',
    'DriftAnalyzer',
]

__version__ = '1.0.0'
