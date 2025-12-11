"""
Configuration for drift analysis visualization.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class VisualizationConfig:
    """Configuration for generating visualizations from drift analysis results."""
    
    # Input: Drift analysis results directory
    drift_results_dir: Path = Path("drift_analysis/results")
    
    # Output: Where to save visualizations (default: same as input, per model/date)
    output_dir: Optional[Path] = None
    
    # Model selection
    model_name: Optional[str] = None  # None = process all models
    date: Optional[str] = None  # None = use latest date for each model
    
    # Visualization settings
    dpi: int = 300
    figure_size: tuple = (16, 6)
    figure_size_large: tuple = (16, 12)
    
    # Style settings
    style: str = "whitegrid"
    font_size: int = 10
    
    def __post_init__(self):
        """Validate and normalize paths."""
        if isinstance(self.drift_results_dir, str):
            self.drift_results_dir = Path(self.drift_results_dir)
        
        if self.output_dir is None:
            # Default: save visualizations in the same directory as the data
            self.output_dir = self.drift_results_dir
        elif isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

