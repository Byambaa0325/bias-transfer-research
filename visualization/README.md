# Visualization

Generates visualizations from drift analysis results to analyze bias transfer patterns across models.

## Input
- Drift analysis CSV files: `drift_analysis/results/{model_name}/{date}/drift_analysis.csv`
- Configuration: model selection, date selection, output directory

## Output
- PNG visualizations saved to same directory as input:
  - drift_distribution.png: Histogram of drift scores
  - control_vs_bias_comparison.png: Scatter plot comparing control vs bias responses
  - drift_by_bias_type.png: Box plots showing drift by cognitive bias type
  - drift_by_stereotype_type.png: Analysis by stereotype category
  - heatmap_bias_x_stereotype.png: Interaction heatmaps
  - stereotype_similarity_analysis.png: Response similarity analysis
  - trait_mention_analysis.png: Trait mention frequency
  - raw_drift_analysis.png: Raw drift analysis showing direction of bias influence
  - semantic_similarity_drift_direction.png: Semantic similarity drift direction analysis

## How It Works
1. Loads drift analysis CSV files
2. Filters valid entries (with HEARTS scores)
3. Generates 9 visualization types using matplotlib/seaborn
4. Saves PNG files to model-specific directories

**Command**: `python -m visualization.main --model gemma2_9b`
