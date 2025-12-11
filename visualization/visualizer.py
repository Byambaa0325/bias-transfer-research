"""
Drift analysis visualizer.

Generates visualizations from drift analysis CSV files.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import Optional, List
import warnings

from visualization.config import VisualizationConfig

warnings.filterwarnings('ignore')


class DriftVisualizer:
    """Generates visualizations from drift analysis results."""
    
    def __init__(self, config: VisualizationConfig):
        """
        Initialize the visualizer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config
        self._setup_plotting_style()
    
    def _setup_plotting_style(self):
        """Configure matplotlib and seaborn plotting style."""
        sns.set_style(self.config.style)
        plt.rcParams['figure.figsize'] = self.config.figure_size
        plt.rcParams['font.size'] = self.config.font_size
    
    def find_available_models(self) -> List[str]:
        """
        Find all available models in the drift results directory.
        
        Returns:
            List of model names
        """
        if not self.config.drift_results_dir.exists():
            return []
        
        models = [
            d.name for d in self.config.drift_results_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ]
        return sorted(models)
    
    def find_available_dates(self, model_name: str) -> List[str]:
        """
        Find all available dates for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of date strings (YYYYMMDD format)
        """
        model_dir = self.config.drift_results_dir / model_name
        if not model_dir.exists():
            return []
        
        dates = [
            d.name for d in model_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ]
        return sorted(dates, reverse=True)  # Latest first
    
    def get_data_file_path(self, model_name: str, date: Optional[str] = None) -> Optional[Path]:
        """
        Get the path to the drift analysis CSV file for a model.
        
        Args:
            model_name: Name of the model
            date: Date string (YYYYMMDD) or None for latest
            
        Returns:
            Path to the CSV file, or None if not found
        """
        model_dir = self.config.drift_results_dir / model_name
        if not model_dir.exists():
            return None
        
        if date is None:
            dates = self.find_available_dates(model_name)
            if not dates:
                return None
            date = dates[0]
        
        data_file = model_dir / date / 'drift_analysis.csv'
        return data_file if data_file.exists() else None
    
    def get_output_dir(self, model_name: str, date: Optional[str] = None) -> Optional[Path]:
        """
        Get the output directory for visualizations.
        
        Args:
            model_name: Name of the model
            date: Date string (YYYYMMDD) or None for latest
            
        Returns:
            Path to output directory
        """
        if date is None:
            dates = self.find_available_dates(model_name)
            if not dates:
                return None
            date = dates[0]
        
        return self.config.output_dir / model_name / date
    
    def load_data(self, model_name: str, date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Load drift analysis data for a model.
        
        Args:
            model_name: Name of the model
            date: Date string (YYYYMMDD) or None for latest
            
        Returns:
            DataFrame with drift analysis data, or None if not found
        """
        data_file = self.get_data_file_path(model_name, date)
        if data_file is None:
            return None
        
        df = pd.read_csv(data_file)
        df['model_id_short'] = model_name
        return df
    
    def filter_valid_entries(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter to entries with valid drift scores.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Filtered DataFrame
        """
        # Filter by any available drift metric (HEARTS or semantic similarity)
        has_drift_score = 'drift_drift_score' in df.columns and df['drift_drift_score'].notna().any()
        has_semantic_drift = 'similarity_semantic_similarity_drift' in df.columns and df['similarity_semantic_similarity_drift'].notna().any()
        has_cosine_sim = 'drift_cosine_similarity' in df.columns and df['drift_cosine_similarity'].notna().any()
        
        if has_drift_score:
            # Use HEARTS drift score if available
            valid_df = df[
                df['drift_drift_score'].notna() &
                (df['error'].isna() if 'error' in df.columns else True)
            ].copy()
        elif has_semantic_drift:
            # Fallback to semantic similarity drift
            valid_df = df[
                df['similarity_semantic_similarity_drift'].notna() &
                (df['error'].isna() if 'error' in df.columns else True)
            ].copy()
        elif has_cosine_sim:
            # Fallback to cosine similarity
            valid_df = df[
                df['drift_cosine_similarity'].notna() &
                (df['error'].isna() if 'error' in df.columns else True)
            ].copy()
        else:
            # Last resort: just filter by error
            valid_df = df[
                df['error'].isna() if 'error' in df.columns else True
            ].copy()
        
        return valid_df
    
    def visualize_drift_distribution(self, df: pd.DataFrame, model_name: str, output_dir: Path):
        """Generate drift score distribution visualization."""
        if 'drift_drift_score' not in df.columns:
            print("WARNING: HEARTS drift scores not available in dataset")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=self.config.figure_size)
        
        # Drift score distribution
        axes[0].hist(df['drift_drift_score'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].axvline(0.2, color='red', linestyle='--', linewidth=2, label='Significance threshold (0.2)')
        axes[0].set_xlabel('HEARTS Drift Score (Absolute)')
        axes[0].set_ylabel('Count')
        axes[0].set_title(f'Distribution of HEARTS Drift Scores: {model_name}', fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Add statistics annotation
        mean_drift = df['drift_drift_score'].mean()
        median_drift = df['drift_drift_score'].median()
        significant_rate = df['drift_significant_drift'].mean() if 'drift_significant_drift' in df.columns else 0
        
        stats_text = f"Mean: {mean_drift:.3f}\nMedian: {median_drift:.3f}\nSignificant: {significant_rate:.1%}"
        axes[0].text(0.98, 0.97, stats_text, transform=axes[0].transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=10)
        
        # Box plot by bias type
        if 'bias_type' in df.columns:
            df.boxplot(column='drift_drift_score', by='bias_type', ax=axes[1])
            axes[1].axhline(0.2, color='red', linestyle='--', linewidth=2, label='Significance threshold (0.2)')
            axes[1].set_xlabel('Bias Type')
            axes[1].set_ylabel('HEARTS Drift Score')
            axes[1].set_title(f'Drift Score by Bias Type: {model_name}', fontweight='bold')
            axes[1].get_figure().suptitle('')
            plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
            axes[1].legend()
            axes[1].grid(alpha=0.3)
        else:
            # If no bias_type, show KDE plot
            df['drift_drift_score'].plot(kind='kde', ax=axes[1], color='steelblue', linewidth=2)
            axes[1].axvline(0.2, color='red', linestyle='--', linewidth=2, label='Significance threshold (0.2)')
            axes[1].set_xlabel('HEARTS Drift Score')
            axes[1].set_ylabel('Density')
            axes[1].set_title(f'Drift Score Distribution: {model_name}', fontweight='bold')
            axes[1].legend()
            axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        output_file = output_dir / 'drift_distribution.png'
        plt.savefig(output_file, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        print(f"OK: Saved: {output_file}")
    
    def visualize_control_vs_bias(self, df: pd.DataFrame, model_name: str, output_dir: Path):
        """Generate control vs bias-injected comparison visualization."""
        if 'drift_control_stereotype_score' not in df.columns or 'drift_bias_stereotype_score' not in df.columns:
            print("WARNING: Control/bias stereotype scores not available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=self.config.figure_size)
        
        # Comparison histogram
        axes[0].hist(df['drift_control_stereotype_score'], bins=30, alpha=0.6,
                    label='Control (no bias)', color='blue', edgecolor='black')
        axes[0].hist(df['drift_bias_stereotype_score'], bins=30, alpha=0.6,
                    label='Bias-injected', color='red', edgecolor='black')
        axes[0].set_xlabel('HEARTS Stereotype Score')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Stereotype Scores: Control vs Bias-Injected', fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Scatter plot (paired comparison)
        axes[1].scatter(df['drift_control_stereotype_score'],
                       df['drift_bias_stereotype_score'],
                       alpha=0.4, s=20, color='steelblue')
        axes[1].plot([0, 1], [0, 1], 'r--', linewidth=2, label='No effect line (y=x)')
        axes[1].set_xlabel('Control Stereotype Score')
        axes[1].set_ylabel('Bias-Injected Stereotype Score')
        axes[1].set_title('Paired Comparison: Control vs Bias-Injected', fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)
        
        plt.tight_layout()
        output_file = output_dir / 'control_vs_bias_comparison.png'
        plt.savefig(output_file, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        print(f"OK: Saved: {output_file}")
        
        # Statistical test
        t_stat, p_value = stats.ttest_rel(
            df['drift_control_stereotype_score'],
            df['drift_bias_stereotype_score']
        )
        
        print("\nPaired t-test (Control vs Bias-Injected):")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.6f}")
        
        if p_value < 0.001:
            print("  WARNING: HIGHLY SIGNIFICANT difference (p < 0.001)")
        elif p_value < 0.05:
            print("  WARNING: SIGNIFICANT difference (p < 0.05)")
        else:
            print("  OK: No significant difference detected")
    
    def visualize_drift_by_bias_type(self, df: pd.DataFrame, model_name: str, output_dir: Path):
        """Generate drift by bias type visualization."""
        if 'bias_type' not in df.columns or 'drift_drift_score' not in df.columns:
            print("WARNING: Bias type information not available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=self.config.figure_size)
        
        # Box plot by bias type
        df.boxplot(column='drift_drift_score', by='bias_type', ax=axes[0])
        axes[0].axhline(0.2, color='red', linestyle='--', linewidth=2, alpha=0.5)
        axes[0].set_xlabel('Cognitive Bias Type')
        axes[0].set_ylabel('HEARTS Drift Score')
        axes[0].set_title('Drift Score by Cognitive Bias Type', fontweight='bold')
        axes[0].get_figure().suptitle('')
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        axes[0].grid(alpha=0.3)
        
        # Bar plot of significant drift rate by bias type
        if 'drift_significant_drift' in df.columns:
            bias_type_stats = df.groupby('bias_type').agg({
                'drift_significant_drift': 'mean',
                'drift_drift_score': 'mean'
            }).sort_values('drift_drift_score', ascending=False)
            
            bias_type_stats['drift_significant_drift'].plot(kind='bar', ax=axes[1], color='coral', edgecolor='black')
            axes[1].set_xlabel('Cognitive Bias Type')
            axes[1].set_ylabel('Significant Drift Rate')
            axes[1].set_title('Significant Drift Rate by Bias Type', fontweight='bold')
            axes[1].set_ylim(0, 1)
            axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
            plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
            axes[1].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_file = output_dir / 'drift_by_bias_type.png'
        plt.savefig(output_file, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        print(f"OK: Saved: {output_file}")
    
    def visualize_drift_by_stereotype_type(self, df: pd.DataFrame, model_name: str, output_dir: Path):
        """Generate drift by stereotype type visualization."""
        if 'emgsd_stereotype_type' not in df.columns or 'drift_drift_score' not in df.columns:
            print("WARNING: Stereotype type information not available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=self.config.figure_size)
        
        # Box plot by stereotype type
        df.boxplot(column='drift_drift_score', by='emgsd_stereotype_type', ax=axes[0])
        axes[0].axhline(0.2, color='red', linestyle='--', linewidth=2, alpha=0.5)
        axes[0].set_xlabel('Stereotype Type')
        axes[0].set_ylabel('HEARTS Drift Score')
        axes[0].set_title('Drift Score by Stereotype Type', fontweight='bold')
        axes[0].get_figure().suptitle('')
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        axes[0].grid(alpha=0.3)
        
        # Bar plot of mean drift by stereotype type
        stereo_stats = df.groupby('emgsd_stereotype_type')['drift_drift_score'].mean().sort_values(ascending=False)
        stereo_stats.plot(kind='bar', ax=axes[1], color='lightcoral', edgecolor='black')
        axes[1].set_xlabel('Stereotype Type')
        axes[1].set_ylabel('Mean HEARTS Drift Score')
        axes[1].set_title('Mean Drift Score by Stereotype Type', fontweight='bold')
        axes[1].axhline(0.2, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Threshold (0.2)')
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        axes[1].legend()
        axes[1].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_file = output_dir / 'drift_by_stereotype_type.png'
        plt.savefig(output_file, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        print(f"OK: Saved: {output_file}")
    
    def visualize_heatmaps(self, df: pd.DataFrame, model_name: str, output_dir: Path):
        """Generate heatmap visualizations."""
        if 'drift_drift_score' not in df.columns:
            print("WARNING: Cannot create heatmaps - missing required columns")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size_large)
        
        # 1. Heatmap: Drift score by bias type and stereotype type
        if 'bias_type' in df.columns and 'emgsd_stereotype_type' in df.columns:
            heatmap_data = df.pivot_table(
                values='drift_drift_score',
                index='bias_type',
                columns='emgsd_stereotype_type',
                aggfunc='mean'
            )
            sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[0, 0],
                       cbar_kws={'label': 'Mean Drift Score'}, linewidths=0.5)
            axes[0, 0].set_title(f'Drift Score Heatmap: {model_name}', fontweight='bold')
            axes[0, 0].set_xlabel('Stereotype Type')
            axes[0, 0].set_ylabel('Bias Type')
        
        # 2. Heatmap: Significant drift rate
        if 'drift_significant_drift' in df.columns and 'bias_type' in df.columns and 'emgsd_stereotype_type' in df.columns:
            sig_heatmap_data = df.pivot_table(
                values='drift_significant_drift',
                index='bias_type',
                columns='emgsd_stereotype_type',
                aggfunc='mean'
            )
            sns.heatmap(sig_heatmap_data, annot=True, fmt='.2%', cmap='Reds', ax=axes[0, 1],
                       cbar_kws={'label': 'Significant Drift Rate'}, linewidths=0.5, vmin=0, vmax=1)
            axes[0, 1].set_title(f'Significant Drift Rate Heatmap: {model_name}', fontweight='bold')
            axes[0, 1].set_xlabel('Stereotype Type')
            axes[0, 1].set_ylabel('Bias Type')
        
        # 3. Heatmap: Control stereotype score
        if 'drift_control_stereotype_score' in df.columns and 'bias_type' in df.columns and 'emgsd_stereotype_type' in df.columns:
            control_heatmap_data = df.pivot_table(
                values='drift_control_stereotype_score',
                index='bias_type',
                columns='emgsd_stereotype_type',
                aggfunc='mean'
            )
            sns.heatmap(control_heatmap_data, annot=True, fmt='.3f', cmap='Blues', ax=axes[1, 0],
                       cbar_kws={'label': 'Control Stereotype Score'}, linewidths=0.5, vmin=0, vmax=1)
            axes[1, 0].set_title(f'Control Stereotype Score Heatmap: {model_name}', fontweight='bold')
            axes[1, 0].set_xlabel('Stereotype Type')
            axes[1, 0].set_ylabel('Bias Type')
        
        # 4. Heatmap: Bias-injected stereotype score
        if 'drift_bias_stereotype_score' in df.columns and 'bias_type' in df.columns and 'emgsd_stereotype_type' in df.columns:
            bias_heatmap_data = df.pivot_table(
                values='drift_bias_stereotype_score',
                index='bias_type',
                columns='emgsd_stereotype_type',
                aggfunc='mean'
            )
            sns.heatmap(bias_heatmap_data, annot=True, fmt='.3f', cmap='Reds', ax=axes[1, 1],
                       cbar_kws={'label': 'Bias-Injected Stereotype Score'}, linewidths=0.5, vmin=0, vmax=1)
            axes[1, 1].set_title(f'Bias-Injected Stereotype Score Heatmap: {model_name}', fontweight='bold')
            axes[1, 1].set_xlabel('Stereotype Type')
            axes[1, 1].set_ylabel('Bias Type')
        
        plt.tight_layout()
        output_file = output_dir / 'heatmap_bias_x_stereotype.png'
        plt.savefig(output_file, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        print(f"OK: Saved: {output_file}")
    
    def visualize_similarity_analysis(self, df: pd.DataFrame, model_name: str, output_dir: Path):
        """Generate semantic similarity analysis visualization using the four metrics."""
        # Check for semantic similarity metrics
        has_control_sim = 'similarity_control_semantic_similarity_to_stereotype' in df.columns or 'similarity_control_semantic_similarity' in df.columns
        has_bias_sim = 'similarity_bias_semantic_similarity_to_stereotype' in df.columns or 'similarity_turn2_semantic_similarity' in df.columns
        
        if not (has_control_sim and has_bias_sim):
            print("WARNING: Semantic similarity scores not available")
            return
        
        # Get column names (support both new and legacy)
        control_sim_col = 'similarity_control_semantic_similarity_to_stereotype' if 'similarity_control_semantic_similarity_to_stereotype' in df.columns else 'similarity_control_semantic_similarity'
        bias_sim_col = 'similarity_bias_semantic_similarity_to_stereotype' if 'similarity_bias_semantic_similarity_to_stereotype' in df.columns else 'similarity_turn2_semantic_similarity'
        
        similarity_df = df[df[control_sim_col].notna() & df[bias_sim_col].notna()].copy()
        print(f"Entries with semantic similarity scores: {len(similarity_df)}")
        
        fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size_large)
        
        # 1. Distribution of control semantic similarity to stereotype
        axes[0, 0].hist(similarity_df[control_sim_col], bins=40,
                        edgecolor='black', alpha=0.7, color='blue', label='Control')
        axes[0, 0].hist(similarity_df[bias_sim_col], bins=40,
                        edgecolor='black', alpha=0.7, color='red', label='Bias')
        axes[0, 0].set_xlabel('Semantic Similarity to Stereotype')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title(f'Distribution: Control vs Bias Similarity to Stereotype', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Add statistics
        mean_control = similarity_df[control_sim_col].mean()
        mean_bias = similarity_df[bias_sim_col].mean()
        stats_text = f"Control Mean: {mean_control:.3f}\nBias Mean: {mean_bias:.3f}"
        axes[0, 0].text(0.98, 0.97, stats_text, transform=axes[0, 0].transAxes,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 2. Semantic similarity drift vs HEARTS drift
        if 'drift_drift_score' in similarity_df.columns and 'similarity_semantic_similarity_drift' in similarity_df.columns:
            axes[0, 1].scatter(similarity_df['drift_drift_score'],
                             similarity_df['similarity_semantic_similarity_drift'],
                             alpha=0.4, s=20,
                             c=similarity_df['drift_significant_drift'].map({True: 'red', False: 'blue'})
                             if 'drift_significant_drift' in similarity_df.columns else 'steelblue')
            axes[0, 1].set_xlabel('HEARTS Drift Score')
            axes[0, 1].set_ylabel('Semantic Similarity Drift')
            axes[0, 1].set_title('HEARTS Drift vs Semantic Similarity Drift', fontweight='bold')
            axes[0, 1].axvline(0.2, color='red', linestyle='--', linewidth=1, alpha=0.5)
            axes[0, 1].axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            axes[0, 1].grid(alpha=0.3)
            
            # Calculate correlation
            corr = similarity_df['drift_drift_score'].corr(similarity_df['similarity_semantic_similarity_drift'])
            axes[0, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}',
                           transform=axes[0, 1].transAxes,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 3. Semantic similarity to stereotype by bias type
        if 'bias_type' in similarity_df.columns:
            bias_control_sim = similarity_df.groupby('bias_type')[control_sim_col].mean().sort_values(ascending=False)
            bias_bias_sim = similarity_df.groupby('bias_type')[bias_sim_col].mean().sort_values(ascending=False)
            
            x = np.arange(len(bias_control_sim))
            width = 0.35
            axes[1, 0].bar(x - width/2, bias_control_sim, width, label='Control', color='blue', alpha=0.7, edgecolor='black')
            axes[1, 0].bar(x + width/2, bias_bias_sim, width, label='Bias', color='red', alpha=0.7, edgecolor='black')
            axes[1, 0].set_xlabel('Bias Type')
            axes[1, 0].set_ylabel('Mean Semantic Similarity to Stereotype')
            axes[1, 0].set_title(f'Mean Semantic Similarity by Bias Type: {model_name}', fontweight='bold')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(bias_control_sim.index, rotation=45, ha='right')
            axes[1, 0].legend()
            axes[1, 0].grid(alpha=0.3, axis='y')
        
        # 4. Semantic similarity drift by bias type
        if 'bias_type' in similarity_df.columns and 'similarity_semantic_similarity_drift' in similarity_df.columns:
            bias_sem_drift = similarity_df.groupby('bias_type')['similarity_semantic_similarity_drift'].mean().sort_values(ascending=False)
            colors_bar = ['red' if x > 0 else 'blue' for x in bias_sem_drift.values]
            bias_sem_drift.plot(kind='barh', ax=axes[1, 1], color=colors_bar, edgecolor='black')
            axes[1, 1].axvline(0, color='black', linestyle='-', linewidth=1)
            axes[1, 1].set_xlabel('Mean Semantic Similarity Drift')
            axes[1, 1].set_ylabel('Bias Type')
            axes[1, 1].set_title(f'Mean Semantic Similarity Drift by Bias Type: {model_name}', fontweight='bold')
            axes[1, 1].grid(alpha=0.3, axis='x')
            plt.setp(axes[1, 1].yaxis.get_majorticklabels(), rotation=0)
        
        plt.tight_layout()
        output_file = output_dir / 'stereotype_similarity_analysis.png'
        plt.savefig(output_file, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        print(f"OK: Saved: {output_file}")
    
    def visualize_trait_mention(self, df: pd.DataFrame, model_name: str, output_dir: Path):
        """Generate trait mention analysis visualization."""
        if 'similarity_trait_mentioned' not in df.columns:
            print("WARNING: Trait mention data not available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=self.config.figure_size)
        
        # 1. Overall trait mention rate
        trait_mention_rate = df['similarity_trait_mentioned'].mean()
        labels = ['Trait Mentioned', 'Trait Not Mentioned']
        sizes = [trait_mention_rate, 1 - trait_mention_rate]
        colors = ['#ff9999', '#66b3ff']
        explode = (0.1, 0)
        
        axes[0].pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                   shadow=True, startangle=90)
        axes[0].set_title('Overall Trait Mention Rate', fontweight='bold', pad=20)
        
        # 2. Trait mention rate by bias type
        if 'bias_type' in df.columns:
            trait_by_bias = df.groupby('bias_type')['similarity_trait_mentioned'].mean().sort_values(ascending=False)
            trait_by_bias.plot(kind='barh', ax=axes[1], color='lightcoral', edgecolor='black')
            axes[1].set_xlabel('Trait Mention Rate')
            axes[1].set_ylabel('Bias Type')
            axes[1].set_title(f'Trait Mention Rate by Bias Type: {model_name}', fontweight='bold')
            axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
            axes[1].grid(alpha=0.3, axis='x')
            plt.setp(axes[1].yaxis.get_majorticklabels(), rotation=0)
        
        plt.tight_layout()
        output_file = output_dir / 'trait_mention_analysis.png'
        plt.savefig(output_file, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        print(f"OK: Saved: {output_file}")
        print(f"\nOverall trait mention rate: {trait_mention_rate:.2%}")
    
    
    def visualize_raw_drift_analysis(self, df: pd.DataFrame, model_name: str, output_dir: Path):
        """Generate raw drift analysis visualization showing direction of bias influence."""
        if 'drift_raw_drift' not in df.columns:
            print("WARNING: Raw drift scores not available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size_large)
        
        # 1. Distribution of raw drift (showing positive vs negative)
        axes[0, 0].hist(df['drift_raw_drift'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='No change (y=0)')
        axes[0, 0].set_xlabel('Raw Drift Score (Bias - Control)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title(f'Raw Drift Distribution: {model_name}', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Add statistics
        mean_raw = df['drift_raw_drift'].mean()
        positive_count = (df['drift_raw_drift'] > 0).sum()
        negative_count = (df['drift_raw_drift'] < 0).sum()
        zero_count = (df['drift_raw_drift'] == 0).sum()
        stats_text = f"Mean: {mean_raw:.3f}\nPositive: {positive_count} ({positive_count/len(df):.1%})\nNegative: {negative_count} ({negative_count/len(df):.1%})"
        axes[0, 0].text(0.98, 0.97, stats_text, transform=axes[0, 0].transAxes,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                       fontsize=10)
        
        # 2. Box plot by bias type (raw drift)
        if 'bias_type' in df.columns:
            df.boxplot(column='drift_raw_drift', by='bias_type', ax=axes[0, 1])
            axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=2, label='No change')
            axes[0, 1].set_xlabel('Bias Type')
            axes[0, 1].set_ylabel('Raw Drift Score')
            axes[0, 1].set_title(f'Raw Drift by Bias Type: {model_name}', fontweight='bold')
            axes[0, 1].get_figure().suptitle('')
            plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
            axes[0, 1].legend()
            axes[0, 1].grid(alpha=0.3)
        
        # 3. Scatter plot: Control vs Bias scores with direction
        if 'drift_control_stereotype_score' in df.columns and 'drift_bias_stereotype_score' in df.columns:
            # Color by direction
            colors = df['drift_raw_drift'].apply(lambda x: 'red' if x > 0 else ('blue' if x < 0 else 'gray'))
            axes[1, 0].scatter(df['drift_control_stereotype_score'], 
                              df['drift_bias_stereotype_score'],
                              c=colors, alpha=0.5, s=20)
            axes[1, 0].plot([0, 1], [0, 1], 'k--', linewidth=2, label='No change line (y=x)')
            axes[1, 0].set_xlabel('Control Stereotype Score')
            axes[1, 0].set_ylabel('Bias Stereotype Score')
            axes[1, 0].set_title(f'Control vs Bias Scores (Direction): {model_name}', fontweight='bold')
            axes[1, 0].legend(['No change', 'Positive drift', 'Negative drift'])
            axes[1, 0].grid(alpha=0.3)
            axes[1, 0].set_xlim(0, 1)
            axes[1, 0].set_ylim(0, 1)
        
        # 4. Bar chart: Positive vs Negative drift by bias type
        if 'bias_type' in df.columns:
            bias_drift = df.groupby('bias_type')['drift_raw_drift'].mean().sort_values(ascending=False)
            colors_bar = ['red' if x > 0 else 'blue' for x in bias_drift.values]
            bias_drift.plot(kind='bar', ax=axes[1, 1], color=colors_bar, edgecolor='black')
            axes[1, 1].axhline(0, color='black', linestyle='-', linewidth=1)
            axes[1, 1].set_xlabel('Bias Type')
            axes[1, 1].set_ylabel('Mean Raw Drift Score')
            axes[1, 1].set_title(f'Mean Raw Drift by Bias Type: {model_name}', fontweight='bold')
            plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
            axes[1, 1].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_file = output_dir / 'raw_drift_analysis.png'
        plt.savefig(output_file, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        print(f"OK: Saved: {output_file}")
    
    def visualize_semantic_similarity_drift_direction(self, df: pd.DataFrame, model_name: str, output_dir: Path):
        """Generate semantic similarity drift direction visualization."""
        if 'similarity_semantic_similarity_drift' not in df.columns:
            print("WARNING: Semantic similarity drift not available")
            return
        
        semantic_drift_df = df[df['similarity_semantic_similarity_drift'].notna()].copy()
        if semantic_drift_df.empty:
            print("WARNING: No valid semantic similarity drift data")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=self.config.figure_size)
        
        # 1. Distribution of semantic similarity drift
        axes[0].hist(semantic_drift_df['similarity_semantic_similarity_drift'], bins=50, 
                    edgecolor='black', alpha=0.7, color='mediumpurple')
        axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='No change (y=0)')
        axes[0].set_xlabel('Semantic Similarity Drift (Control - Bias)')
        axes[0].set_ylabel('Count')
        axes[0].set_title(f'Semantic Similarity Drift Distribution: {model_name}', fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Add statistics
        mean_sem_drift = semantic_drift_df['similarity_semantic_similarity_drift'].mean()
        positive_sem = (semantic_drift_df['similarity_semantic_similarity_drift'] > 0).sum()
        negative_sem = (semantic_drift_df['similarity_semantic_similarity_drift'] < 0).sum()
        stats_text = f"Mean: {mean_sem_drift:.3f}\nPositive: {positive_sem} ({positive_sem/len(semantic_drift_df):.1%})\nNegative: {negative_sem} ({negative_sem/len(semantic_drift_df):.1%})"
        axes[0].text(0.98, 0.97, stats_text, transform=axes[0].transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=10)
        
        # 2. Bar chart by bias type
        if 'bias_type' in semantic_drift_df.columns:
            bias_sem_drift = semantic_drift_df.groupby('bias_type')['similarity_semantic_similarity_drift'].mean().sort_values(ascending=False)
            colors_bar = ['red' if x > 0 else 'blue' for x in bias_sem_drift.values]
            bias_sem_drift.plot(kind='bar', ax=axes[1], color=colors_bar, edgecolor='black')
            axes[1].axhline(0, color='black', linestyle='-', linewidth=1)
            axes[1].set_xlabel('Bias Type')
            axes[1].set_ylabel('Mean Semantic Similarity Drift')
            axes[1].set_title(f'Mean Semantic Similarity Drift by Bias Type: {model_name}', fontweight='bold')
            plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
            axes[1].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_file = output_dir / 'semantic_similarity_drift_direction.png'
        plt.savefig(output_file, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        print(f"OK: Saved: {output_file}")
    
    def generate_all_visualizations(self, model_name: str, date: Optional[str] = None):
        """
        Generate all visualizations for a model.
        
        Args:
            model_name: Name of the model
            date: Date string (YYYYMMDD) or None for latest
        """
        print(f"\n{'='*70}")
        print(f"Generating visualizations for: {model_name}")
        print(f"{'='*70}")
        
        # Load data
        df = self.load_data(model_name, date)
        if df is None:
            print(f"WARNING: No data found for model: {model_name}")
            return
        
        # Get output directory
        output_dir = self.get_output_dir(model_name, date)
        if output_dir is None:
            print(f"WARNING: Could not determine output directory for model: {model_name}")
            return
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Filter valid entries
        valid_df = self.filter_valid_entries(df)
        print(f"Valid entries: {len(valid_df)} ({len(valid_df)/len(df)*100:.1f}%)")
        
        # Generate all visualizations
        self.visualize_drift_distribution(valid_df, model_name, output_dir)
        self.visualize_control_vs_bias(valid_df, model_name, output_dir)
        self.visualize_drift_by_bias_type(valid_df, model_name, output_dir)
        self.visualize_drift_by_stereotype_type(valid_df, model_name, output_dir)
        self.visualize_heatmaps(valid_df, model_name, output_dir)
        self.visualize_similarity_analysis(valid_df, model_name, output_dir)
        self.visualize_trait_mention(valid_df, model_name, output_dir)
        self.visualize_raw_drift_analysis(valid_df, model_name, output_dir)
        self.visualize_semantic_similarity_drift_direction(valid_df, model_name, output_dir)
        
        print(f"\nOK: Completed visualizations for {model_name}")
    
    def process_all_models(self):
        """Process all available models."""
        models = self.find_available_models()
        if not models:
            print("WARNING: No models found in drift results directory")
            return
        
        print(f"Found {len(models)} models to process")
        
        for model_name in models:
            if self.config.model_name is None or model_name == self.config.model_name:
                self.generate_all_visualizations(model_name, self.config.date)

