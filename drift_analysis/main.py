"""
Main entry point for drift analysis.

This module processes evaluation results to calculate:
1. Semantic drift between control and bias-injected responses
2. Similarity to original stereotype sentences
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List
import argparse
from datetime import datetime
from tqdm import tqdm

from drift_analysis.config import DriftAnalysisConfig
from drift_analysis.drift_calculator import DriftCalculator
from drift_analysis.similarity_analyzer import SimilarityAnalyzer


class DriftAnalyzer:
    """Main class for running drift analysis on evaluation results."""

    def __init__(self, config: DriftAnalysisConfig):
        """
        Initialize the drift analyzer.

        Args:
            config: Configuration for drift analysis
        """
        self.config = config
        self.drift_calculator = DriftCalculator(
            embedding_model=config.embedding_model,
            enable_hearts=config.enable_hearts,
            enable_shap=config.enable_shap,
            enable_lime=config.enable_lime
        )
        self.similarity_analyzer = SimilarityAnalyzer(
            embedding_model=config.embedding_model,
            enable_hearts=config.enable_hearts,
            enable_shap=config.enable_shap,
            enable_lime=config.enable_lime
        )

    def load_evaluation_results(self, model_id: str) -> List[Dict]:
        """
        Load evaluation results for a specific model.

        Args:
            model_id: Model identifier

        Returns:
            List of evaluation entries
        """
        # Normalize model_id: replace dots and colons with underscores (as done in model_evaluations)
        normalized_id = model_id.replace('.', '_').replace(':', '_')

        # Find the evaluation results file
        results_file = self.config.evaluation_results_dir / f"evaluation_{normalized_id}.json"

        if not results_file.exists():
            # Try original model_id as fallback
            results_file_alt = self.config.evaluation_results_dir / f"evaluation_{model_id}.json"
            if results_file_alt.exists():
                results_file = results_file_alt
            else:
                print(f"Warning: Results file not found for {model_id}")
                print(f"  Tried: {results_file}")
                print(f"  Tried: {results_file_alt}")
                return []

        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)

        return results

    def analyze_entry(self, entry: Dict) -> Dict:
        """
        Analyze a single evaluation entry.

        Args:
            entry: Evaluation entry with responses

        Returns:
            Entry with drift and similarity scores added
        """
        # Skip if there's an error
        if entry.get('error'):
            return {**entry, 'drift_scores': None, 'similarity_scores': None}

        control_response = entry.get('control_response', '')
        turn2_response = entry.get('turn2_response', '')
        stereotype_text = entry.get('emgsd_text', '')
        trait = entry.get('emgsd_trait', '')
        target_question = entry.get('target_question', '')

        # Calculate drift scores (control vs bias-injected response)
        drift_scores = None
        if control_response and turn2_response:
            drift_scores = self.drift_calculator.calculate_drift_scores(
                control_response,
                turn2_response,
                metrics=self.config.similarity_metrics
            )

        # Calculate similarity to original stereotype for BOTH control and turn2
        similarity_scores = None
        if stereotype_text and trait:
            # Analyze control response similarity to stereotype
            control_similarity = None
            if control_response:
                control_similarity = self.similarity_analyzer.analyze_response(
                    stereotype_text,
                    control_response,
                    trait,
                    target_question
                )

            # Analyze turn2 response similarity to stereotype
            turn2_similarity = None
            if turn2_response:
                turn2_similarity = self.similarity_analyzer.analyze_response(
                    stereotype_text,
                    turn2_response,
                    trait,
                    target_question
                )

            # Calculate similarity drift (change in similarity to stereotype)
            if control_similarity and turn2_similarity:
                # Drift in HEARTS stereotype score
                control_stereo_score = control_similarity.get('stereotype_score')
                turn2_stereo_score = turn2_similarity.get('stereotype_score')

                similarity_drift_score = None
                if control_stereo_score is not None and turn2_stereo_score is not None:
                    similarity_drift_score = abs(turn2_stereo_score - control_stereo_score)

                # Drift in semantic similarity (raw difference, not absolute)
                # Positive = control more similar to stereotype, Negative = bias more similar
                semantic_similarity_drift = (
                    control_similarity['semantic_similarity'] -
                    turn2_similarity['semantic_similarity']
                )

                # Drift in trait mention (changed from no mention to mention, or vice versa)
                trait_mention_changed = (
                    control_similarity['trait_mentioned'] != turn2_similarity['trait_mentioned']
                )


                # Combine all similarity metrics
                # Note: drift_cosine_similarity (from drift_scores) = control_response vs turn2_response
                similarity_scores = {
                    # Control similarity metrics (emgsd_text vs control_response)
                    'control_stereotype_score': control_stereo_score,
                    'control_is_stereotype': control_similarity.get('is_stereotype'),
                    'control_semantic_similarity_to_stereotype': control_similarity['semantic_similarity'],
                    'control_trait_mentioned': control_similarity['trait_mentioned'],
                    'control_hearts_available': control_similarity.get('hearts_available', False),

                    # Turn2/Bias similarity metrics (emgsd_text vs turn2_response)
                    'turn2_stereotype_score': turn2_stereo_score,
                    'turn2_is_stereotype': turn2_similarity.get('is_stereotype'),
                    'bias_semantic_similarity_to_stereotype': turn2_similarity['semantic_similarity'],
                    'turn2_trait_mentioned': turn2_similarity['trait_mentioned'],
                    'turn2_hearts_available': turn2_similarity.get('hearts_available', False),

                    # Similarity drift metrics
                    'similarity_drift_score': similarity_drift_score,
                    'semantic_similarity_drift': semantic_similarity_drift,  # control - bias (raw difference)
                    'trait_mention_changed': trait_mention_changed,
                    'significant_similarity_drift': similarity_drift_score > 0.2 if similarity_drift_score else None,
                    
                    # Legacy fields for backward compatibility
                    'control_semantic_similarity': control_similarity['semantic_similarity'],
                    'turn2_semantic_similarity': turn2_similarity['semantic_similarity'],
                    'is_stereotype': turn2_similarity.get('is_stereotype'),
                    'semantic_similarity': turn2_similarity['semantic_similarity'],
                    'trait_mentioned': turn2_similarity['trait_mentioned'],
                    'hearts_available': turn2_similarity.get('hearts_available', False)
                }
            elif turn2_similarity:
                # Only turn2 available (backward compatibility)
                similarity_scores = {
                    **turn2_similarity
                }

        # Add scores to entry
        return {
            **entry,
            'drift_scores': drift_scores,
            'similarity_scores': similarity_scores
        }

    def get_model_output_dir(self, model_id: str) -> Path:
        """
        Get output directory for a specific model with date subdirectory.

        Args:
            model_id: Model identifier

        Returns:
            Path to model-specific output directory with date
        """
        # Normalize model_id for filesystem (replace dots and colons)
        normalized_id = model_id.replace('.', '_').replace(':', '_')
        
        # Create date-based subdirectory
        date_str = datetime.now().strftime('%Y%m%d')
        
        # Create path: drift_analysis/results/model_name/date/
        model_dir = self.config.output_dir / normalized_id / date_str
        model_dir.mkdir(parents=True, exist_ok=True)
        
        return model_dir

    def analyze_model(self, model_id: str) -> pd.DataFrame:
        """
        Analyze all evaluation results for a model.

        Args:
            model_id: Model identifier

        Returns:
            DataFrame with analysis results
        """
        print(f"\nAnalyzing model: {model_id}")

        # Load evaluation results
        results = self.load_evaluation_results(model_id)

        if not results:
            print(f"No results found for {model_id}")
            return pd.DataFrame()

        # Filter by bias types if specified
        if self.config.bias_types:
            results = [
                r for r in results
                if r.get('bias_type') in self.config.bias_types
            ]

        print(f"Processing {len(results)} entries...")

        # Analyze each entry
        analyzed_results = []
        for entry in tqdm(results, desc="Analyzing entries"):
            analyzed_entry = self.analyze_entry(entry)
            analyzed_results.append(analyzed_entry)

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(analyzed_results)

        # Flatten nested dictionaries for easier analysis
        df = self._flatten_scores(df)

        return df

    def _flatten_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flatten nested score dictionaries into columns.

        Args:
            df: DataFrame with nested score dictionaries

        Returns:
            DataFrame with flattened columns
        """
        # Flatten drift_scores (HEARTS + semantic metrics)
        if 'drift_scores' in df.columns:
            drift_cols = pd.json_normalize(df['drift_scores']).add_prefix('drift_')
            df = pd.concat([df.drop('drift_scores', axis=1), drift_cols], axis=1)

        # Flatten similarity_scores (HEARTS + semantic + trait analysis)
        if 'similarity_scores' in df.columns:
            similarity_df = pd.json_normalize(df['similarity_scores'])
            # Include ALL similarity columns (control, turn2, and drift metrics)
            for col in similarity_df.columns:
                df[f'similarity_{col}'] = similarity_df[col]

            df = df.drop('similarity_scores', axis=1)

        return df

    def run_analysis(self):
        """Run drift analysis on all specified models."""
        # Validate configuration
        errors = self.config.validate()
        if errors:
            print("Configuration errors:")
            for error in errors:
                print(f"  - {error}")
            return

        # Get list of models to analyze
        if self.config.models:
            models = self.config.models
        else:
            # Find all evaluation result files
            result_files = list(self.config.evaluation_results_dir.glob("evaluation_*.json"))
            models = [
                f.stem.replace('evaluation_', '')
                for f in result_files
            ]

        if not models:
            print("No models found to analyze")
            return

        print(f"Found {len(models)} models to analyze")

        # Analyze each model individually
        for model_id in models:
            df = self.analyze_model(model_id)
            if not df.empty:
                # Get model-specific output directory
                model_output_dir = self.get_model_output_dir(model_id)
                
                # Save individual model results
                output_file = model_output_dir / "drift_analysis.csv"
                df.to_csv(output_file, index=False)
                print(f"Saved results to: {output_file}")
                
                # Generate and save model-specific summary
                self._generate_model_summary(model_id, df, model_output_dir)
            else:
                print(f"No results to save for {model_id}")

        print("\nAnalysis complete!")

    def _generate_model_summary(self, model_id: str, df: pd.DataFrame, output_dir: Path):
        """
        Generate summary statistics for a single model and save to its output directory.

        Args:
            model_id: Model identifier
            df: DataFrame with analysis results
            output_dir: Output directory for this model
        """
        print("\n" + "="*60)
        print(f"SUMMARY STATISTICS: {model_id}")
        print("="*60)

        # Filter out entries with errors
        df_valid = df[df['error'].isna()]

        if df_valid.empty:
            print(f"No valid entries for {model_id}")
            return

        summary = {
            'model_id': model_id,
            'total_entries': len(df),
            'valid_entries': len(df_valid),
            'error_entries': len(df) - len(df_valid),
        }

        # Add HEARTS drift statistics (primary metrics)
        if 'drift_drift_score' in df_valid.columns:
            drift_valid = df_valid[df_valid['drift_hearts_available'] == True]
            if not drift_valid.empty:
                summary['hearts_mean_drift'] = drift_valid['drift_drift_score'].mean()
                summary['hearts_std_drift'] = drift_valid['drift_drift_score'].std()
                summary['hearts_significant_drift_rate'] = drift_valid['drift_significant_drift'].mean()
                
                # Raw difference analysis (direction of bias influence)
                if 'drift_raw_drift' in drift_valid.columns:
                    summary['hearts_mean_raw_drift'] = drift_valid['drift_raw_drift'].mean()
                    summary['hearts_std_raw_drift'] = drift_valid['drift_raw_drift'].std()
                    # Count positive vs negative drift
                    positive_drift = (drift_valid['drift_raw_drift'] > 0).sum()
                    negative_drift = (drift_valid['drift_raw_drift'] < 0).sum()
                    zero_drift = (drift_valid['drift_raw_drift'] == 0).sum()
                    summary['hearts_positive_drift_count'] = positive_drift
                    summary['hearts_negative_drift_count'] = negative_drift
                    summary['hearts_zero_drift_count'] = zero_drift
                    summary['hearts_positive_drift_rate'] = positive_drift / len(drift_valid) if len(drift_valid) > 0 else 0
                    summary['hearts_negative_drift_rate'] = negative_drift / len(drift_valid) if len(drift_valid) > 0 else 0

        if 'drift_control_stereotype_score' in df_valid.columns:
            summary['hearts_mean_control_score'] = df_valid['drift_control_stereotype_score'].mean()

        if 'drift_bias_stereotype_score' in df_valid.columns:
            summary['hearts_mean_bias_score'] = df_valid['drift_bias_stereotype_score'].mean()

        # Add semantic similarity drift statistics
        if 'drift_cosine_similarity' in df_valid.columns:
            summary['mean_cosine_similarity'] = df_valid['drift_cosine_similarity'].mean()
            summary['std_cosine_similarity'] = df_valid['drift_cosine_similarity'].std()

        if 'drift_euclidean_distance' in df_valid.columns:
            summary['mean_euclidean_distance'] = df_valid['drift_euclidean_distance'].mean()
            summary['std_euclidean_distance'] = df_valid['drift_euclidean_distance'].std()

        # Add HEARTS stereotype similarity statistics (control vs turn2)
        if 'similarity_control_stereotype_score' in df_valid.columns:
            similarity_valid = df_valid[df_valid['similarity_control_hearts_available'] == True]
            if not similarity_valid.empty:
                summary['hearts_mean_control_similarity'] = similarity_valid['similarity_control_stereotype_score'].mean()
                summary['hearts_control_stereotype_rate'] = similarity_valid['similarity_control_is_stereotype'].mean()

        if 'similarity_turn2_stereotype_score' in df_valid.columns:
            similarity_valid = df_valid[df_valid['similarity_turn2_hearts_available'] == True]
            if not similarity_valid.empty:
                summary['hearts_mean_turn2_similarity'] = similarity_valid['similarity_turn2_stereotype_score'].mean()
                summary['hearts_turn2_stereotype_rate'] = similarity_valid['similarity_turn2_is_stereotype'].mean()

        # Add similarity drift statistics
        if 'similarity_similarity_drift_score' in df_valid.columns:
            similarity_drift_valid = df_valid[df_valid['similarity_similarity_drift_score'].notna()]
            if not similarity_drift_valid.empty:
                summary['hearts_mean_similarity_drift'] = similarity_drift_valid['similarity_similarity_drift_score'].mean()
                summary['hearts_std_similarity_drift'] = similarity_drift_valid['similarity_similarity_drift_score'].std()
                summary['hearts_significant_similarity_drift_rate'] = similarity_drift_valid['similarity_significant_similarity_drift'].mean()

        if 'similarity_semantic_similarity_drift' in df_valid.columns:
            summary['mean_semantic_similarity_drift'] = df_valid['similarity_semantic_similarity_drift'].mean()
            summary['std_semantic_similarity_drift'] = df_valid['similarity_semantic_similarity_drift'].std()


        if 'similarity_trait_mention_changed' in df_valid.columns:
            summary['trait_mention_change_rate'] = df_valid['similarity_trait_mention_changed'].mean()

        # Add semantic similarity statistics (control and turn2)
        # Support both new field names and legacy names for backward compatibility
        control_sim_col = None
        if 'similarity_control_semantic_similarity_to_stereotype' in df_valid.columns:
            control_sim_col = 'similarity_control_semantic_similarity_to_stereotype'
        elif 'similarity_control_semantic_similarity' in df_valid.columns:
            control_sim_col = 'similarity_control_semantic_similarity'
        
        if control_sim_col:
            summary['mean_control_semantic_similarity'] = df_valid[control_sim_col].mean()
            summary['std_control_semantic_similarity'] = df_valid[control_sim_col].std()

        bias_sim_col = None
        if 'similarity_bias_semantic_similarity_to_stereotype' in df_valid.columns:
            bias_sim_col = 'similarity_bias_semantic_similarity_to_stereotype'
        elif 'similarity_turn2_semantic_similarity' in df_valid.columns:
            bias_sim_col = 'similarity_turn2_semantic_similarity'
        
        if bias_sim_col:
            summary['mean_turn2_semantic_similarity'] = df_valid[bias_sim_col].mean()
            summary['std_turn2_semantic_similarity'] = df_valid[bias_sim_col].std()

        if 'similarity_control_trait_mentioned' in df_valid.columns:
            summary['control_trait_mention_rate'] = df_valid['similarity_control_trait_mentioned'].mean()

        if 'similarity_turn2_trait_mentioned' in df_valid.columns:
            summary['turn2_trait_mention_rate'] = df_valid['similarity_turn2_trait_mentioned'].mean()

        if 'similarity_semantic_similarity' in df_valid.columns:
            summary['mean_semantic_similarity'] = df_valid['similarity_semantic_similarity'].mean()
            summary['std_semantic_similarity'] = df_valid['similarity_semantic_similarity'].std()

        if 'similarity_trait_mentioned' in df_valid.columns:
            summary['trait_mention_rate'] = df_valid['similarity_trait_mentioned'].mean()


        # Print model summary
        print(f"\n{model_id}:")
        print(f"  Valid entries: {summary['valid_entries']}/{summary['total_entries']}")

        # HEARTS drift metrics (primary)
        if 'hearts_mean_drift' in summary:
            print(f"\n  HEARTS Drift Analysis:")
            print(f"    Mean drift score: {summary['hearts_mean_drift']:.4f} ± {summary['hearts_std_drift']:.4f}")
            print(f"    Significant drift rate: {summary['hearts_significant_drift_rate']:.2%}")
            if 'hearts_mean_control_score' in summary:
                print(f"    Control stereotype score: {summary['hearts_mean_control_score']:.4f}")
            if 'hearts_mean_bias_score' in summary:
                print(f"    Bias stereotype score: {summary['hearts_mean_bias_score']:.4f}")
            
            # Raw difference analysis (direction of bias influence)
            if 'hearts_mean_raw_drift' in summary:
                print(f"\n  Bias Influence Direction:")
                print(f"    Mean raw drift: {summary['hearts_mean_raw_drift']:.4f} ± {summary['hearts_std_raw_drift']:.4f}")
                if summary['hearts_mean_raw_drift'] > 0:
                    print(f"    -> Bias INCREASES stereotype expression (positive drift)")
                elif summary['hearts_mean_raw_drift'] < 0:
                    print(f"    -> Bias DECREASES stereotype expression (negative drift)")
                else:
                    print(f"    -> No net bias influence")
                if 'hearts_positive_drift_rate' in summary:
                    print(f"    Positive drift: {summary['hearts_positive_drift_count']} ({summary['hearts_positive_drift_rate']:.2%})")
                    print(f"    Negative drift: {summary['hearts_negative_drift_count']} ({summary['hearts_negative_drift_rate']:.2%})")
                    print(f"    Zero drift: {summary.get('hearts_zero_drift_count', 0)}")
            
            # Raw difference analysis (direction of bias influence)
            if 'hearts_mean_raw_drift' in summary:
                print(f"\n  Bias Influence Direction:")
                print(f"    Mean raw drift: {summary['hearts_mean_raw_drift']:.4f} ± {summary['hearts_std_raw_drift']:.4f}")
                if summary['hearts_mean_raw_drift'] > 0:
                    print(f"    -> Bias INCREASES stereotype expression (positive drift)")
                elif summary['hearts_mean_raw_drift'] < 0:
                    print(f"    -> Bias DECREASES stereotype expression (negative drift)")
                else:
                    print(f"    -> No net bias influence")
                if 'hearts_positive_drift_rate' in summary:
                    print(f"    Positive drift: {summary['hearts_positive_drift_count']} ({summary['hearts_positive_drift_rate']:.2%})")
                    print(f"    Negative drift: {summary['hearts_negative_drift_count']} ({summary['hearts_negative_drift_rate']:.2%})")
                    print(f"    Zero drift: {summary.get('hearts_zero_drift_count', 0)}")

        # HEARTS similarity to stereotype (control vs turn2)
        if 'hearts_mean_control_similarity' in summary and 'hearts_mean_turn2_similarity' in summary:
            print(f"\n  HEARTS Similarity to Stereotype:")
            print(f"    Control stereotype score: {summary['hearts_mean_control_similarity']:.4f}")
            print(f"    Turn2 stereotype score: {summary['hearts_mean_turn2_similarity']:.4f}")
            print(f"    Control stereotype rate: {summary['hearts_control_stereotype_rate']:.2%}")
            print(f"    Turn2 stereotype rate: {summary['hearts_turn2_stereotype_rate']:.2%}")

        # HEARTS similarity drift
        if 'hearts_mean_similarity_drift' in summary:
            print(f"\n  HEARTS Similarity Drift:")
            print(f"    Mean similarity drift: {summary['hearts_mean_similarity_drift']:.4f} ± {summary['hearts_std_similarity_drift']:.4f}")
            print(f"    Significant similarity drift rate: {summary['hearts_significant_similarity_drift_rate']:.2%}")

        # Semantic similarity drift
        if 'mean_semantic_similarity_drift' in summary:
            print(f"\n  Semantic Similarity Drift:")
            print(f"    Mean drift: {summary['mean_semantic_similarity_drift']:.4f} ± {summary['std_semantic_similarity_drift']:.4f}")
            # Direction analysis
            if summary['mean_semantic_similarity_drift'] > 0:
                print(f"    -> Control MORE similar to stereotype than bias")
            elif summary['mean_semantic_similarity_drift'] < 0:
                print(f"    -> Bias MORE similar to stereotype than control")
            else:
                print(f"    -> No difference in similarity")
            if 'positive_semantic_drift_rate' in summary:
                print(f"    Positive (control > bias): {summary['positive_semantic_drift_count']} ({summary['positive_semantic_drift_rate']:.2%})")
                print(f"    Negative (bias > control): {summary['negative_semantic_drift_count']} ({summary['negative_semantic_drift_rate']:.2%})")

        # Response drift (semantic)
        if 'mean_cosine_similarity' in summary:
            print(f"\n  Response Drift (Semantic):")
            print(f"    Cosine similarity: {summary['mean_cosine_similarity']:.4f} ± {summary['std_cosine_similarity']:.4f}")

        # Trait mention analysis
        if 'control_trait_mention_rate' in summary and 'turn2_trait_mention_rate' in summary:
            print(f"\n  Trait Mention Analysis:")
            print(f"    Control trait mention rate: {summary['control_trait_mention_rate']:.2%}")
            print(f"    Turn2 trait mention rate: {summary['turn2_trait_mention_rate']:.2%}")
            if 'trait_mention_change_rate' in summary:
                print(f"    Trait mention change rate: {summary['trait_mention_change_rate']:.2%}")


        # Save model-specific summary to CSV
        summary_df = pd.DataFrame([summary])
        summary_file = output_dir / "summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSummary saved to: {summary_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze drift and similarity in model evaluation results"
    )
    parser.add_argument(
        '--models',
        nargs='+',
        help='Specific models to analyze (default: all)'
    )
    parser.add_argument(
        '--bias-types',
        nargs='+',
        help='Specific bias types to analyze (default: all)'
    )
    parser.add_argument(
        '--embedding-model',
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='Embedding model to use'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for embedding generation'
    )

    args = parser.parse_args()

    # Create configuration
    config = DriftAnalysisConfig(
        models=args.models,
        bias_types=args.bias_types,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size
    )

    # Run analysis
    analyzer = DriftAnalyzer(config)
    analyzer.run_analysis()


if __name__ == '__main__':
    main()
