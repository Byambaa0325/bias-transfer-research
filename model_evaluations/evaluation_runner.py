"""
Runs evaluation for all models on the dataset.
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from tqdm import tqdm
import json

from .config import EvaluationConfig
from .evaluator import ModelEvaluator

# Import parallel processor
try:
    from .parallel_processor import ParallelProcessor
except ImportError:
    # Fallback: import from dataset_generation
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dataset_generation.parallel_processor import ParallelProcessor


class EvaluationRunner:
    """
    Orchestrates evaluation across all models.
    """
    
    def __init__(self, config: EvaluationConfig):
        """
        Initialize evaluation runner.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.results: Dict[str, List[Dict[str, Any]]] = {}  # Results per model
        
    def load_dataset(self) -> pd.DataFrame:
        """Load the multi-turn EMGSD dataset."""
        dataset_path = self.config.dataset_path
        
        # If path is a directory, find the latest dataset file
        if dataset_path.is_dir():
            dataset_files = list(dataset_path.glob('multiturn_emgsd_dataset_*.csv'))
            if not dataset_files:
                raise FileNotFoundError(
                    f"No dataset files found in: {dataset_path}\n"
                    f"Expected pattern: multiturn_emgsd_dataset_*.csv"
                )
            # Get the latest file (by modification time)
            dataset_path = max(dataset_files, key=lambda p: p.stat().st_mtime)
            print(f"Found latest dataset: {dataset_path.name}")
        
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {dataset_path}\n"
                f"Please specify the correct path with --dataset"
            )
        
        df = pd.read_csv(dataset_path)
        print(f"✓ Loaded dataset: {len(df):,} entries from {dataset_path.name}")
        
        # Verify required columns exist
        required_cols = ['target_question'] + [f'turn1_question_{bt}' for bt in self.config.bias_types]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Dataset missing required columns: {missing_cols}\n"
                f"Available columns: {list(df.columns)[:10]}..."
            )
        
        # Apply sample limit if specified
        if self.config.sample_limit:
            df = df.head(self.config.sample_limit)
            print(f"✓ Limited to {len(df):,} entries")
        
        return df
    
    def run_evaluation(self):
        """Run evaluation for all models."""
        # Load dataset
        df = self.load_dataset()
        
        print(f"\n{'='*70}")
        print(f"MODEL EVALUATION")
        print(f"{'='*70}")
        print(f"Models to evaluate: {len(self.config.models)}")
        print(f"Entries per model: {len(df):,}")
        print(f"Bias types: {len(self.config.bias_types)}")
        print(f"Total evaluations: {len(self.config.models) * len(df) * len(self.config.bias_types):,}")
        print()
        
        # Evaluate each model
        failed_models = []
        for model_id in self.config.models:
            print(f"\n{'='*70}")
            print(f"Evaluating: {model_id}")
            print(f"{'='*70}")
            
            try:
                self._evaluate_model(model_id, df)
                
                # Save results for this model
                self._save_model_results(model_id)
                print(f"✓ Successfully completed evaluation for {model_id}")
                
            except Exception as e:
                print(f"\n❌ Error evaluating {model_id}: {str(e)}")
                print(f"⚠️  Skipping this model and continuing with others...")
                failed_models.append((model_id, str(e)))
                # Continue with next model
                continue
        
        print(f"\n{'='*70}")
        print("EVALUATION COMPLETE")
        print(f"{'='*70}")
        
        # Report failed models
        if failed_models:
            print(f"\n⚠️  Failed Models ({len(failed_models)}):")
            for model_id, error in failed_models:
                print(f"  • {model_id}")
                print(f"    Error: {error[:100]}..." if len(error) > 100 else f"    Error: {error}")
        
        successful = len(self.config.models) - len(failed_models)
        print(f"\n✓ Successfully evaluated: {successful}/{len(self.config.models)} models")
    
    def _evaluate_model(self, model_id: str, df: pd.DataFrame):
        """Evaluate a single model on all entries."""
        evaluator = ModelEvaluator(model_id)
        model_results = []
        
        # Process entries
        if self.config.use_parallel:
            self._evaluate_parallel(evaluator, df, model_results)
        else:
            self._evaluate_sequential(evaluator, df, model_results)
        
        self.results[model_id] = model_results
    
    def _evaluate_sequential(
        self,
        evaluator: ModelEvaluator,
        df: pd.DataFrame,
        results: List[Dict[str, Any]]
    ):
        """Evaluate sequentially."""
        total_tasks = len(df) * len(self.config.bias_types)
        
        with tqdm(total=total_tasks, desc=f"Processing {evaluator.model_id}") as pbar:
            for idx, entry in df.iterrows():
                for bias_type in self.config.bias_types:
                    result = evaluator.evaluate_entry(entry, bias_type)
                    results.append(result)
                    pbar.update(1)
                    
                    # Checkpoint
                    if len(results) % self.config.checkpoint_interval == 0:
                        self._save_checkpoint(evaluator.model_id, results)
    
    def _evaluate_parallel(
        self,
        evaluator: ModelEvaluator,
        df: pd.DataFrame,
        results: List[Dict[str, Any]]
    ):
        """Evaluate in parallel."""
        # Prepare tasks
        tasks = []
        for idx, entry in df.iterrows():
            for bias_type in self.config.bias_types:
                tasks.append((evaluator.evaluate_entry, (entry, bias_type), {}))
        
        # Initialize parallel processor
        processor = ParallelProcessor(
            max_workers=self.config.max_workers,
            max_requests_per_second=self.config.max_requests_per_second,
            retry_on_failure=True,
            max_retries=3
        )
        
        # Process in batches for checkpointing
        batch_size = self.config.checkpoint_interval
        total_tasks = len(tasks)
        
        for batch_start in range(0, total_tasks, batch_size):
            batch_end = min(batch_start + batch_size, total_tasks)
            batch_tasks = tasks[batch_start:batch_end]
            
            # Process batch
            batch_results = processor.process(
                batch_tasks,
                desc=f"Batch {batch_start//batch_size + 1}",
                show_progress=True
            )
            
            # Collect results and check for errors
            for task_id, result, error in batch_results:
                if error:
                    # Raise the error instead of silently catching it
                    raise RuntimeError(
                        f"Error during evaluation (entry {task_id}): {error}"
                    ) from error
                results.append(result)
            
            # Checkpoint
            self._save_checkpoint(evaluator.model_id, results)
    
    def _save_checkpoint(self, model_id: str, results: List[Dict[str, Any]]):
        """Save checkpoint for a model."""
        if not results:
            return
        
        safe_model_name = model_id.replace(':', '_').replace('.', '_')
        checkpoint_file = self.config.output_dir / f"checkpoint_{safe_model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Save as JSON (better for multi-line text)
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    def _save_model_results(self, model_id: str):
        """Save final results for a model."""
        if model_id not in self.results:
            return
        
        results = self.results[model_id]
        if not results:
            return
        
        safe_model_name = model_id.replace(':', '_').replace('.', '_')
        
        # Save as JSON (better for multi-line text and paragraphs)
        json_file = self.config.output_dir / f"evaluation_{safe_model_name}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved results: {json_file} ({len(results):,} evaluations)")

