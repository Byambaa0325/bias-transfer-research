"""
Main dataset builder class.

Orchestrates the dataset generation process.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from tqdm import tqdm
import pandas as pd

from .config import DatasetConfig
from .bias_generator import BiasGenerator
from .emgsd_processor import EMGSDProcessor
from .utils import create_output_paths, generate_timestamp
from .parallel_processor import ParallelProcessor
from .logger import get_logger
from data.emgsd_loader import EMGSDEntry


class DatasetBuilder:
    """
    Builds multi-turn EMGSD dataset with bias injection questions.
    
    This is the main orchestrator class that coordinates:
    - Loading EMGSD entries
    - Generating bias injection questions
    - Building dataset rows
    - Saving results
    """
    
    def __init__(
        self,
        config: DatasetConfig,
        bias_generator: BiasGenerator,
        emgsd_processor: EMGSDProcessor
    ):
        """
        Initialize dataset builder.
        
        Args:
            config: Dataset configuration
            bias_generator: Bias question generator
            emgsd_processor: EMGSD data processor
        """
        self.config = config
        self.bias_generator = bias_generator
        self.emgsd_processor = emgsd_processor
        self.results: List[Dict[str, Any]] = []
        self.start_time: Optional[datetime] = None
        self.logger = get_logger()
        
        # Suppress verbose output from LLM service during batch processing
        # (unless verbose mode is explicitly enabled)
        if hasattr(bias_generator, 'llm_service'):
            bias_generator.llm_service._verbose_output = self.logger.verbose
    
    def build_entry_row(self, entry: EMGSDEntry) -> Dict[str, Any]:
        """
        Build a dataset row for a single EMGSD entry.
        
        Args:
            entry: EMGSD entry to process
            
        Returns:
            Dictionary representing a dataset row
        """
        from .utils import extract_target_question
        
        # Extract target question
        target_question = extract_target_question(entry)
        if not target_question:
            return self._create_error_row(entry, "Could not extract target question")
        
        # Base row with original EMGSD fields
        row = {
            # Original EMGSD fields
            'emgsd_text': entry.text,
            'emgsd_text_with_marker': entry.text_with_marker,
            'emgsd_stereotype_type': entry.stereotype_type,
            'emgsd_category': entry.category,
            'emgsd_data_source': entry.data_source,
            'emgsd_label': entry.label,
            'emgsd_target_group': entry.target_group,
            'emgsd_trait': entry.trait,
            'emgsd_target_word': entry.target_word,
            
            # Target question
            'target_question': target_question,
            'target_question_source': 'final_prompt' if entry.final_prompt else 'extracted_from_markers',
            
            # Metadata
            'bias_generator_model': self.config.bias_generator_model,
            'prompt_approach': 'persona-based' if self.config.use_persona_prompts else 'psycholinguistic',
            'generation_timestamp': datetime.now().isoformat(),
        }
        
        # Generate bias injection questions for each bias type
        for bias_type in self.config.bias_types:
            result = self.bias_generator.generate(
                target_question=target_question,
                bias_type=bias_type,
                stereotype_entry=entry
            )
            
            # Add columns for this bias type
            row[f'turn1_question_{bias_type}'] = result['question']
            row[f'bias_name_{bias_type}'] = result['bias_name']
            row[f'bias_description_{bias_type}'] = result['bias_description']
            row[f'bias_source_{bias_type}'] = result['bias_source']
            row[f'refusal_detected_{bias_type}'] = result['refusal_detected']
            row[f'refusal_reason_{bias_type}'] = result['refusal_reason']
            row[f'error_{bias_type}'] = result['error']
        
        return row
    
    def _create_error_row(self, entry: EMGSDEntry, error: str) -> Dict[str, Any]:
        """Create a row for an entry that failed processing."""
        return {
            'emgsd_text': entry.text,
            'emgsd_text_with_marker': entry.text_with_marker,
            'emgsd_stereotype_type': entry.stereotype_type,
            'emgsd_category': entry.category,
            'emgsd_data_source': entry.data_source,
            'emgsd_label': entry.label,
            'target_question': None,
            'error': error
        }
    
    def build_dataset(
        self,
        resume_from: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Build the complete dataset.
        
        Args:
            resume_from: Path to checkpoint CSV to resume from (optional)
            
        Returns:
            Dictionary with build statistics
        """
        self.start_time = datetime.now()
        
        # Get entries
        entries = self.emgsd_processor.get_entries(
            category=self.config.category_filter,
            stereotype_type=self.config.stereotype_type_filter,
            limit=self.config.sample_limit
        )
        
        # Handle resume
        if resume_from and resume_from.exists():
            existing_df = pd.read_csv(resume_from)
            processed_texts = set(existing_df['emgsd_text'].tolist())
            entries = [e for e in entries if e.text not in processed_texts]
            self.results = existing_df.to_dict('records')
            start_index = len(self.results)
        else:
            self.results = []
            start_index = 0
        
        # Validate entries
        valid_entries, errors = self.emgsd_processor.validate_entries(entries)
        if errors:
            self.logger.warning(f"{len(errors)} entries failed validation (out of {len(entries)} {self.config.category_filter} entries)")
            if self.logger.verbose:
                for error in errors[:5]:
                    print(f"  {error}")
                if len(errors) > 5:
                    print(f"  ... and {len(errors) - 5} more")
        
        print(f"\n{'='*70}")
        print(f"Processing {len(valid_entries)} valid entries (from {len(entries)} {self.config.category_filter} entries)")
        print(f"{'='*70}")
        
        # Calculate estimates
        total_api_calls = len(valid_entries) * len(self.config.bias_types)
        if self.config.use_parallel:
            # With parallel processing, estimate based on rate limit
            estimated_time_min = total_api_calls / (self.config.max_requests_per_second * 60)
            estimated_time_max = estimated_time_min * 1.5  # Add buffer
            speedup = min(self.config.max_workers, total_api_calls / 100)  # Rough speedup estimate
            print(f"Parallel: {self.config.max_workers} workers @ {self.config.max_requests_per_second} req/s")
            print(f"Est. time: {estimated_time_min:.1f}-{estimated_time_max:.1f} min (~{speedup:.1f}x speedup)")
        else:
            est_min = len(valid_entries) * len(self.config.bias_types) * 0.5
            est_max = len(valid_entries) * len(self.config.bias_types) * 1.0
            print(f"Sequential processing")
            print(f"Est. time: {est_min:.1f}-{est_max:.1f} minutes")
        print(f"Total API calls: {total_api_calls:,}")
        print(f"Checkpoint interval: Every {self.config.checkpoint_interval} entries\n")
        
        # Process entries
        if self.config.use_parallel:
            self._process_parallel(valid_entries, start_index)
        else:
            self._process_sequential(valid_entries, start_index)
        
        # Save final dataset
        return self._save_final_dataset()
    
    def _process_sequential(self, valid_entries: List[EMGSDEntry], start_index: int):
        """Process entries sequentially (original method)."""
        for i, entry in enumerate(tqdm(valid_entries, desc="Processing"), start=start_index):
            try:
                row = self.build_entry_row(entry)
                self.results.append(row)
            except Exception as e:
                error_row = self._create_error_row(entry, str(e))
                self.results.append(error_row)
            
            # Save checkpoint
            if (i + 1) % self.config.checkpoint_interval == 0:
                self._save_checkpoint()
                self._print_progress(i + 1, len(valid_entries))
    
    def _process_parallel(self, valid_entries: List[EMGSDEntry], start_index: int):
        """Process entries in parallel with rate limiting."""
        # Initialize parallel processor
        processor = ParallelProcessor(
            max_workers=self.config.max_workers,
            max_requests_per_second=self.config.max_requests_per_second,
            retry_on_failure=True,
            max_retries=3
        )
        
        # Prepare tasks: each entry needs to be processed
        tasks = []
        for entry in valid_entries:
            tasks.append((self.build_entry_row, (entry,), {}))
        
        # Process in batches to allow checkpointing
        batch_size = self.config.checkpoint_interval
        total_entries = len(valid_entries)
        
        for batch_start in range(0, total_entries, batch_size):
            batch_end = min(batch_start + batch_size, total_entries)
            batch_entries = valid_entries[batch_start:batch_end]
            batch_tasks = [(self.build_entry_row, (entry,), {}) for entry in batch_entries]
            
            # Process batch
            batch_num = batch_start//batch_size + 1
            total_batches = (total_entries + batch_size - 1) // batch_size
            batch_results = processor.process(
                batch_tasks,
                desc=f"Batch {batch_num}/{total_batches}",
                show_progress=True
            )
            
            # Collect results
            for task_id, result, error in batch_results:
                entry = batch_entries[task_id]
                if error:
                    row = self._create_error_row(entry, str(error))
                else:
                    row = result
                self.results.append(row)
            
            # Save checkpoint after each batch
            current_index = start_index + batch_end
            self._save_checkpoint()
            self._print_progress(current_index, total_entries, batch_num, total_batches)
        
        # Print final stats
        stats = processor.get_stats()
        print(f"\n{'='*70}")
        print(f"📊 Processing Complete")
        print(f"{'='*70}")
        print(f"Total tasks:    {stats['total']:,}")
        print(f"Successful:     {stats['success']:,}")
        print(f"Failed:         {stats['failed']:,}")
        if stats['retries'] > 0:
            print(f"Retries:        {stats['retries']:,}")
    
    def _save_checkpoint(self):
        """Save checkpoint to CSV."""
        checkpoint_path = self.config.output_dir / f'checkpoint_multiturn_emgsd_{generate_timestamp()}.csv'
        df = pd.DataFrame(self.results)
        df.to_csv(checkpoint_path, index=False)
    
    def _print_progress(self, current: int, total: int, batch_num: int = None, total_batches: int = None):
        """Print progress statistics."""
        successful = sum(1 for r in self.results if r.get('target_question'))
        pct = current/total*100
        
        # Calculate time elapsed
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds() / 60
            if current > 0:
                rate = current / elapsed
                remaining = (total - current) / rate if rate > 0 else 0
                eta_str = f" | ETA: {remaining:.1f} min"
            else:
                eta_str = ""
        else:
            elapsed = 0
            eta_str = ""
        
        # Compact progress line
        if batch_num and total_batches:
            print(f"\n✓ Batch {batch_num}/{total_batches} complete | "
                  f"Progress: {current:,}/{total:,} ({pct:.1f}%) | "
                  f"Success: {successful:,}/{current:,} | "
                  f"Elapsed: {elapsed:.1f} min{eta_str}")
        else:
            print(f"\n✓ Progress: {current:,}/{total:,} ({pct:.1f}%) | "
                  f"Successful: {successful:,}/{current:,} | "
                  f"Elapsed: {elapsed:.1f} min{eta_str}")
        
        # Count refusals (only show if significant)
        total_refusals = 0
        for bias_type in self.config.bias_types:
            refusals = sum(1 for r in self.results if r.get(f'refusal_detected_{bias_type}', False))
            total_refusals += refusals
        if total_refusals > 0 and total_refusals > current * 0.01:  # Show if >1% refusals
            print(f"  ⚠️  Refusals: {total_refusals:,} ({total_refusals/current*100:.1f}%)")
    
    def _save_final_dataset(self) -> Dict[str, Any]:
        """Save final dataset and return statistics."""
        paths = create_output_paths(self.config.output_dir)
        
        # Save CSV
        df = pd.DataFrame(self.results)
        df.to_csv(paths['csv'], index=False)
        print(f"  ✓ CSV: {paths['csv']}")
        
        # Save JSON
        with open(paths['json'], 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"  ✓ JSON: {paths['json']}")
        
        # Calculate statistics
        duration = (datetime.now() - self.start_time).total_seconds() / 60
        stats = self._calculate_statistics(duration)
        
        # Save summary
        summary = {
            'configuration': self.config.to_dict(),
            'statistics': stats,
            'output_files': {
                'csv': str(paths['csv']),
                'json': str(paths['json']),
            }
        }
        
        with open(paths['summary'], 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Summary: {paths['summary']}")
        
        return summary
    
    def _calculate_statistics(self, duration: float) -> Dict[str, Any]:
        """Calculate dataset generation statistics."""
        stats = {
            'total_entries': len(self.results),
            'entries_with_target_question': sum(1 for r in self.results if r.get('target_question')),
            'duration_minutes': round(duration, 2)
        }
        
        # Count refusals and errors per bias type
        for bias_type in self.config.bias_types:
            refusals = sum(1 for r in self.results if r.get(f'refusal_detected_{bias_type}', False))
            errors = sum(1 for r in self.results if r.get(f'error_{bias_type}') and not r.get(f'refusal_detected_{bias_type}', False))
            questions_generated = sum(1 for r in self.results if r.get(f'turn1_question_{bias_type}'))
            
            stats[f'{bias_type}_refusals'] = refusals
            stats[f'{bias_type}_errors'] = errors
            stats[f'{bias_type}_questions_generated'] = questions_generated
        
        return stats

