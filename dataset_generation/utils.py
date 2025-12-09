"""
Utility functions for dataset generation.
"""

from pathlib import Path
from typing import Optional
from datetime import datetime
from data.emgsd_loader import EMGSDEntry


def extract_target_question(entry: EMGSDEntry) -> Optional[str]:
    """
    Extract the target question from EMGSD entry.
    
    Priority:
    1. Use pre-transformed final_prompt if available
    2. Extract from text_with_marker using === markers
    
    Args:
        entry: EMGSD entry
        
    Returns:
        Target question string, or None if extraction fails
    """
    # Option 1: Use pre-transformed prompt
    if entry.final_prompt:
        return entry.final_prompt
    
    # Option 2: Extract from === markers
    if entry.text_with_marker:
        parts = entry.text_with_marker.split('===')
        if len(parts) >= 2:
            candidate = parts[0]
            if candidate.strip():
                return candidate
    
    return None


def generate_timestamp() -> str:
    """Generate a timestamp string for filenames."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def create_output_paths(output_dir: Path, prefix: str = 'multiturn_emgsd') -> dict:
    """
    Create standardized output file paths.
    
    Args:
        output_dir: Output directory
        prefix: Filename prefix
        
    Returns:
        Dictionary with 'csv', 'json', 'summary', and 'checkpoint' paths
    """
    timestamp = generate_timestamp()
    
    return {
        'csv': output_dir / f'{prefix}_dataset_{timestamp}.csv',
        'json': output_dir / f'{prefix}_dataset_{timestamp}.json',
        'summary': output_dir / f'{prefix}_dataset_{timestamp}_summary.json',
        'checkpoint': output_dir / f'checkpoint_{prefix}_{timestamp}.csv',
    }

