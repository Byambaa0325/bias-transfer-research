"""
EMGSD dataset processing and filtering.

Handles loading, filtering, and processing of EMGSD entries.
"""

from typing import List, Optional
from pathlib import Path
from data.emgsd_loader import load_emgsd, EMGSDEntry
from .utils import extract_target_question


class EMGSDProcessor:
    """
    Processes EMGSD dataset entries.
    
    Handles loading, filtering, and validation of EMGSD entries.
    """
    
    def __init__(
        self,
        transformed_path: Optional[Path] = None,
        default_path: Optional[Path] = None
    ):
        """
        Initialize EMGSD processor.
        
        Args:
            transformed_path: Path to transformed EMGSD CSV (with prompts)
            default_path: Path to default EMGSD CSV (fallback)
        """
        self.transformed_path = transformed_path or Path('data/emgsd_with_prompts.csv')
        self.default_path = default_path
        self.emgsd = None
        self._load_dataset()
    
    def _load_dataset(self):
        """Load EMGSD dataset from available paths."""
        if self.transformed_path.exists():
            self.emgsd = load_emgsd(dataset_path=str(self.transformed_path))
        elif self.default_path and self.default_path.exists():
            self.emgsd = load_emgsd(dataset_path=str(self.default_path))
        else:
            self.emgsd = load_emgsd()
    
    def get_entries(
        self,
        category: str = 'stereotype',
        stereotype_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[EMGSDEntry]:
        """
        Get filtered EMGSD entries.
        
        Args:
            category: Entry category ('stereotype', 'neutral', 'unrelated')
            stereotype_type: Filter by stereotype type (optional)
            limit: Maximum number of entries to return (optional)
            
        Returns:
            List of EMGSD entries
        """
        if category == 'stereotype':
            entries = self.emgsd.get_stereotypes(stereotype_type=stereotype_type)
        elif category == 'neutral':
            entries = self.emgsd.get_neutral(stereotype_type=stereotype_type)
        elif category == 'unrelated':
            entries = self.emgsd.get_unrelated(stereotype_type=stereotype_type)
        else:
            raise ValueError(f"Invalid category: {category}")
        
        if limit:
            entries = entries[:limit]
        
        return entries
    
    def validate_entries(self, entries: List[EMGSDEntry]) -> tuple[List[EMGSDEntry], List[str]]:
        """
        Validate entries and extract prompts.
        
        Args:
            entries: List of EMGSD entries to validate
            
        Returns:
            Tuple of (valid_entries, error_messages)
        """
        valid_entries = []
        errors = []
        
        for i, entry in enumerate(entries):
            # Extract target question
            target_question = extract_target_question(entry)
            
            if not target_question:
                errors.append(
                    f"Entry {i}: Could not extract target question from "
                    f"'{entry.text[:50]}...'"
                )
                continue
            
            # Store extracted prompt if not already present
            if not entry.final_prompt:
                entry.final_prompt = target_question
            
            valid_entries.append(entry)
        
        return valid_entries, errors
    
    def get_statistics(self) -> dict:
        """Get dataset statistics."""
        return self.emgsd.get_statistics() if self.emgsd else {}

