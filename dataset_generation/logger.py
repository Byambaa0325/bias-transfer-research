"""
Logging utilities for dataset generation.

Provides controlled output for long-running scripts.
"""

import sys
from typing import Optional
from datetime import datetime


class DatasetLogger:
    """
    Logger for dataset generation with verbosity control.
    
    Supports different log levels to reduce noise in long-running scripts.
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize logger.
        
        Args:
            verbose: If True, show detailed logs. If False, only show important messages.
        """
        self.verbose = verbose
        self._last_summary_time = None
        self._summary_interval = 60  # Show summary every 60 seconds
    
    def debug(self, message: str):
        """Log debug message (only if verbose)."""
        if self.verbose:
            self._print(f"🔍 {message}")
    
    def info(self, message: str):
        """Log info message (always shown)."""
        self._print(f"ℹ️  {message}")
    
    def success(self, message: str):
        """Log success message (always shown)."""
        self._print(f"✓ {message}")
    
    def warning(self, message: str):
        """Log warning message (always shown)."""
        self._print(f"⚠️  {message}")
    
    def error(self, message: str):
        """Log error message (always shown)."""
        self._print(f"❌ {message}", file=sys.stderr)
    
    def progress(self, message: str):
        """Log progress message (only if verbose)."""
        if self.verbose:
            self._print(f"→ {message}")
    
    def _print(self, message: str, file=sys.stdout):
        """Internal print with timestamp if verbose."""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}", file=file, flush=True)
        else:
            print(message, file=file, flush=True)
    
    def set_verbose(self, verbose: bool):
        """Change verbosity level."""
        self.verbose = verbose


# Global logger instance
_logger: Optional[DatasetLogger] = None


def get_logger() -> DatasetLogger:
    """Get global logger instance."""
    global _logger
    if _logger is None:
        _logger = DatasetLogger(verbose=False)
    return _logger


def set_logger_verbose(verbose: bool):
    """Set global logger verbosity."""
    logger = get_logger()
    logger.set_verbose(verbose)

