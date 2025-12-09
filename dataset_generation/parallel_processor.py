"""
Parallel processing utilities for dataset generation.

Handles concurrent API calls with rate limiting and error handling.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore, Lock
from typing import Callable, List, Tuple, Any, Optional
from tqdm import tqdm


class RateLimiter:
    """
    Simple rate limiter using token bucket algorithm.
    
    Limits the number of requests per second to avoid hitting API rate limits.
    """
    
    def __init__(self, max_requests_per_second: float = 5.0):
        """
        Initialize rate limiter.
        
        Args:
            max_requests_per_second: Maximum requests per second (default: 5.0)
        """
        self.max_requests_per_second = max_requests_per_second
        self.min_interval = 1.0 / max_requests_per_second
        self.last_request_time = 0.0
        self.lock = Lock()
    
    def wait(self):
        """Wait if necessary to respect rate limit."""
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()


class ParallelProcessor:
    """
    Process tasks in parallel with rate limiting and progress tracking.
    """
    
    def __init__(
        self,
        max_workers: int = 10,
        max_requests_per_second: float = 5.0,
        retry_on_failure: bool = True,
        max_retries: int = 3
    ):
        """
        Initialize parallel processor.
        
        Args:
            max_workers: Maximum number of concurrent workers (default: 10)
            max_requests_per_second: Rate limit in requests per second (default: 5.0)
            retry_on_failure: Whether to retry failed tasks (default: True)
            max_retries: Maximum number of retries per task (default: 3)
        """
        self.max_workers = max_workers
        self.rate_limiter = RateLimiter(max_requests_per_second)
        self.retry_on_failure = retry_on_failure
        self.max_retries = max_retries
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'retries': 0
        }
        self.stats_lock = Lock()
    
    def _update_stats(self, success: bool, retried: bool = False):
        """Update statistics thread-safely."""
        with self.stats_lock:
            self.stats['total'] += 1
            if success:
                self.stats['success'] += 1
            else:
                self.stats['failed'] += 1
            if retried:
                self.stats['retries'] += 1
    
    def _execute_with_retry(
        self,
        task_func: Callable,
        task_id: Any,
        *args,
        **kwargs
    ) -> Tuple[Any, Any, Optional[Exception]]:
        """
        Execute a task with retry logic.
        
        Returns:
            Tuple of (task_id, result, error)
        """
        last_error = None
        
        for attempt in range(self.max_retries if self.retry_on_failure else 1):
            try:
                # Rate limit before each request
                self.rate_limiter.wait()
                
                # Execute task
                result = task_func(*args, **kwargs)
                self._update_stats(success=True, retried=(attempt > 0))
                return (task_id, result, None)
                
            except Exception as e:
                last_error = e
                # Check if it's a rate limit error (429)
                if hasattr(e, 'status_code') and e.status_code == 429:
                    # Exponential backoff for rate limits
                    wait_time = (2 ** attempt) * 0.5
                    time.sleep(wait_time)
                elif attempt < self.max_retries - 1:
                    # Small delay before retry
                    time.sleep(0.5 * (attempt + 1))
        
        # All retries failed
        self._update_stats(success=False, retried=(self.max_retries > 1))
        return (task_id, None, last_error)
    
    def process(
        self,
        tasks: List[Tuple[Callable, tuple, dict]],
        desc: str = "Processing",
        show_progress: bool = True
    ) -> List[Tuple[Any, Any, Optional[Exception]]]:
        """
        Process tasks in parallel.
        
        Args:
            tasks: List of (function, args_tuple, kwargs_dict) tuples
            desc: Progress bar description
            show_progress: Whether to show progress bar
            
        Returns:
            List of (task_id, result, error) tuples in completion order
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {}
            for i, (func, args, kwargs) in enumerate(tasks):
                future = executor.submit(self._execute_with_retry, func, i, *args, **kwargs)
                future_to_task[future] = i
            
            # Process completed tasks with progress bar
            if show_progress:
                with tqdm(total=len(tasks), desc=desc) as pbar:
                    for future in as_completed(future_to_task):
                        result = future.result()
                        results.append(result)
                        pbar.update(1)
            else:
                for future in as_completed(future_to_task):
                    result = future.result()
                    results.append(result)
        
        # Sort results by original task order
        results.sort(key=lambda x: x[0])
        return [(task_id, result, error) for task_id, result, error in results]
    
    def get_stats(self) -> dict:
        """Get processing statistics."""
        with self.stats_lock:
            return self.stats.copy()

