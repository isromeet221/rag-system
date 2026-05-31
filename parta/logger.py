import time
import logging
import functools
from typing import Callable, Any

import os
from pathlib import Path

# Create .logs directory
log_dir = Path(os.path.dirname(os.path.abspath(__file__))) / ".logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "rag_timing.log"

logger = logging.getLogger("RAG_Timing")
logger.setLevel(logging.INFO)

if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - [%(name)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    
    fh = logging.FileHandler(log_file, encoding='utf-8', mode='a')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    import sys
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    # Ensure logs show up in terminal by propagating to root logger
    logger.propagate = True

import atexit

_private_stats = {}

def _log_private_stats():
    if not _private_stats:
        return
    logger.info("=== Private Functions Total Time ===")
    for func_name, stats in _private_stats.items():
        logger.info(f"Function {func_name} called {stats['count']} times, total time: {stats['total_time']:.4f}s")
    logger.info("====================================")

atexit.register(_log_private_stats)

def time_it(func: Callable) -> Callable:
    """Decorator to measure execution time of a synchronous function."""
    is_private = func.__name__.startswith('_')

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        
        if is_private:
            name = f"{func.__module__}.{func.__name__}"
            if name not in _private_stats:
                _private_stats[name] = {'count': 0, 'total_time': 0.0}
            _private_stats[name]['count'] += 1
            _private_stats[name]['total_time'] += elapsed
        else:
            logger.info(f"Function {func.__module__}.{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper

def async_time_it(func: Callable) -> Callable:
    """Decorator to measure execution time of an asynchronous function."""
    is_private = func.__name__.startswith('_')

    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        
        if is_private:
            name = f"{func.__module__}.{func.__name__}"
            if name not in _private_stats:
                _private_stats[name] = {'count': 0, 'total_time': 0.0}
            _private_stats[name]['count'] += 1
            _private_stats[name]['total_time'] += elapsed
        else:
            logger.info(f"Async Function {func.__module__}.{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper

class TimeItContext:
    """Context manager for timing specific code blocks inside functions."""
    def __init__(self, block_name: str):
        self.block_name = block_name
        self.start_time = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        logger.info(f"Block [{self.block_name}] took {end_time - self.start_time:.4f}s")
