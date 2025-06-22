"""
Data layer initialization
"""
import os
import sys

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Add parent directory to path
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import with fallback
try:
    from .fetcher import StockDataFetcher
    from .database import StockDatabase
except ImportError:
    try:
        from data.fetcher import StockDataFetcher
        from data.database import StockDatabase
    except ImportError:
        # Create dummy classes
        class StockDataFetcher:
            pass
        class StockDatabase:
            pass

__all__ = ['StockDatabase', 'StockDataFetcher']
