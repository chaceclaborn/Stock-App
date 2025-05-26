# test_setup.py
"""
Test script to diagnose stock loading issues
"""
import os
import sys

print("=== Stock Analyzer Diagnostic Test ===\n")

# Check Python version
print(f"Python version: {sys.version}")

# Check current directory
print(f"Current directory: {os.getcwd()}")

# Check if src is in path
print(f"\nPython path includes:")
for path in sys.path[:5]:
    print(f"  - {path}")

# Test imports
print("\n--- Testing imports ---")

modules_to_test = [
    ('yfinance', 'yfinance'),
    ('pandas', 'pandas'),
    ('numpy', 'numpy'),
    ('flask', 'flask'),
    ('textblob', 'textblob'),
    ('requests', 'requests')
]

for name, module in modules_to_test:
    try:
        __import__(module)
        print(f"✓ {name} - OK")
    except ImportError as e:
        print(f"✗ {name} - MISSING: {e}")

# Test Yahoo Finance
print("\n--- Testing Yahoo Finance ---")
try:
    import yfinance as yf
    ticker = yf.Ticker("AAPL")
    info = ticker.info
    print(f"✓ Yahoo Finance working")
    print(f"  AAPL price: ${info.get('regularMarketPrice', 'N/A')}")
except Exception as e:
    print(f"✗ Yahoo Finance error: {e}")

# Test file structure
print("\n--- Checking file structure ---")
required_files = [
    'src/web/app.py',
    'src/web/services/__init__.py',
    'src/web/routes/stocks.py',
    'src/data/database.py',
    'src/data/fetcher.py',
    'src/models/predictor.py',
    'src/analysis/indicators.py',
    'src/analysis/pattern_recognition.py'
]

for file in required_files:
    if os.path.exists(file):
        print(f"✓ {file}")
    else:
        print(f"✗ {file} - MISSING")

# Test database creation
print("\n--- Testing database ---")
try:
    # Add parent directory to path
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from src.data.database import StockDatabase
    db = StockDatabase()
    print("✓ Database initialized")
except Exception as e:
    print(f"✗ Database error: {e}")

# Test fetcher
print("\n--- Testing stock fetcher ---")
try:
    from src.data.fetcher import StockDataFetcher
    fetcher = StockDataFetcher()
    quote = fetcher.get_quote("AAPL")
    if quote:
        print(f"✓ Fetcher working - AAPL: ${quote['price']}")
    else:
        print("✗ Fetcher returned no data")
except Exception as e:
    print(f"✗ Fetcher error: {e}")

print("\n=== End of diagnostic test ===")
print("\nIf you see errors above, fix them before running the app.")
print("Run the app with: python run_webapp.py")