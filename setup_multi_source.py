# setup_multi_source.py
"""
Setup script for multi-source stock data fetching
Configures API keys and tests data sources
"""
import os
import sys
import time

def setup_environment():
    """Setup environment variables for API keys"""
    print("=" * 60)
    print("MULTI-SOURCE STOCK FETCHER SETUP")
    print("=" * 60)
    
    env_file = '.env'
    env_vars = {}
    
    # Check if .env exists
    if os.path.exists(env_file):
        print(f"Found existing {env_file}")
        with open(env_file, 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    env_vars[key] = value
    
    print("\nCurrent configuration:")
    print("-" * 40)
    
    # Alpha Vantage (Free tier: 5 calls/min, 500/day)
    print("\n1. Alpha Vantage (FREE)")
    print("   - Get free API key at: https://www.alphavantage.co/support/#api-key")
    print("   - Limits: 5 calls/minute, 500 calls/day")
    current = env_vars.get('ALPHA_VANTAGE_API_KEY', '')
    if current and current != 'demo':
        print(f"   - Current key: {current[:8]}...")
    else:
        print("   - Current key: Not set (using 'demo' key)")
    
    # Twelve Data (Free tier: 8 calls/min, 800/day)
    print("\n2. Twelve Data (FREE)")
    print("   - Get free API key at: https://twelvedata.com/apikey")
    print("   - Limits: 8 calls/minute, 800 calls/day")
    current = env_vars.get('TWELVE_DATA_API_KEY', '')
    if current:
        print(f"   - Current key: {current[:8]}...")
    else:
        print("   - Current key: Not set (disabled)")
    
    # Polygon.io (Free tier: 5 calls/min)
    print("\n3. Polygon.io (FREE)")
    print("   - Get free API key at: https://polygon.io/dashboard/signup")
    print("   - Limits: 5 calls/minute")
    current = env_vars.get('POLYGON_API_KEY', '')
    if current:
        print(f"   - Current key: {current[:8]}...")
    else:
        print("   - Current key: Not set (disabled)")
    
    print("\n" + "-" * 40)
    
    # Ask if user wants to update keys
    update = input("\nDo you want to update API keys? (y/N): ").lower().strip()
    
    if update == 'y':
        print("\nLeave blank to keep current value or skip")
        
        # Alpha Vantage
        new_key = input("\nAlpha Vantage API Key: ").strip()
        if new_key:
            env_vars['ALPHA_VANTAGE_API_KEY'] = new_key
        
        # Twelve Data
        new_key = input("Twelve Data API Key (optional): ").strip()
        if new_key:
            env_vars['TWELVE_DATA_API_KEY'] = new_key
        
        # Polygon
        new_key = input("Polygon.io API Key (optional): ").strip()
        if new_key:
            env_vars['POLYGON_API_KEY'] = new_key
        
        # Write .env file
        with open(env_file, 'w') as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")
        
        print(f"\n✅ Configuration saved to {env_file}")
    
    # Set environment variables for current session
    for key, value in env_vars.items():
        os.environ[key] = value
    
    return env_vars

def test_sources():
    """Test all configured data sources"""
    print("\n" + "=" * 60)
    print("TESTING DATA SOURCES")
    print("=" * 60)
    
    # Import fetcher
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    try:
        from src.data.fetcher_multi_source import MultiSourceStockFetcher
    except ImportError:
        print("❌ Could not import fetcher_multi_source.py")
        print("Make sure the file is in src/data/")
        return
    
    fetcher = MultiSourceStockFetcher()
    
    # Test each source
    print("\nTesting quote retrieval for AAPL:")
    print("-" * 40)
    
    test_symbol = 'AAPL'
    
    for source in fetcher.sources:
        print(f"\n{source.name}:")
        if not source.enabled:
            print("  ❌ Disabled")
            continue
        
        # Test the source
        start = time.time()
        quote = source.get_quote(test_symbol)
        elapsed = time.time() - start
        
        if quote:
            print(f"  ✅ Success! (took {elapsed:.2f}s)")
            print(f"     Price: ${quote['price']}")
            print(f"     Change: {quote.get('change_percent', 'N/A')}%")
        else:
            print(f"  ❌ Failed: {source.last_error}")
        
        # Show rate limit status
        usage = source.rate_tracker.get_usage()
        if 'minute' in usage:
            minute = usage['minute']
            print(f"     Rate: {minute['used']}/{minute['limit']} per minute")
    
    # Test combined fetcher
    print("\n" + "-" * 40)
    print("Testing combined fetcher:")
    
    quote = fetcher.get_quote('MSFT')
    if quote:
        print(f"✅ Got quote for MSFT from {quote['source']}")
        print(f"   Price: ${quote['price']}")
    else:
        print("❌ Failed to get quote from any source")

def main():
    """Main setup function"""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Setup environment
    setup_environment()
    
    # Test sources
    test = input("\nDo you want to test the data sources? (Y/n): ").lower().strip()
    if test != 'n':
        test_sources()
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print("\nYour app will now automatically use multiple data sources:")
    print("1. Yahoo Finance (primary)")
    print("2. Alpha Vantage (fallback)")
    print("3. Twelve Data (if configured)")
    print("4. Polygon.io (if configured)")
    print("5. Dummy data (last resort)")
    print("\nRun your app with: python run_webapp.py")

if __name__ == "__main__":
    main()