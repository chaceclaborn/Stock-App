# run_webapp.py
import os
import sys
import argparse
import webbrowser
import subprocess
import time
import warnings
import signal
import atexit

#%%

def check_requirements():
    """Check if required packages are installed"""
    required_packages = ['flask', 'yfinance', 'pandas', 'numpy', 'requests', 'textblob','scipy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("\n⚠️  Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstalling missing packages...")
        
        # Install missing packages
        for package in missing_packages:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        
        print("\n✅ All packages installed successfully!\n")

def setup_directories():
    """Create necessary directories"""

    directories = [
        'src',
        'src/data',
        'src/web',
        'src/web/templates',
        'src/web/templates/partials',
        'src/web/services',
        'src/web/routes',
        'src/web/utils',
        'src/models',
        'src/analysis',
        'data'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Create __init__.py files
    init_files = [
        'src/__init__.py',
        'src/data/__init__.py',
        'src/web/__init__.py',
        'src/models/__init__.py',
        'src/analysis/__init__.py'
    ]
    
    for init_file in init_files:
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write('# Package initialization\n')

    # Check file structure
    expected_files = [
        'src/data/fetcher.py', 
        'src/data/database.py',
        'src/models/predictor.py',
        'src/web/app.py', 
        'src/web/templates/index.html',
        'src/analysis/indicators.py',
        'src/analysis/pattern_recognition.py'
    ]
    
    missing_files = []
    for file in expected_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("\n❌ ERROR: Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease ensure all required files are in place.")
        print("\nIf you're missing analysis files, basic versions have been created.")
        return

def preload_cache(app):
    """Preload cache with initial data"""
    print("\n📊 Preloading market data cache...")
    print("⏳ This may take a minute on first run...")
    
    with app.app_context():
        try:
            from src.web.services import get_service
            
            # Get stock service
            stock_service = get_service('stock')
            if stock_service:
                # Load initial cache from database
                cached_data = stock_service.get_cached_stocks(limit=50)
                if cached_data['stocks']:
                    print(f"✅ Loaded {len(cached_data['stocks'])} stocks from database cache")
                else:
                    print("📥 No cached data found. Will fetch fresh data on first request.")
                    print("💡 Tip: The app will gradually build up its cache over time.")
            
        except Exception as e:
            print(f"⚠️  Could not preload cache: {e}")
            print("   The app will fetch data on first request.")

def cleanup_handler(signum=None, frame=None):
    """Cleanup handler for graceful shutdown"""
    print("\n\n👋 Shutting down Chace's Stock App...")
    print("💾 Saving cache data...")
    print("Thanks for using the app!")
    sys.exit(0)

#%% Main
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Chace's Stock App")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the web server on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    parser.add_argument("--no-cache", action="store_true", help="Start without preloading cache")
    parser.add_argument("--refresh-interval", type=int, default=60, 
                       help="Cache refresh interval in seconds (default: 60)")
    parser.add_argument("--max-stocks", type=int, default=50, 
                       help="Maximum number of stocks to track (default: 50)")
    args = parser.parse_args()
    
    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)
    atexit.register(cleanup_handler)
    
    # Verify Working directory: Should be Stock-App Folder. If not print a warning and continue
    if __file__.split('\\')[-2] != 'Stock-App' and __file__.split('/')[-2] != 'Stock-App':
        warnings.warn('Files not in a recognizable file structure.')
    else:
        os.chdir(os.path.split(__file__)[0])

    # Check and install requirements
    check_requirements()
    
    # Setup directories
    setup_directories()
    
    print("\n" + "="*60)
    print("🚀 CHACE'S STOCK APP 🚀".center(60))
    print("="*60)
    print("\n✨ Features:")
    print("   📈 Real-time stock data with smart caching")
    print("   🤖 AI-powered technical analysis")
    print("   📊 Advanced trading indicators")
    print("   🎯 Smart entry/exit recommendations")
    print("   📱 Real-time news and events")
    print("   💚 Clean black, green & white theme")
    print("   🔮 Market sentiment analysis")
    print("   📉 Enhanced individual stock pages")
    print("   📊 Performance tracking & analytics")
    print("   ⚡ Optimized for Yahoo Finance rate limits")
    print("\n" + "="*60 + "\n")
    
    # Import only after checks pass
    try:
        # Add the current directory to Python path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # Set environment variables for configuration
        os.environ['CACHE_REFRESH_INTERVAL'] = str(args.refresh_interval)
        os.environ['MAX_TRACKED_STOCKS'] = str(args.max_stocks)
        
        # Import the create_app function
        from src.web.app import create_app
        
        # Create the app instance
        app = create_app('development' if args.debug else 'production')
        
        # Preload cache unless disabled
        if not args.no_cache:
            preload_cache(app)
        
        print(f"\n🌐 Starting server on port {args.port}...")
        print(f"📊 Cache refresh interval: {args.refresh_interval} seconds")
        print(f"📈 Tracking up to {args.max_stocks} stocks")
        print(f"\n🌍 Open your browser and navigate to: http://localhost:{args.port}")
        print(f"\n📡 Debug endpoint available at: http://localhost:{args.port}/api/debug")
        print(f"🧪 Test stock endpoint: http://localhost:{args.port}/api/test-stock/AAPL")
        print("\n💡 Tips:")
        print("   - Data is cached to minimize API calls")
        print("   - Cache refreshes automatically based on market hours")
        print("   - Use the refresh button sparingly to avoid rate limits")
        print("\nPress Ctrl+C to stop the server\n")
        
        # Automatically open browser unless --no-browser flag is used
        if not args.no_browser:
            # Wait a moment for server to start
            import threading
            def open_browser():
                time.sleep(1.5)
                webbrowser.open(f"http://localhost:{args.port}")
            
            browser_thread = threading.Thread(target=open_browser)
            browser_thread.daemon = True
            browser_thread.start()
        
        # Run the app with optimized settings
        app.run(
            debug=args.debug, 
            port=args.port, 
            threaded=True,
            use_reloader=False  # Disable reloader to prevent duplicate processes
        )
        
    except ImportError as e:
        print(f"\n❌ Error importing required modules: {e}")
        print("\nTry running:")
        print(f"  export PYTHONPATH={current_dir}:$PYTHONPATH")
        print(f"  python {__file__}")
    except KeyboardInterrupt:
        cleanup_handler()
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()