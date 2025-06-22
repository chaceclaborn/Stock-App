# run_webapp.py
"""
Enhanced startup script for the Stock Analyzer Web Application
"""
import os
import sys
import argparse
import webbrowser
import subprocess
import time
import warnings
import signal
import atexit
import threading

def check_requirements():
    """Check if required packages are installed"""
    required_packages = {
        'flask': 'flask',
        'yfinance': 'yfinance',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'requests': 'requests',
        'textblob': 'textblob',
        'scipy': 'scipy'
    }
    
    missing_packages = []
    
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("\n⚠️  Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstalling missing packages...")
        
        # Install missing packages
        for package in missing_packages:
            print(f"Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✓ {package} installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to install {package}: {e}")
                return False
        
        print("\n✅ All packages installed successfully!\n")
    
    return True

def setup_directories():
    """Create necessary directories and files"""
    # Create directories
    directories = [
        'src',
        'src/data',
        'src/web',
        'src/web/templates',
        'src/web/templates/partials',
        'src/web/services',
        'src/web/routes',
        'src/web/utils',
        'src/web/static',
        'src/web/static/js',
        'src/web/static/js/modules',
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
        'src/analysis/__init__.py',
        'src/web/services/__init__.py',
        'src/web/routes/__init__.py',
        'src/web/utils/__init__.py'
    ]
    
    for init_file in init_files:
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write('# Package initialization\n')
    
    return True

def check_file_structure():
    """Check if all required files are present"""
    required_files = {
        'src/data/fetcher.py': 'Data fetcher',
        'src/data/database.py': 'Database module',
        'src/models/predictor.py': 'Prediction model',
        'src/web/app.py': 'Flask application',
        'src/web/templates/index.html': 'Main template',
        'src/analysis/indicators.py': 'Technical indicators',
        'src/analysis/pattern_recognition.py': 'Pattern recognition'
    }
    
    missing_files = []
    for file_path, description in required_files.items():
        if not os.path.exists(file_path):
            missing_files.append((file_path, description))
    
    if missing_files:
        print("\n❌ Missing required files:")
        for file_path, description in missing_files:
            print(f"  - {file_path} ({description})")
        
        print("\n⚠️  Please ensure all required files are in place.")
        print("Some features may not work without these files.\n")
        return False
    
    return True

def test_yahoo_finance():
    """Test if Yahoo Finance is accessible"""
    print("Testing Yahoo Finance connection...", end=' ')
    try:
        import yfinance as yf
        ticker = yf.Ticker("AAPL")
        _ = ticker.fast_info
        print("✓ Connected")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        print("\n⚠️  Yahoo Finance connection failed.")
        print("The app will still start but may have limited functionality.")
        return False

def preload_cache(port):
    """Preload cache by making initial API calls"""
    def _preload():
        time.sleep(3)  # Wait for server to start
        try:
            import requests
            # Try to load cached stocks
            response = requests.get(f'http://localhost:{port}/api/stocks/cached', timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('data'):
                    print(f"📊 Pre-loaded {len(data['data'])} stocks from cache")
        except:
            pass  # Ignore errors during preload
    
    threading.Thread(target=_preload, daemon=True).start()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Run the Stock Analyzer Web Application')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--no-browser', action='store_true', help='Don\'t open browser automatically')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    
    args = parser.parse_args()
    
    # Suppress warnings
    warnings.filterwarnings('ignore')
    
    print("\n" + "="*60)
    print("🚀 STOCK MARKET ANALYZER - STARTUP")
    print("="*60)
    
    # Step 1: Check requirements
    print("\n1️⃣ Checking requirements...")
    if not check_requirements():
        print("\n❌ Failed to install required packages")
        return 1
    
    # Step 2: Setup directories
    print("\n2️⃣ Setting up directories...")
    if not setup_directories():
        print("\n❌ Failed to setup directories")
        return 1
    print("✓ Directories ready")
    
    # Step 3: Check file structure
    print("\n3️⃣ Checking file structure...")
    file_check = check_file_structure()
    
    # Step 4: Test Yahoo Finance
    print("\n4️⃣ Testing external connections...")
    yf_test = test_yahoo_finance()
    
    # Step 5: Import and run the app
    print("\n5️⃣ Starting the application...")
    
    # Add parent directory to path
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    try:
        from src.web.app import run_app
        
        # Open browser after a delay
        if not args.no_browser:
            def open_browser():
                time.sleep(2)
                webbrowser.open(f'http://localhost:{args.port}')
            
            threading.Thread(target=open_browser, daemon=True).start()
        
        # Preload cache
        preload_cache(args.port)
        
        # Print final instructions
        print("\n" + "="*60)
        print("📊 STOCK ANALYZER READY!")
        print("="*60)
        print(f"🌐 Open your browser and navigate to: http://localhost:{args.port}")
        print(f"📡 Debug endpoint: http://localhost:{args.port}/api/debug")
        print(f"🧪 Test stock: http://localhost:{args.port}/api/test-stock/AAPL")
        print("="*60)
        print("\n💡 First-time tips:")
        print("   - Initial data load may take 10-30 seconds")
        print("   - Check /api/debug if stocks don't appear")
        print("   - Data is cached to reduce API calls")
        print("   - Refresh button updates all data")
        print("\nPress Ctrl+C to stop the server\n")
        
        # Run the app
        run_app(debug=args.debug, port=args.port)
        
    except ImportError as e:
        print(f"\n❌ Failed to import application: {e}")
        print("Please ensure all files are properly set up.")
        return 1
    except KeyboardInterrupt:
        print("\n\n✋ Server stopped by user")
        return 0
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())