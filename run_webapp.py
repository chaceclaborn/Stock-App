# run_webapp.py
import os
import sys
import argparse
import webbrowser
import subprocess
import time
import warnings

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

#%% Main
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Chace's Stock App")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the web server on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    args = parser.parse_args()
    
    # Verify Working directory: Should be Stock-App Folder. If not print a warning and continue
    if __file__.split('\\')[-2] != 'Stock-App':
        warnings.warn('Files not in a recognizeable file structure.')
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
    print("   📈 Real-time stock data with company names")
    print("   🤖 AI-powered technical analysis")
    print("   📊 Advanced trading indicators")
    print("   🎯 Smart entry/exit recommendations")
    print("   📱 Real-time news and events")
    print("   💚 Clean black, green & white theme")
    print("   🔮 Market sentiment analysis")
    print("   📉 Enhanced individual stock pages")
    print("   📊 Performance tracking & analytics")
    print("\n" + "="*60 + "\n")
    
    # Import only after checks pass
    try:
        # Add the current directory to Python path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # Import the create_app function
        from src.web.app import create_app
        
        # Create the app instance
        app = create_app('development' if args.debug else 'production')
        
        print(f"🌐 Starting server on port {args.port}...")
        print(f"📊 Open your browser and navigate to: http://localhost:{args.port}")
        print(f"\n📡 Debug endpoint available at: http://localhost:{args.port}/api/debug")
        print(f"🧪 Test stock endpoint: http://localhost:{args.port}/api/test-stock/AAPL")
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
        
        # Run the app
        app.run(debug=args.debug, port=args.port, threaded=True)
        
    except ImportError as e:
        print(f"\n❌ Error importing required modules: {e}")
        print("\nTry running:")
        print(f"  export PYTHONPATH={current_dir}:$PYTHONPATH")
        print(f"  python {__file__}")
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down Chace's Stock App...")
        print("Thanks for using the app!")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()