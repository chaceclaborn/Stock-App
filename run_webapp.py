# run_webapp.py
import os
import sys
import argparse
import webbrowser
import subprocess
import time

def check_requirements():
    """Check if required packages are installed"""
    required_packages = ['flask', 'yfinance', 'pandas', 'numpy', 'requests', 'textblob']
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

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Chace's Stock App")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the web server on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    args = parser.parse_args()
    
    # Check and install requirements
    check_requirements()
    
    # Check file structure
    expected_folders = ['src', 'src/data', 'src/web', 'src/web/templates', 'src/models']
    missing_folders = []
    
    for folder in expected_folders:
        if not os.path.exists(folder):
            missing_folders.append(folder)
    
    if missing_folders:
        print("\n❌ ERROR: Missing required folders:")
        for folder in missing_folders:
            print(f"  - {folder}")
        print("\nCreating missing folders...")
        for folder in missing_folders:
            os.makedirs(folder, exist_ok=True)
        print("✅ Folders created successfully!")
        
    # Check for required files
    required_files = [
        'src/data/fetcher.py', 
        'src/data/database.py',
        'src/models/predictor.py',
        'src/web/app.py', 
        'src/web/templates/index.html'
    ]
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("\n❌ ERROR: Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease ensure all required files are in place.")
        return
    
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
    print("\n" + "="*60 + "\n")
    
    # Import only after checks pass
    try:
        # Add the current directory to Python path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # Import the Flask app
        from src.web.app import app
        
        print(f"🌐 Starting server on port {args.port}...")
        print(f"📊 Open your browser and navigate to: http://localhost:{args.port}")
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
        print("Please check that all files are properly structured.")
        print("\nExpected structure:")
        print("  run_webapp.py (this file)")
        print("  src/")
        print("    ├── web/")
        print("    │   ├── app.py")
        print("    │   └── templates/")
        print("    ├── data/")
        print("    │   ├── fetcher.py")
        print("    │   └── database.py")
        print("    └── models/")
        print("        └── predictor.py")
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down Chace's Stock App...")
        print("Thanks for using the app!")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()