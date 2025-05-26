# src/web/app.py
"""
Main Flask application setup - Refactored for modularity
"""
import os
import sys
import logging
from flask import Flask, render_template, jsonify
from flask.json.provider import DefaultJSONProvider
import numpy as np
import pandas as pd

# Add parent directory to path so we can import our modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CustomJSONProvider(DefaultJSONProvider):
    """Custom JSON provider to handle numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        return super().default(obj)

def create_app(config_name='development'):
    """Application factory pattern"""
    
    # Create Flask app
    app = Flask(__name__,
                template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'))
    
    # Set custom JSON provider
    app.json = CustomJSONProvider(app)
    
    # Load configuration
    if config_name == 'production':
        app.config['DEBUG'] = False
    else:
        app.config['DEBUG'] = True
    
    # Additional configuration
    app.config['JSON_SORT_KEYS'] = False
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
    
    # Ensure template directories exist
    partials_dir = os.path.join(app.template_folder, 'partials')
    os.makedirs(partials_dir, exist_ok=True)
    
    # Initialize services (moved to separate initialization file)
    try:
        from .services import init_services
        services = init_services(app)
        logger.info(f"Initialized {len(services)} services")
    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}", exc_info=True)
        # Continue anyway to allow debugging
    
    # Register blueprints
    try:
        from .routes import all_blueprints
        for blueprint in all_blueprints:
            app.register_blueprint(blueprint)
            logger.info(f"Registered blueprint: {blueprint.name}")
    except Exception as e:
        logger.error(f"Failed to register blueprints: {str(e)}", exc_info=True)
    
    # Register main routes
    @app.route('/')
    def index():
        """Render the main page"""
        return render_template('index.html')
    
    # Debug endpoint
    @app.route('/api/debug')
    def debug_info():
        """Debug endpoint to check system status"""
        debug_data = {
            'status': 'running',
            'python_version': sys.version,
            'working_directory': os.getcwd(),
            'parent_directory': parent_dir,
            'sys_path': sys.path,
            'services_initialized': False,
            'services_count': 0,
            'services_available': [],
            'errors': []
        }
        
        try:
            # Check if services are initialized
            if hasattr(app, 'extensions') and 'services' in app.extensions:
                services = app.extensions['services']
                debug_data['services_initialized'] = True
                debug_data['services_count'] = len(services)
                debug_data['services_available'] = list(services.keys())
                
                # Test stock service
                if 'stock' in services:
                    stock_service = services['stock']
                    debug_data['stock_service'] = {
                        'initialized': True,
                        'has_db': stock_service._db is not None,
                        'has_fetcher': stock_service._fetcher is not None
                    }
                    
                    # Try to get categories
                    try:
                        categories = stock_service.get_categories()
                        debug_data['stock_categories'] = categories
                    except Exception as e:
                        debug_data['errors'].append(f"Error getting categories: {str(e)}")
                    
                    # Try to get cached stocks
                    try:
                        cached = stock_service.get_cached_stocks()
                        debug_data['cached_stocks_count'] = len(cached.get('stocks', []))
                    except Exception as e:
                        debug_data['errors'].append(f"Error getting cached stocks: {str(e)}")
                        
        except Exception as e:
            debug_data['errors'].append(f"Error checking services: {str(e)}")
        
        # Check for required modules
        required_modules = [
            'yfinance',
            'pandas',
            'numpy',
            'textblob',
            'flask'
        ]
        
        debug_data['modules'] = {}
        for module in required_modules:
            try:
                __import__(module)
                debug_data['modules'][module] = 'installed'
            except ImportError:
                debug_data['modules'][module] = 'missing'
        
        return jsonify(debug_data)
    
    # Test endpoint for stock data
    @app.route('/api/test-stock/<symbol>')
    def test_stock(symbol):
        """Test endpoint to check if we can fetch a single stock"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol.upper())
            info = ticker.info
            
            return jsonify({
                'success': True,
                'symbol': symbol.upper(),
                'name': info.get('longName', 'Unknown'),
                'price': info.get('regularMarketPrice', info.get('currentPrice', 0)),
                'info_keys': list(info.keys())[:20]  # First 20 keys
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e),
                'type': type(e).__name__
            })
    
    # Error handlers
    @app.errorhandler(404)
    def page_not_found(e):
        return {'error': 'Not found'}, 404
    
    @app.errorhandler(500)
    def server_error(e):
        return {'error': 'Server error', 'details': str(e)}, 500
    
    # Health check endpoint
    @app.route('/health')
    def health_check():
        return {
            'status': 'healthy',
            'version': '2.0.0',
            'service': 'stock-analyzer'
        }
    
    logger.info(f"Created Flask app with config: {config_name}")
    return app

def run_app(debug=True, port=5000):
    """Run the Flask application"""
    app = create_app('development' if debug else 'production')
    
    logger.info(f"Starting Enhanced Stock Market Analyzer on port {port}")
    logger.info("Features: Real-time data, Market sentiment analysis, AI predictions")
    logger.info("Using Yahoo Finance for real-time stock data.")
    
    app.run(debug=debug, port=port, threaded=True)

if __name__ == '__main__':
    run_app()