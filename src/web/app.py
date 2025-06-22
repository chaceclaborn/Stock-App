# src/web/app.py
"""
Flask application with minimal fixes for compatibility
"""
import os
import sys
import logging
from flask import Flask, render_template, jsonify
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_app(config_name='production'):
    """Create and configure Flask application"""
    
    # Fix import paths
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    
    # Create Flask app
    app = Flask(__name__,
                template_folder='templates',
                static_folder='static',
                static_url_path='/static')
    
    # Configure app
    app.config['SECRET_KEY'] = 'dev-secret-key-change-in-production'
    app.config['JSON_SORT_KEYS'] = False
    
    # Initialize extensions
    app.extensions = {}
    
    # Initialize services
    try:
        from .services import init_services
        services = init_services(app)
        logger.info(f"Initialized {len(services)} services")
    except Exception as e:
        logger.error(f"Service initialization error: {e}")
    
    # Register blueprints
    try:
        from .routes import all_blueprints
        for blueprint in all_blueprints:
            app.register_blueprint(blueprint)
            logger.info(f"Registered blueprint: {blueprint.name}")
    except Exception as e:
        logger.error(f"Blueprint registration error: {e}")
    
    # Root route
    @app.route('/')
    def index():
        """Render the main application page"""
        return render_template('index.html')
    
    # Debug endpoint
    @app.route('/api/debug')
    def debug_info():
        """Debug endpoint to check system status"""
        services = app.extensions.get('services', {})
        return jsonify({
            'status': 'operational',
            'timestamp': datetime.now().isoformat(),
            'services': list(services.keys()),
            'service_count': len(services)
        })
    
    # Test endpoint
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
                'price': info.get('regularMarketPrice', info.get('currentPrice', 0))
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
    logger.info(f"Static folder: {app.static_folder}")
    logger.info(f"Template folder: {app.template_folder}")
    
    return app

def run_app(debug=True, port=5000):
    """Run the Flask application"""
    app = create_app('development' if debug else 'production')
    
    # Suppress the werkzeug startup banner if not in debug mode
    if not debug:
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
    
    logger.info(f"Starting Enhanced Stock Market Analyzer on port {port}")
    logger.info("Features: Real-time data, Market sentiment analysis, AI predictions")
    logger.info("Using Yahoo Finance for real-time stock data.")
    
    # Run with reloader disabled to prevent duplicate output
    app.run(debug=debug, port=port, threaded=True, use_reloader=False)

if __name__ == '__main__':
    run_app()