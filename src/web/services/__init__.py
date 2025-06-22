# src/web/services/__init__.py
"""
Service layer initialization with improved error handling
"""
import os
import sys
import logging
import threading

logger = logging.getLogger(__name__)

# Service instances (singleton pattern)
_services = {}
_services_lock = threading.RLock()
_initialized = False

def init_services(app):
    """Initialize all services with app context"""
    global _initialized
    
    with _services_lock:
        if _initialized:
            logger.info("Services already initialized")
            return _services
        
        try:
            # Fix import paths
            parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            
            # Initialize database
            logger.info("Initializing database...")
            from src.data.database import StockDatabase
            db_dir = os.path.join(parent_dir, 'data')
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, 'stock_data.db')
            db = StockDatabase(db_path=db_path)
            logger.info(f"Database initialized at: {db_path}")
            
            # Initialize data fetcher
            logger.info("Initializing stock fetcher...")
            from src.data.fetcher import StockDataFetcher
            fetcher = StockDataFetcher(db=db)
            logger.info("Stock fetcher initialized with conservative rate limiting")
            
            # Initialize predictor
            logger.info("Initializing predictor...")
            try:
                from src.models.predictor import DayTradePredictor
                predictor = DayTradePredictor()
                logger.info("Predictor initialized")
            except Exception as e:
                logger.warning(f"Predictor initialization failed: {e}")
                predictor = None
            
            # Import service classes
            from .stock_service import StockService
            from .analysis_service import AnalysisService
            from .prediction_service import PredictionService
            from .market_service import MarketService
            from .sentiment_service import MarketSentimentAnalyzer
            
            # Initialize Stock Service
            logger.info("Initializing stock service...")
            stock_service = StockService()
            stock_service.init_app(db, fetcher)
            _services['stock'] = stock_service
            logger.info("Stock service initialized")
            
            # Initialize Analysis Service
            logger.info("Initializing analysis service...")
            analysis_service = AnalysisService()
            analysis_service.init_app(db, fetcher, predictor)
            _services['analysis'] = analysis_service
            logger.info("Analysis service initialized")
            
            # Initialize Prediction Service
            if predictor:
                logger.info("Initializing prediction service...")
                prediction_service = PredictionService()
                prediction_service.init_app(db, fetcher, predictor)
                _services['prediction'] = prediction_service
                logger.info("Prediction service initialized")
            else:
                logger.warning("Skipping prediction service due to predictor failure")
            
            # Initialize Market Service
            logger.info("Initializing market service...")
            market_service = MarketService()
            market_service.init_app(db, fetcher)
            _services['market'] = market_service
            logger.info("Market service initialized")
            
            # Initialize Sentiment Service
            logger.info("Initializing sentiment service...")
            try:
                sentiment_service = MarketSentimentAnalyzer()
                _services['sentiment'] = sentiment_service
                logger.info("Sentiment service initialized")
            except Exception as e:
                logger.warning(f"Sentiment service initialization failed: {e}")
            
            # Check for performance tracker service
            try:
                from .performance_tracker_service import PerformanceTrackerService
                logger.info("Initializing performance tracker service...")
                performance_service = PerformanceTrackerService()
                performance_service.init_app(db)
                _services['performance'] = performance_service
                logger.info("Performance tracker service initialized")
            except ImportError:
                logger.info("Performance tracker service not found, skipping")
            except Exception as e:
                logger.warning(f"Performance tracker initialization failed: {e}")
            
            # Store services in app extensions
            if app:
                app.extensions['services'] = _services
                logger.info(f"Registered {len(_services)} services with Flask app")
            
            _initialized = True
            
            # Log summary
            logger.info(f"Service initialization complete. Available services: {list(_services.keys())}")
            
            return _services
            
        except Exception as e:
            logger.error(f"Critical error during service initialization: {e}", exc_info=True)
            # Return partial services if any were initialized
            if app and _services:
                app.extensions['services'] = _services
            return _services

def get_service(service_name):
    """Get a specific service instance"""
    with _services_lock:
        return _services.get(service_name)

def get_all_services():
    """Get all service instances"""
    with _services_lock:
        return dict(_services)

def reset_services():
    """Reset all services (mainly for testing)"""
    global _initialized
    with _services_lock:
        _services.clear()
        _initialized = False
        logger.info("All services reset")

# Export service getter functions
__all__ = ['init_services', 'get_service', 'get_all_services', 'reset_services']