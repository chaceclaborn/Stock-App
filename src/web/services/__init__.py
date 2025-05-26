# src/web/services/__init__.py
"""
Service layer initialization
"""
import os
import sys
import logging

logger = logging.getLogger(__name__)

# Service instances (singleton pattern)
_services = {}

def init_services(app):
    """Initialize all services with app context"""
    
    # Fix import paths
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    
    try:
        # Database setup
        from src.data.database import StockDatabase
        db_dir = os.path.join(parent_dir, 'data')
        os.makedirs(db_dir, exist_ok=True)
        db_path = os.path.join(db_dir, 'stock_data.db')
        db = StockDatabase(db_path=db_path)
        logger.info(f"Database initialized at: {db_path}")
        
        # Data fetcher setup
        from src.data.fetcher import StockDataFetcher
        fetcher = StockDataFetcher(db=db)
        logger.info("Stock fetcher initialized")
        
        # Predictor setup
        from src.models.predictor import DayTradePredictor
        predictor = DayTradePredictor()
        logger.info("Predictor initialized")
        
        # Initialize services
        from .stock_service import StockService
        from .analysis_service import AnalysisService
        from .prediction_service import PredictionService
        from .market_service import MarketService
        from .sentiment_service import MarketSentimentAnalyzer
        
        # Stock service
        stock_service = StockService()
        stock_service.init_app(db, fetcher)
        _services['stock'] = stock_service
        
        # Analysis service
        analysis_service = AnalysisService()
        analysis_service.init_app(db, fetcher, predictor)
        _services['analysis'] = analysis_service
        
        # Prediction service
        prediction_service = PredictionService()
        prediction_service.init_app(db, fetcher, predictor)
        _services['prediction'] = prediction_service
        
        # Market service
        market_service = MarketService()
        market_service.init_app(db, fetcher)
        _services['market'] = market_service
        
        # Sentiment analyzer
        sentiment_analyzer = MarketSentimentAnalyzer()
        _services['sentiment'] = sentiment_analyzer
        
        # Store in app context
        app.extensions['services'] = _services
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing services: {str(e)}", exc_info=True)
        raise
    
    return _services

def get_service(name):
    """Get a service instance by name"""
    return _services.get(name)

__all__ = ['init_services', 'get_service']