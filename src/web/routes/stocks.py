# src/web/routes/stocks.py
"""
Stock-related API endpoints with improved error handling
"""
from flask import Blueprint, request, current_app, jsonify
from datetime import datetime, timedelta
import logging

from ..utils.api_response import APIResponse

logger = logging.getLogger(__name__)

# Create blueprint
stocks_bp = Blueprint('stocks', __name__, url_prefix='/api')

def get_services():
    """Get services from app context with error handling"""
    try:
        services = current_app.extensions.get('services', {})
        stock_service = services.get('stock')
        sentiment_service = services.get('sentiment')
        
        if not stock_service:
            logger.warning("Stock service not found in app extensions")
            # Try to get from import
            try:
                from ..services.stock_service import StockService
                stock_service = StockService()
                if hasattr(stock_service, 'fetcher') and not stock_service.fetcher:
                    # Initialize if needed
                    from src.data.database import StockDatabase
                    from src.data.fetcher import StockDataFetcher
                    db = StockDatabase()
                    fetcher = StockDataFetcher(db)
                    stock_service.init_app(db, fetcher)
            except Exception as e:
                logger.error(f"Failed to create stock service: {e}")
        
        return stock_service, sentiment_service
    except Exception as e:
        logger.error(f"Error getting services: {e}")
        return None, None

@stocks_bp.route('/stocks')
def get_stocks():
    """Get current stock data with comprehensive error handling"""
    try:
        stock_service, sentiment_analyzer = get_services()
        
        if not stock_service:
            logger.error("Stock service not initialized")
            return APIResponse.error("Service temporarily unavailable", 503)
        
        # Parse parameters
        force_refresh = request.args.get('force', 'false').lower() == 'true'
        category = request.args.get('category', 'all')
        sort_by = request.args.get('sort', 'market_cap')
        
        logger.info(f"Getting stocks - category: {category}, force_refresh: {force_refresh}")
        
        # Get stocks data with error handling
        try:
            stocks_data = stock_service.get_stocks(
                category=category,
                force_refresh=force_refresh,
                sort_by=sort_by
            )
        except Exception as e:
            logger.error(f"Error from stock service: {str(e)}", exc_info=True)
            
            # Try to return cached data on error
            try:
                stocks_data = stock_service.get_cached_stocks()
                if stocks_data and stocks_data.get('stocks'):
                    stocks_data['from_cache'] = True
                    stocks_data['error_fallback'] = True
                    logger.info("Returning cached data after error")
                else:
                    # No cache available
                    return APIResponse.error(
                        "Unable to retrieve stock data. Please try again later.",
                        503,
                        error_code="STOCK_DATA_UNAVAILABLE"
                    )
            except Exception as cache_error:
                logger.error(f"Cache fallback failed: {cache_error}")
                return APIResponse.error(
                    "Service temporarily unavailable",
                    503,
                    error_code="SERVICE_ERROR"
                )
        
        # Validate response
        if not stocks_data or not isinstance(stocks_data.get('stocks'), list):
            logger.warning("Invalid stock data structure returned")
            return APIResponse.no_data("No stock data available")
        
        # Add sentiment data if available (limit to avoid delays)
        if sentiment_analyzer and len(stocks_data['stocks']) > 0:
            try:
                # Only process top 5 stocks for sentiment to avoid delays
                for stock in stocks_data['stocks'][:5]:
                    try:
                        sentiment = sentiment_analyzer.get_market_sentiment(
                            stock['symbol'], 
                            include_social=False
                        )
                        stock['sentiment'] = sentiment.get('overall', 0.5)
                        stock['sentiment_description'] = sentiment.get('description', 'Neutral')
                    except Exception as e:
                        logger.debug(f"Sentiment analysis failed for {stock['symbol']}: {e}")
                        stock['sentiment'] = 0.5
                        stock['sentiment_description'] = 'Unknown'
            except Exception as e:
                logger.error(f"Sentiment analysis error: {e}")
        
        # Build response
        response_data = {
            'stocks': stocks_data['stocks'],
            'categories': stock_service.get_categories() if stock_service else [],
            'last_updated': stocks_data.get('last_updated'),
            'next_update': stocks_data.get('next_update'),
            'from_cache': stocks_data.get('from_cache', False),
            'update_scheduled': stocks_data.get('update_scheduled', False),
            'market_status': stock_service.get_market_status() if stock_service else {}
        }
        
        # Add warning if using fallback data
        if stocks_data.get('error_fallback'):
            response_data['warning'] = "Using cached data due to temporary service issues"
        
        return APIResponse.success(data=response_data)
        
    except Exception as e:
        logger.error(f"Unexpected error in get_stocks: {str(e)}", exc_info=True)
        return APIResponse.error(
            "An unexpected error occurred",
            500,
            error_code="INTERNAL_ERROR"
        )

@stocks_bp.route('/stocks/cached')
def get_cached_stocks():
    """Get cached stocks immediately from memory/database"""
    try:
        stock_service, _ = get_services()
        
        if not stock_service:
            return APIResponse.error("Service not available", 503)
        
        cached_data = stock_service.get_cached_stocks()
        
        if not cached_data or not cached_data.get('stocks'):
            return APIResponse.no_data("No cached data available yet. Please wait for initial data load.")
        
        return APIResponse.cached(
            data=cached_data['stocks'],
            last_updated=cached_data.get('last_updated', datetime.now().isoformat()),
            categories=stock_service.get_categories()
        )
        
    except Exception as e:
        logger.error(f"Error getting cached stocks: {str(e)}", exc_info=True)
        return APIResponse.error("Error retrieving cached data", 500)

@stocks_bp.route('/stock/<symbol>')
def get_stock_details(symbol):
    """Get detailed information for a specific stock"""
    try:
        stock_service, sentiment_analyzer = get_services()
        
        if not stock_service:
            return APIResponse.error("Service not available", 503)
        
        # Validate symbol
        symbol = symbol.upper().strip()
        if not symbol or len(symbol) > 10:
            return APIResponse.error("Invalid stock symbol", 400)
        
        # Get stock details
        details = stock_service.get_stock_details(symbol)
        
        if not details:
            return APIResponse.error(f"Stock {symbol} not found", 404)
        
        # Add sentiment if available
        if sentiment_analyzer:
            try:
                sentiment = sentiment_analyzer.get_market_sentiment(symbol)
                details['sentiment'] = sentiment
            except Exception as e:
                logger.debug(f"Could not get sentiment for {symbol}: {e}")
        
        return APIResponse.success(data=details)
        
    except Exception as e:
        logger.error(f"Error getting details for {symbol}: {str(e)}", exc_info=True)
        return APIResponse.error(f"Error retrieving stock details", 500)

@stocks_bp.route('/stocks/search')
def search_stocks():
    """Search for stocks by symbol or company name"""
    try:
        stock_service, _ = get_services()
        
        if not stock_service:
            return APIResponse.error("Service not available", 503)
        
        # Get search query
        query = request.args.get('q', '').strip()
        if not query or len(query) < 1:
            return APIResponse.error("Search query too short", 400)
        
        limit = min(int(request.args.get('limit', 10)), 20)
        
        # Search
        results = stock_service.search_stocks(query, limit)
        
        # Format results
        formatted_results = []
        for symbol in results:
            formatted_results.append({
                'symbol': symbol,
                'name': stock_service.fetcher.get_company_name(symbol)
            })
        
        return APIResponse.success(
            data=formatted_results,
            query=query,
            count=len(formatted_results)
        )
        
    except Exception as e:
        logger.error(f"Error searching stocks: {str(e)}", exc_info=True)
        return APIResponse.error("Search failed", 500)

@stocks_bp.route('/stocks/refresh-status')
def get_refresh_status():
    """Get the current data refresh status"""
    try:
        stock_service, _ = get_services()
        
        if not stock_service:
            return APIResponse.error("Service not available", 503)
        
        now = datetime.now()
        
        # Get cache info
        cache_info = {
            'cache_timestamp': stock_service.cache_timestamp.isoformat() if stock_service.cache_timestamp else None,
            'cache_age_seconds': (now - stock_service.cache_timestamp).total_seconds() if stock_service.cache_timestamp else 0,
            'update_in_progress': stock_service.update_in_progress,
            'last_update_attempt': stock_service.last_update_attempt.isoformat() if stock_service.last_update_attempt else None,
            'failed_update_count': stock_service.failed_update_count,
            'next_update_seconds': stock_service._get_update_interval(),
            'market_status': stock_service.get_market_status(),
            'cached_stock_count': len(stock_service.stocks_cache)
        }
        
        return APIResponse.success(data=cache_info)
        
    except Exception as e:
        logger.error(f"Error getting refresh status: {str(e)}", exc_info=True)
        return APIResponse.error("Error getting status", 500)

# Health check endpoint
@stocks_bp.route('/stocks/health')
def stocks_health():
    """Health check for stocks service"""
    try:
        stock_service, _ = get_services()
        
        health = {
            'service': 'stocks',
            'status': 'healthy' if stock_service else 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'cache_size': len(stock_service.stocks_cache) if stock_service else 0,
            'has_fetcher': bool(stock_service and stock_service.fetcher),
            'has_db': bool(stock_service and stock_service.db)
        }
        
        if stock_service:
            return APIResponse.success(data=health)
        else:
            return APIResponse.error("Service unhealthy", 503, data=health)
            
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return APIResponse.error("Health check failed", 503)