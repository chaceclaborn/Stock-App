# src/web/routes/stocks.py
"""
Stock-related API endpoints
"""
from flask import Blueprint, request, current_app
from datetime import datetime, timedelta
import logging

from ..utils.api_response import APIResponse

logger = logging.getLogger(__name__)

# Create blueprint
stocks_bp = Blueprint('stocks', __name__, url_prefix='/api')

def get_services():
    """Get services from app context"""
    services = current_app.extensions.get('services', {})
    return services.get('stock'), services.get('sentiment')

@stocks_bp.route('/stocks')
def get_stocks():
    """Get current stock data with status info"""
    try:
        stock_service, sentiment_analyzer = get_services()
        
        if not stock_service:
            logger.error("Stock service not initialized")
            return APIResponse.error("Service not available", 503)
        
        # Get parameters
        force_refresh = request.args.get('force', 'false').lower() == 'true'
        category = request.args.get('category', 'all')
        sort_by = request.args.get('sort', 'market_cap')
        
        logger.info(f"Getting stocks - category: {category}, force_refresh: {force_refresh}")
        
        # Get stocks data
        try:
            stocks_data = stock_service.get_stocks(
                category=category,
                force_refresh=force_refresh,
                sort_by=sort_by
            )
        except Exception as e:
            logger.error(f"Error from stock service: {str(e)}", exc_info=True)
            # Try to return cached data on error
            stocks_data = stock_service.get_cached_stocks()
            if stocks_data:
                stocks_data['from_cache'] = True
            else:
                return APIResponse.error(f"Unable to retrieve stock data: {str(e)}", 503)
        
        if not stocks_data or not stocks_data.get('stocks'):
            logger.warning("No stock data returned from service")
            return APIResponse.no_data("No stock data available")
        
        # Add sentiment data if available
        if sentiment_analyzer:
            for stock in stocks_data['stocks'][:10]:  # Limit sentiment analysis to avoid slowdown
                try:
                    sentiment = sentiment_analyzer.get_market_sentiment(
                        stock['symbol'], 
                        include_social=False
                    )
                    stock['sentiment'] = sentiment['overall']
                    stock['sentiment_description'] = sentiment['description']
                except Exception as e:
                    logger.debug(f"Could not get sentiment for {stock['symbol']}: {e}")
                    stock['sentiment'] = 0.5
                    stock['sentiment_description'] = 'Unknown'
        
        return APIResponse.success(
            data=stocks_data['stocks'],
            categories=stock_service.get_categories() if stock_service else [],
            last_updated=stocks_data.get('last_updated'),
            next_update=stocks_data.get('next_update'),
            from_cache=stocks_data.get('from_cache', False)
        )
        
    except Exception as e:
        logger.error(f"Error getting stock data: {str(e)}", exc_info=True)
        return APIResponse.error(f"Error retrieving stock data: {str(e)}", 500)

@stocks_bp.route('/stocks/cached')
def get_cached_stocks():
    """Get cached stocks immediately from database"""
    try:
        stock_service, _ = get_services()
        
        if not stock_service:
            return APIResponse.error("Service not available", 503)
        
        cached_data = stock_service.get_cached_stocks()
        
        if not cached_data or not cached_data.get('stocks'):
            return APIResponse.no_data("No cached data available")
        
        return APIResponse.cached(
            data=cached_data['stocks'],
            last_updated=cached_data.get('last_updated', datetime.now().isoformat())
        )
        
    except Exception as e:
        logger.error(f"Error getting cached stocks: {str(e)}", exc_info=True)
        return APIResponse.error(f"Error getting cached stocks: {str(e)}", 500)

@stocks_bp.route('/stocks/realtime')
def get_realtime_updates():
    """Get real-time stock updates for specific symbols"""
    try:
        stock_service, _ = get_services()
        
        if not stock_service:
            return APIResponse.error("Service not available", 503)
        
        # Get symbols from request
        symbols = request.args.get('symbols', '').split(',')
        if not symbols or symbols == ['']:
            return APIResponse.error('No symbols provided', 400)
        
        # Limit to 20 symbols for performance
        symbols = [s.strip().upper() for s in symbols[:20] if s.strip()]
        
        # Get real-time quotes
        quotes = stock_service.get_realtime_quotes(symbols)
        
        if not quotes:
            return APIResponse.no_data("Unable to fetch real-time data")
        
        return APIResponse.success(
            data={'quotes': quotes},
            message="Real-time data fetched successfully"
        )
        
    except Exception as e:
        logger.error(f"Error getting realtime updates: {str(e)}", exc_info=True)
        return APIResponse.error(f"Error getting realtime updates: {str(e)}", 500)

@stocks_bp.route('/stock/<symbol>')
def get_stock_detail(symbol):
    """Get detailed data for a specific stock"""
    try:
        stock_service, sentiment_analyzer = get_services()
        
        if not stock_service:
            return APIResponse.error("Service not available", 503)
        
        # Get period parameter
        period = request.args.get('period', '1m')
        
        # Get stock details
        stock_data = stock_service.get_stock_detail(symbol.upper(), period)
        
        if not stock_data:
            return APIResponse.error(f"Stock {symbol} not found", 404)
        
        # Get sentiment analysis if available
        if sentiment_analyzer:
            try:
                sentiment = sentiment_analyzer.get_market_sentiment(symbol.upper())
                stock_data['sentiment'] = sentiment
            except Exception as e:
                logger.debug(f"Could not get sentiment for {symbol}: {e}")
        
        return APIResponse.success(data=stock_data)
        
    except Exception as e:
        logger.error(f"Error getting stock detail for {symbol}: {str(e)}", exc_info=True)
        return APIResponse.error(f"Error retrieving data: {str(e)}", 500)

@stocks_bp.route('/search')
def search_stocks():
    """Search for stocks by symbol or name"""
    try:
        stock_service, sentiment_analyzer = get_services()
        
        if not stock_service:
            return APIResponse.error("Service not available", 503)
        
        query = request.args.get('q', '')
        if not query:
            return APIResponse.success(data={'results': []})
        
        results = stock_service.search_stocks(query)
        
        # Add sentiment to results if available
        if sentiment_analyzer and results:
            for stock in results[:5]:  # Limit to avoid slowdown
                try:
                    sentiment = sentiment_analyzer.get_market_sentiment(
                        stock['symbol'], 
                        include_social=False
                    )
                    stock['sentiment'] = sentiment['overall']
                    stock['sentiment_description'] = sentiment['description']
                except:
                    stock['sentiment'] = 0.5
                    stock['sentiment_description'] = 'Unknown'
        
        return APIResponse.success(data={'results': results})
        
    except Exception as e:
        logger.error(f"Error searching stocks: {str(e)}", exc_info=True)
        return APIResponse.error(f"Error searching stocks: {str(e)}", 500)