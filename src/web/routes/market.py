# src/web/routes/market.py
"""
Market overview API endpoints
"""
from flask import Blueprint
import logging

from ..utils.api_response import APIResponse
from ..services.market_service import MarketService
from ..services.sentiment_service import MarketSentimentAnalyzer

logger = logging.getLogger(__name__)

# Create blueprint
market_bp = Blueprint('market', __name__, url_prefix='/api')

# Initialize services
market_service = MarketService()
sentiment_analyzer = MarketSentimentAnalyzer()

@market_bp.route('/market-overview')
def get_market_overview():
    """Get market overview including indices and sentiment"""
    try:
        # Get market data
        market_data = market_service.get_market_overview()
        
        if not market_data:
            return APIResponse.error("Unable to fetch market data", 503)
        
        # Calculate fear & greed index
        fear_greed = sentiment_analyzer.calculate_fear_greed_index(market_data['indices'])
        market_data['indices']['fear_greed_index'] = fear_greed
        
        # Add market sentiment for major indices
        market_sentiment = {
            'spy': sentiment_analyzer.get_market_sentiment('SPY', include_social=False),
            'qqq': sentiment_analyzer.get_market_sentiment('QQQ', include_social=False),
            'vix': sentiment_analyzer.get_market_sentiment('^VIX', include_social=False)
        }
        
        result = {
            'indices': market_data['indices'],
            'market_sentiment': market_sentiment,
            'market_status': market_data.get('market_status', 'unknown')
        }
        
        return APIResponse.success(
            data=result,
            from_cache=market_data.get('from_cache', False)
        )
        
    except Exception as e:
        logger.error(f"Error getting market overview: {str(e)}")
        return APIResponse.error(f'Error getting market overview: {str(e)}', 500)

@market_bp.route('/market/sectors')
def get_sector_performance():
    """Get sector performance data"""
    try:
        sector_data = market_service.get_sector_performance()
        
        if not sector_data:
            return APIResponse.no_data("Sector data not available")
        
        return APIResponse.success(
            data=sector_data,
            message="Sector performance data"
        )
        
    except Exception as e:
        logger.error(f"Error getting sector performance: {str(e)}")
        return APIResponse.error(f'Error getting sector data: {str(e)}', 500)

@market_bp.route('/market/movers')
def get_market_movers():
    """Get top gainers, losers, and most active stocks"""
    try:
        movers = market_service.get_market_movers()
        
        if not movers:
            return APIResponse.no_data("Market movers data not available")
        
        # Add sentiment to movers
        for category in ['gainers', 'losers', 'most_active']:
            if category in movers:
                for stock in movers[category]:
                    sentiment = sentiment_analyzer.get_market_sentiment(
                        stock['symbol'], 
                        include_social=False
                    )
                    stock['sentiment'] = sentiment['overall']
        
        return APIResponse.success(
            data=movers,
            message="Market movers data"
        )
        
    except Exception as e:
        logger.error(f"Error getting market movers: {str(e)}")
        return APIResponse.error(f'Error getting market movers: {str(e)}', 500)

@market_bp.route('/market/calendar')
def get_market_calendar():
    """Get market calendar events (earnings, economic data, etc.)"""
    try:
        # Get calendar data
        calendar = market_service.get_market_calendar()
        
        return APIResponse.success(
            data=calendar,
            message="Market calendar events"
        )
        
    except Exception as e:
        logger.error(f"Error getting market calendar: {str(e)}")
        return APIResponse.error(f'Error getting market calendar: {str(e)}', 500)

@market_bp.route('/market/breadth')
def get_market_breadth():
    """Get market breadth indicators"""
    try:
        breadth_data = market_service.get_market_breadth()
        
        if not breadth_data:
            return APIResponse.no_data("Market breadth data not available")
        
        return APIResponse.success(
            data=breadth_data,
            message="Market breadth indicators"
        )
        
    except Exception as e:
        logger.error(f"Error getting market breadth: {str(e)}")
        return APIResponse.error(f'Error getting market breadth: {str(e)}', 500)