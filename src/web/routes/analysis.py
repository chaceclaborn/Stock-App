# src/web/routes/analysis.py
"""
Technical analysis API endpoints
"""
from flask import Blueprint, request
import pandas as pd
import numpy as np
import logging

from ..utils.api_response import APIResponse
from ..services.analysis_service import AnalysisService
from ..services.sentiment_service import MarketSentimentAnalyzer

logger = logging.getLogger(__name__)

# Create blueprint
analysis_bp = Blueprint('analysis', __name__, url_prefix='/api')

# Initialize services
analysis_service = AnalysisService()
sentiment_analyzer = MarketSentimentAnalyzer()

@analysis_bp.route('/stock/<symbol>/analysis')
def get_stock_analysis(symbol):
    """Get detailed technical analysis for a stock"""
    try:
        # Get analysis data
        analysis_data = analysis_service.analyze_stock(symbol.upper())
        
        if not analysis_data:
            return APIResponse.error(f'Unable to analyze {symbol}', 404)
        
        # Get sentiment analysis
        sentiment = sentiment_analyzer.get_market_sentiment(symbol.upper())
        
        # Convert numpy types for JSON serialization
        analysis_data = APIResponse.convert_numpy_types(analysis_data)
        
        # Combine analysis and sentiment
        result = {
            **analysis_data,
            'sentiment': sentiment
        }
        
        return APIResponse.success(data=result)
        
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {str(e)}", exc_info=True)
        return APIResponse.error(f'Error analyzing stock: {str(e)}', 500)

@analysis_bp.route('/stock/<symbol>/indicators')
def get_stock_indicators(symbol):
    """Get specific technical indicators for a stock"""
    try:
        # Get requested indicators
        indicators = request.args.get('indicators', 'all').split(',')
        period = request.args.get('period', '1m')
        
        # Get indicator data
        indicator_data = analysis_service.get_indicators(
            symbol.upper(),
            indicators,
            period
        )
        
        if not indicator_data:
            return APIResponse.error(f'Unable to calculate indicators for {symbol}', 404)
        
        # Convert numpy types
        indicator_data = APIResponse.convert_numpy_types(indicator_data)
        
        return APIResponse.success(data=indicator_data)
        
    except Exception as e:
        logger.error(f"Error getting indicators for {symbol}: {str(e)}")
        return APIResponse.error(f'Error calculating indicators: {str(e)}', 500)

@analysis_bp.route('/stock/<symbol>/patterns')
def get_stock_patterns(symbol):
    """Get detected chart patterns for a stock"""
    try:
        # Get pattern detection parameters
        min_strength = float(request.args.get('min_strength', '0.5'))
        lookback_days = int(request.args.get('lookback', '60'))
        
        # Get pattern data
        patterns = analysis_service.detect_patterns(
            symbol.upper(),
            min_strength=min_strength,
            lookback_days=lookback_days
        )
        
        return APIResponse.success(
            data={'patterns': patterns},
            message=f"Found {len(patterns)} patterns for {symbol}"
        )
        
    except Exception as e:
        logger.error(f"Error detecting patterns for {symbol}: {str(e)}")
        return APIResponse.error(f'Error detecting patterns: {str(e)}', 500)

@analysis_bp.route('/stock/<symbol>/support-resistance')
def get_support_resistance(symbol):
    """Get support and resistance levels for a stock"""
    try:
        # Get parameters
        window = int(request.args.get('window', '20'))
        num_touches = int(request.args.get('touches', '2'))
        
        # Get support/resistance levels
        levels = analysis_service.get_support_resistance(
            symbol.upper(),
            window=window,
            num_touches=num_touches
        )
        
        return APIResponse.success(data=levels)
        
    except Exception as e:
        logger.error(f"Error getting S/R levels for {symbol}: {str(e)}")
        return APIResponse.error(f'Error calculating support/resistance: {str(e)}', 500)

@analysis_bp.route('/stock/<symbol>/sentiment')
def get_stock_sentiment(symbol):
    """Get detailed sentiment analysis for a stock"""
    try:
        sentiment = sentiment_analyzer.get_market_sentiment(symbol.upper())
        
        # Get additional context
        sentiment_context = analysis_service.get_sentiment_context(symbol.upper())
        
        result = {
            'symbol': symbol.upper(),
            'sentiment': sentiment,
            'context': sentiment_context
        }
        
        return APIResponse.success(data=result)
        
    except Exception as e:
        logger.error(f"Error getting sentiment for {symbol}: {str(e)}")
        return APIResponse.error(f'Error getting sentiment: {str(e)}', 500)

@analysis_bp.route('/stock/<symbol>/backtest')
def backtest_strategy(symbol):
    """Backtest a trading strategy on historical data"""
    try:
        # Get backtest parameters
        strategy = request.args.get('strategy', 'default')
        period = request.args.get('period', '1y')
        initial_capital = float(request.args.get('capital', '10000'))
        
        # Run backtest
        backtest_results = analysis_service.backtest_strategy(
            symbol.upper(),
            strategy=strategy,
            period=period,
            initial_capital=initial_capital
        )
        
        if not backtest_results:
            return APIResponse.error(f'Unable to backtest strategy for {symbol}', 404)
        
        return APIResponse.success(
            data=backtest_results,
            message=f"Backtest completed for {symbol} using {strategy} strategy"
        )
        
    except Exception as e:
        logger.error(f"Error backtesting {symbol}: {str(e)}")
        return APIResponse.error(f'Error running backtest: {str(e)}', 500)