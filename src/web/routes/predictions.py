# src/web/routes/predictions.py
"""
AI prediction API endpoints
"""
from flask import Blueprint, request
import logging

from ..utils.api_response import APIResponse
from ..services.prediction_service import PredictionService
from ..services.sentiment_service import MarketSentimentAnalyzer

logger = logging.getLogger(__name__)

# Create blueprint
predictions_bp = Blueprint('predictions', __name__, url_prefix='/api')

# Initialize services
prediction_service = PredictionService()
sentiment_analyzer = MarketSentimentAnalyzer()

@predictions_bp.route('/predictions')
def get_predictions():
    """Get AI-powered trading predictions"""
    try:
        # Get parameters
        force_refresh = request.args.get('force', 'false').lower() == 'true'
        top_n = int(request.args.get('limit', '20'))
        min_score = float(request.args.get('min_score', '2'))
        period = request.args.get('period', '1d')
        
        # Get predictions
        predictions_data = prediction_service.get_predictions(
            force_refresh=force_refresh,
            top_n=top_n,
            min_score=min_score,
            period=period
        )
        
        if not predictions_data:
            return APIResponse.no_data("No trading opportunities found at this time")
        
        # Add sentiment to each opportunity
        for opp in predictions_data.get('opportunities', []):
            sentiment = sentiment_analyzer.get_market_sentiment(opp['ticker'])
            opp['sentiment'] = sentiment
        
        return APIResponse.success(
            data=predictions_data,
            message=f"Found {len(predictions_data.get('opportunities', []))} trading opportunities"
        )
        
    except Exception as e:
        logger.error(f"Error generating predictions: {str(e)}", exc_info=True)
        return APIResponse.error(f'Error generating predictions: {str(e)}', 500)

@predictions_bp.route('/predictions/strategies')
def get_available_strategies():
    """Get list of available prediction strategies"""
    try:
        strategies = prediction_service.get_available_strategies()
        
        return APIResponse.success(
            data={'strategies': strategies},
            message=f"{len(strategies)} strategies available"
        )
        
    except Exception as e:
        logger.error(f"Error getting strategies: {str(e)}")
        return APIResponse.error(f'Error getting strategies: {str(e)}', 500)

@predictions_bp.route('/predictions/strategy/<strategy_name>')
def get_strategy_predictions(strategy_name):
    """Get predictions for a specific strategy"""
    try:
        # Validate strategy exists
        if not prediction_service.strategy_exists(strategy_name):
            return APIResponse.error(f"Strategy '{strategy_name}' not found", 404)
        
        # Get parameters
        top_n = int(request.args.get('limit', '10'))
        symbols = request.args.get('symbols', '').split(',') if request.args.get('symbols') else None
        
        # Get strategy-specific predictions
        predictions = prediction_service.get_strategy_predictions(
            strategy_name=strategy_name,
            symbols=symbols,
            top_n=top_n
        )
        
        return APIResponse.success(
            data=predictions,
            message=f"Predictions from {strategy_name} strategy"
        )
        
    except Exception as e:
        logger.error(f"Error getting {strategy_name} predictions: {str(e)}")
        return APIResponse.error(f'Error getting strategy predictions: {str(e)}', 500)

@predictions_bp.route('/predictions/score/<symbol>')
def get_prediction_score(symbol):
    """Get detailed prediction score for a specific stock"""
    try:
        # Get scoring details
        score_data = prediction_service.get_detailed_score(symbol.upper())
        
        if not score_data:
            return APIResponse.error(f"Unable to score {symbol}", 404)
        
        return APIResponse.success(
            data=score_data,
            message=f"Detailed scoring for {symbol}"
        )
        
    except Exception as e:
        logger.error(f"Error scoring {symbol}: {str(e)}")
        return APIResponse.error(f'Error calculating score: {str(e)}', 500)

@predictions_bp.route('/predictions/feedback', methods=['POST'])
def submit_prediction_feedback():
    """Submit feedback on prediction accuracy for learning"""
    try:
        # Get feedback data
        data = request.get_json()
        
        if not data:
            return APIResponse.error("No feedback data provided", 400)
        
        # Validate required fields
        required_fields = ['symbol', 'predicted_score', 'actual_return']
        for field in required_fields:
            if field not in data:
                return APIResponse.error(f"Missing required field: {field}", 400)
        
        # Submit feedback
        success = prediction_service.submit_feedback(
            symbol=data['symbol'],
            predicted_score=data['predicted_score'],
            actual_return=data['actual_return'],
            patterns=data.get('patterns', []),
            signals=data.get('signals', {})
        )
        
        if success:
            return APIResponse.success(
                message="Feedback recorded successfully"
            )
        else:
            return APIResponse.error("Failed to record feedback", 500)
            
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        return APIResponse.error(f'Error submitting feedback: {str(e)}', 500)

@predictions_bp.route('/predictions/performance')
def get_prediction_performance():
    """Get performance metrics for predictions"""
    try:
        # Get time period
        period = request.args.get('period', '30d')
        strategy = request.args.get('strategy', 'all')
        
        # Get performance metrics
        performance = prediction_service.get_performance_metrics(
            period=period,
            strategy=strategy
        )
        
        return APIResponse.success(
            data=performance,
            message="Prediction performance metrics"
        )
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        return APIResponse.error(f'Error getting performance: {str(e)}', 500)