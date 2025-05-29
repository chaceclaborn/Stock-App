# src/web/routes/performance.py
"""
Performance tracking API endpoints
"""
from flask import Blueprint, request, current_app
import logging

from ..utils.api_response import APIResponse

logger = logging.getLogger(__name__)

# Create blueprint
performance_bp = Blueprint('performance', __name__, url_prefix='/api')

def get_tracker_service():
    """Get performance tracker service from app context"""
    services = current_app.extensions.get('services', {})
    return services.get('performance_tracker')

@performance_bp.route('/performance/summary')
def get_performance_summary():
    """Get performance summary for predictors"""
    try:
        tracker_service = get_tracker_service()
        
        if not tracker_service:
            return APIResponse.error("Performance tracker service not available", 503)
        
        # Get parameters
        days = int(request.args.get('days', '30'))
        
        # Get performance summary
        summary = tracker_service.get_performance_summary(days)
        
        return APIResponse.success(
            data={'predictors': summary},
            message=f"Performance summary for last {days} days"
        )
        
    except Exception as e:
        logger.error(f"Error getting performance summary: {str(e)}")
        return APIResponse.error(f'Error getting performance summary: {str(e)}', 500)

@performance_bp.route('/performance/active')
def get_active_predictions():
    """Get currently active tracked predictions"""
    try:
        tracker_service = get_tracker_service()
        
        if not tracker_service:
            return APIResponse.error("Performance tracker service not available", 503)
        
        # Get active predictions
        active_predictions = tracker_service.get_active_predictions()
        
        return APIResponse.success(
            data={'predictions': active_predictions},
            message=f"Found {len(active_predictions)} active predictions"
        )
        
    except Exception as e:
        logger.error(f"Error getting active predictions: {str(e)}")
        return APIResponse.error(f'Error getting active predictions: {str(e)}', 500)

@performance_bp.route('/performance/history')
def get_performance_history():
    """Get historical performance data"""
    try:
        tracker_service = get_tracker_service()
        
        if not tracker_service:
            return APIResponse.error("Performance tracker service not available", 503)
        
        # Get parameters
        symbol = request.args.get('symbol')
        predictor_type = request.args.get('type')
        days = int(request.args.get('days', '90'))
        
        # Get historical data
        history = tracker_service.get_historical_performance(
            symbol=symbol,
            predictor_type=predictor_type,
            days=days
        )
        
        return APIResponse.success(
            data={'history': history},
            message=f"Historical performance data"
        )
        
    except Exception as e:
        logger.error(f"Error getting performance history: {str(e)}")
        return APIResponse.error(f'Error getting performance history: {str(e)}', 500)

@performance_bp.route('/performance/start-tracking', methods=['POST'])
def start_tracking():
    """Manually start tracking a prediction"""
    try:
        tracker_service = get_tracker_service()
        
        if not tracker_service:
            return APIResponse.error("Performance tracker service not available", 503)
        
        # Get data from request
        data = request.get_json()
        
        if not data:
            return APIResponse.error("No prediction data provided", 400)
        
        required_fields = ['symbol', 'prediction_type', 'score', 'price']
        for field in required_fields:
            if field not in data:
                return APIResponse.error(f"Missing required field: {field}", 400)
        
        # Create prediction object
        prediction = {
            'ticker': data['symbol'],
            'score': data['score'],
            'price': data['price'],
            'period': '1d' if data['prediction_type'] == 'short_term' else '1m',
            'risk_metrics': data.get('risk_metrics', {}),
            'patterns': data.get('patterns', []),
            'signals': data.get('signals', {})
        }
        
        # Add to tracking
        tracker_service._add_prediction_to_tracking(prediction, data['prediction_type'])
        
        return APIResponse.success(
            message=f"Started tracking {data['prediction_type']} prediction for {data['symbol']}"
        )
        
    except Exception as e:
        logger.error(f"Error starting tracking: {str(e)}")
        return APIResponse.error(f'Error starting tracking: {str(e)}', 500)

@performance_bp.route('/performance/tracking-status')
def get_tracking_status():
    """Get current tracking system status"""
    try:
        tracker_service = get_tracker_service()
        
        if not tracker_service:
            return APIResponse.error("Performance tracker service not available", 503)
        
        status = {
            'is_tracking': tracker_service.is_tracking,
            'check_interval': tracker_service.check_interval,
            'active_predictions': len(tracker_service.get_active_predictions()),
            'performance_summary': tracker_service.get_performance_summary(7)  # Last 7 days
        }
        
        return APIResponse.success(
            data=status,
            message="Tracking system status"
        )
        
    except Exception as e:
        logger.error(f"Error getting tracking status: {str(e)}")
        return APIResponse.error(f'Error getting tracking status: {str(e)}', 500)