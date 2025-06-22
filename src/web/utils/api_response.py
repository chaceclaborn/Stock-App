# src/web/utils/api_response.py
"""
API response utilities for consistent response formatting
"""
from flask import jsonify
from datetime import datetime
import numpy as np

class APIResponse:
    """Standardized API response builder"""
    
    @staticmethod
    def success(data=None, message=None, **kwargs):
        """Create a success response"""
        response = {
            'success': True,
            'timestamp': datetime.now().isoformat()
        }
        
        if data is not None:
            response['data'] = data
        
        if message:
            response['message'] = message
        
        # Add any additional fields
        response.update(kwargs)
        
        return jsonify(response), 200
    
    @staticmethod
    def error(message, status_code=400, error_code=None, **kwargs):
        """Create an error response"""
        response = {
            'success': False,
            'error': message,
            'timestamp': datetime.now().isoformat()
        }
        
        if error_code:
            response['error_code'] = error_code
        
        # Add any additional fields
        response.update(kwargs)
        
        return jsonify(response), status_code
    
    @staticmethod
    def no_data(message="No data available", **kwargs):
        """Create a no data response"""
        response = {
            'success': True,
            'data': [],
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        
        response.update(kwargs)
        
        return jsonify(response), 200
    
    @staticmethod
    def cached(data, last_updated=None, **kwargs):
        """Create a cached data response"""
        response = {
            'success': True,
            'data': data,
            'from_cache': True,
            'timestamp': datetime.now().isoformat()
        }
        
        if last_updated:
            response['cache_timestamp'] = last_updated
        
        response.update(kwargs)
        
        return jsonify(response), 200
    
    @staticmethod
    def paginated(data, page, per_page, total, **kwargs):
        """Create a paginated response"""
        response = {
            'success': True,
            'data': data,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total,
                'pages': (total + per_page - 1) // per_page if per_page > 0 else 0
            },
            'timestamp': datetime.now().isoformat()
        }
        
        response.update(kwargs)
        
        return jsonify(response), 200
    
    @staticmethod
    def convert_numpy_types(obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: APIResponse.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [APIResponse.convert_numpy_types(item) for item in obj]
        return obj