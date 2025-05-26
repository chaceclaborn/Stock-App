# src/web/utils/api_response.py
"""
Standardized API response utilities
"""
from datetime import datetime
from flask import jsonify
import numpy as np
import pandas as pd

class APIResponse:
    """Standardized API response format"""
    
    @staticmethod
    def success(data=None, message=None, **kwargs):
        """Return a success response"""
        response = {
            'status': 'success',
            'data': APIResponse.convert_numpy_types(data) if data is not None else None,
            'timestamp': datetime.now().isoformat()
        }
        
        if message:
            response['message'] = message
            
        # Add any additional fields
        for key, value in kwargs.items():
            response[key] = APIResponse.convert_numpy_types(value)
        
        return jsonify(response)
    
    @staticmethod
    def error(message, code=400, details=None):
        """Return an error response"""
        response = {
            'status': 'error',
            'error': message,
            'code': code,
            'timestamp': datetime.now().isoformat()
        }
        
        if details:
            response['details'] = details
            
        return jsonify(response), code
    
    @staticmethod
    def no_data(message="No data available"):
        """Return a no data response"""
        return jsonify({
            'status': 'no_data',
            'message': message,
            'data': [],
            'timestamp': datetime.now().isoformat()
        })
    
    @staticmethod
    def cached(data, last_updated, message="Using cached data"):
        """Return a response indicating cached data"""
        return jsonify({
            'status': 'cached',
            'data': APIResponse.convert_numpy_types(data) if data is not None else None,
            'message': message,
            'last_updated': last_updated,
            'from_cache': True,
            'timestamp': datetime.now().isoformat()
        })
    
    @staticmethod
    def convert_numpy_types(obj):
        """Convert numpy types to Python native types for JSON serialization"""
        # Handle None
        if obj is None:
            return None
            
        # Handle numpy integers
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
            
        # Handle numpy floats
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
            
        # Handle numpy arrays
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
            
        # Handle pandas Series
        elif isinstance(obj, pd.Series):
            return obj.tolist()
            
        # Handle single value NaN check
        elif isinstance(obj, (int, float, str, bool)):
            try:
                if pd.isna(obj):
                    return None
            except:
                pass
            return obj
            
        # Handle dictionaries recursively
        elif isinstance(obj, dict):
            return {k: APIResponse.convert_numpy_types(v) for k, v in obj.items()}
            
        # Handle lists recursively
        elif isinstance(obj, list):
            return [APIResponse.convert_numpy_types(i) for i in obj]
            
        # Handle tuples
        elif isinstance(obj, tuple):
            return tuple(APIResponse.convert_numpy_types(i) for i in obj)
            
        # For any other type, try to convert or return as is
        else:
            # Check if it's array-like but not string
            if hasattr(obj, '__len__') and not isinstance(obj, str):
                try:
                    return [APIResponse.convert_numpy_types(i) for i in obj]
                except:
                    return str(obj)
            return obj