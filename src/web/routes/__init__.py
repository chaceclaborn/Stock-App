# src/web/routes/__init__.py
"""
Route blueprints for the Stock Analyzer application
"""

from .stocks import stocks_bp
from .analysis import analysis_bp
from .predictions import predictions_bp
from .market import market_bp
from .portfolio import portfolio_bp

# List all blueprints for easy registration
all_blueprints = [
    stocks_bp,
    analysis_bp,
    predictions_bp,
    market_bp,
    portfolio_bp
]

__all__ = ['all_blueprints']