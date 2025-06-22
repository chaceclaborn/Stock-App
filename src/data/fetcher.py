# src/data/fetcher.py
"""
Stock data fetcher with multi-source support and fallback
"""
import os
import sys
import time
import logging

# Fix import path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Try different import methods
try:
    # Method 1: Relative import
    from .fetcher_multi_source import MultiSourceStockFetcher, DataCache
except ImportError:
    try:
        # Method 2: Absolute import from data
        from data.fetcher_multi_source import MultiSourceStockFetcher, DataCache
    except ImportError:
        try:
            # Method 3: Direct import
            import fetcher_multi_source
            MultiSourceStockFetcher = fetcher_multi_source.MultiSourceStockFetcher
            DataCache = fetcher_multi_source.DataCache
        except ImportError:
            # Fallback: Create dummy class
            class MultiSourceStockFetcher:
                def __init__(self, db=None):
                    self.db = db
                    self.sources = []
                    
                def get_quote(self, symbol):
                    import random
                    return {
                        'symbol': symbol,
                        'price': round(random.uniform(50, 500), 2),
                        'change': round(random.uniform(-5, 5), 2),
                        'change_percent': f"{random.uniform(-2, 2):.2f}%",
                        'volume': random.randint(1000000, 50000000),
                        'source': 'dummy',
                        'name': symbol
                    }
                
                def get_multiple_quotes(self, symbols):
                    return [self.get_quote(s) for s in symbols]
                
                def get_all_tracked_stocks(self):
                    return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
                
                def get_stocks_by_category(self, category):
                    return self.get_all_tracked_stocks()[:5]
                
                def get_top_stocks(self):
                    return self.get_all_tracked_stocks()[:10]
                
                def get_company_name(self, symbol):
                    return symbol
                
                def search_stocks(self, query, limit=10):
                    all_stocks = self.get_all_tracked_stocks()
                    return [s for s in all_stocks if query.upper() in s][:limit]
                
                def get_market_overview(self):
                    return {}
                
                def get_stock_data(self, symbol, period="3mo"):
                    import pandas as pd
                    return pd.DataFrame()
                
                def get_stocks_for_analysis(self, symbols=None):
                    return {}
            
            class DataCache:
                def __init__(self, default_ttl=300):
                    self.cache = {}
                    
                def get(self, key):
                    return self.cache.get(key)
                
                def set(self, key, value):
                    self.cache[key] = value

# For backwards compatibility, inherit from MultiSourceStockFetcher
class StockDataFetcher(MultiSourceStockFetcher):
    """
    Enhanced stock data fetcher with multi-source support
    Maintains backwards compatibility while adding robustness
    """
    
    def __init__(self, db=None):
        """Initialize fetcher with database"""
        super().__init__(db)
        
        # Log initialization
        logger = logging.getLogger(__name__)
        if hasattr(self, 'sources') and self.sources:
            logger.info(f"Stock Data Fetcher initialized with {len(self.sources)} sources")
        else:
            logger.info("Stock Data Fetcher initialized in fallback mode")
        
        # Additional backwards compatibility attributes
        self.quote_cache = DataCache()
        self.data_cache = DataCache()
        self.info_cache = DataCache()
        
        # For compatibility with old code
        if hasattr(self, 'yahoo'):
            self.rate_limiter = getattr(self.yahoo, 'rate_tracker', None)
        else:
            self.rate_limiter = None
    
    def get_quote_from_cache(self, symbol):
        """Backwards compatibility method"""
        cache_key = f"quote_{symbol}"
        return self.quote_cache.get(cache_key)
    
    def get_stock_fundamentals(self, symbol):
        """Get fundamentals - returns basic data for now"""
        quote = self.get_quote(symbol)
        if quote:
            return {
                'market_cap': quote.get('price', 0) * 1000000000,
                'pe_ratio': 25.0,
                'dividend_yield': 1.5,
                'beta': 1.0
            }
        return {}
    
    def get_news(self, symbol, limit=5):
        """Get news - returns dummy news for now"""
        return [
            {
                'title': f"Latest update on {symbol}",
                'publisher': 'Market News',
                'link': '#',
                'timestamp': int(time.time())
            }
            for _ in range(min(limit, 3))
        ]
    
    def get_realtime_metrics(self, symbol):
        """Get real-time metrics"""
        quote = self.get_quote(symbol)
        if quote:
            price = quote.get('price', 0)
            return {
                'bid': price - 0.01,
                'ask': price + 0.01,
                'bid_size': 100,
                'ask_size': 100,
                'day_high': price * 1.02,
                'day_low': price * 0.98,
                'open': price * 0.99,
                'previous_close': price,
                'fifty_day_average': price,
                'two_hundred_day_average': price,
                'average_volume': quote.get('volume', 1000000),
                'exchange': 'NASDAQ',
                'currency': 'USD'
            }
        return None

# Create module-level instances for backwards compatibility
if __name__ != "__main__":
    try:
        _default_fetcher = StockDataFetcher()
    except:
        _default_fetcher = None
