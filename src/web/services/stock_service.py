# src/web/services/stock_service.py
"""
Stock data service layer with minimal fixes
"""
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class StockService:
    """Service for handling stock data operations"""
    
    def __init__(self):
        self.last_update_time = None
        self.update_interval = 60  # seconds
        self._db = None
        self._fetcher = None
        self.stocks_cache = {}
        self.cache_timestamp = None
    
    def init_app(self, db, fetcher):
        """Initialize with database and fetcher instances"""
        self._db = db
        self._fetcher = fetcher
    
    @property
    def db(self):
        """Lazy load database"""
        if self._db is None:
            from data.database import StockDatabase
            self._db = StockDatabase()
        return self._db
    
    @property
    def fetcher(self):
        """Lazy load fetcher"""
        if self._fetcher is None:
            from data.fetcher import StockDataFetcher
            self._fetcher = StockDataFetcher(self.db)
        return self._fetcher
    
    def get_categories(self):
        """Get available stock categories"""
        return [
            {'id': 'all', 'name': 'All Stocks', 'icon': '📊'},
            {'id': 'trending', 'name': 'Trending', 'icon': '🔥'},
            {'id': 'tech', 'name': 'Technology', 'icon': '💻'},
            {'id': 'finance', 'name': 'Finance', 'icon': '💰'},
            {'id': 'healthcare', 'name': 'Healthcare', 'icon': '🏥'},
            {'id': 'consumer', 'name': 'Consumer', 'icon': '🛒'},
            {'id': 'energy', 'name': 'Energy', 'icon': '⚡'},
            {'id': 'industrial', 'name': 'Industrial', 'icon': '🏭'}
        ]
    
    def get_stocks(self, category='all', force_refresh=False, sort_by='market_cap'):
        """Get stocks with caching logic"""
        now = datetime.now()
        
        # Check if we need to update
        should_update = (
            force_refresh or 
            not self.stocks_cache or
            not self.cache_timestamp or
            (now - self.cache_timestamp).total_seconds() > self.update_interval
        )
        
        if should_update:
            # Get symbols based on category
            if category == 'all':
                symbols = self.fetcher.get_all_tracked_stocks()[:50]
            elif category == 'trending':
                symbols = self.fetcher.get_top_stocks()
            else:
                symbols = self.fetcher.get_stocks_by_category(category)
            
            # Try to get fresh quotes
            try:
                quotes = self.fetcher.get_multiple_quotes(symbols)
                
                if quotes:
                    # Update cache
                    for quote in quotes:
                        self.stocks_cache[quote['symbol']] = quote
                    self.cache_timestamp = now
                    
                    logger.info(f"Updated {len(quotes)} quotes")
            except Exception as e:
                logger.error(f"Error fetching quotes: {e}")
                # Continue with cached data
        
        # Return cached data
        cached_quotes = list(self.stocks_cache.values())
        
        # Filter by category if needed
        if category != 'all' and category != 'trending':
            category_symbols = self.fetcher.get_stocks_by_category(category)
            cached_quotes = [q for q in cached_quotes if q['symbol'] in category_symbols]
        
        # Sort
        cached_quotes = self._sort_quotes(cached_quotes, sort_by)
        
        return {
            'stocks': self._format_quotes(cached_quotes),
            'last_updated': self.cache_timestamp.isoformat() if self.cache_timestamp else now.isoformat(),
            'next_update': (now + timedelta(seconds=self.update_interval)).isoformat(),
            'from_cache': not should_update
        }
    
    def get_cached_stocks(self):
        """Get stocks from cache immediately"""
        if not self.stocks_cache:
            # Try loading from database
            try:
                symbols = self.fetcher.get_all_tracked_stocks()[:50]
                db_quotes = self.db.get_latest_quotes(symbols)
                
                if db_quotes:
                    for quote in db_quotes:
                        self.stocks_cache[quote['symbol']] = quote
                    self.cache_timestamp = datetime.now()
            except Exception as e:
                logger.error(f"Error loading from database: {e}")
        
        return {
            'stocks': self._format_quotes(list(self.stocks_cache.values())),
            'last_updated': self.cache_timestamp.isoformat() if self.cache_timestamp else None,
            'from_cache': True
        }
    
    def _format_quotes(self, quotes):
        """Format quotes for response"""
        formatted = []
        for quote in quotes:
            if isinstance(quote, dict):
                formatted.append({
                    'symbol': quote.get('symbol', ''),
                    'name': quote.get('name', ''),
                    'price': float(quote.get('price', 0)),
                    'change': float(quote.get('change', 0)),
                    'change_percent': quote.get('change_percent', '0.00%'),
                    'volume': int(quote.get('volume', 0)),
                    'last_updated': quote.get('last_updated', '')
                })
        return formatted
    
    def _sort_quotes(self, quotes, sort_by='market_cap'):
        """Sort quotes by specified criteria"""
        if not quotes:
            return quotes
        
        # Simple sorting by change percent for now
        if sort_by == 'change_percent':
            try:
                return sorted(quotes, 
                            key=lambda x: float(x.get('change_percent', '0').replace('%', '')), 
                            reverse=True)
            except:
                pass
        
        return quotes
    
    def get_stock_details(self, symbol):
        """Get detailed information for a single stock"""
        try:
            # Get basic quote
            quote = self.fetcher.get_quote(symbol)
            if not quote:
                return None
            
            # Add to cache
            self.stocks_cache[symbol] = quote
            
            # Get additional details
            details = {
                **quote,
                'fundamentals': self.fetcher.get_stock_fundamentals(symbol),
                'news': self.fetcher.get_news(symbol, limit=3)
            }
            
            return details
            
        except Exception as e:
            logger.error(f"Error getting details for {symbol}: {e}")
            return None
    
    def search_stocks(self, query, limit=10):
        """Search for stock symbols"""
        return self.fetcher.search_stocks(query, limit)
    
    def get_market_status(self):
        """Get current market status"""
        now = datetime.now()
        
        # Simple market hours check
        is_weekday = now.weekday() < 5
        is_market_hours = is_weekday and 9 <= now.hour < 16
        
        return {
            'status': 'open' if is_market_hours else 'closed',
            'is_open': is_market_hours,
            'current_time': now.isoformat()
        }