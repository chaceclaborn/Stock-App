# src/web/services/stock_service.py
"""
Stock data service layer
"""
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class StockService:
    """Service for handling stock data operations"""
    
    def __init__(self):
        self.last_update_time = None
        self.update_interval = 30  # seconds
        self._db = None
        self._fetcher = None
    
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
    
    def get_stocks(self, category='all', force_refresh=False, sort_by='market_cap'):
        """Get stocks with caching logic"""
        now = datetime.now()
        
        # Get symbols based on category
        if category == 'all':
            symbols = self.fetcher.get_all_tracked_stocks()[:50]
        elif category == 'trending':
            symbols = self.fetcher.get_top_stocks()
        else:
            symbols = self.fetcher.get_stocks_by_category(category)
        
        # Check cache first
        cached_quotes = self.db.get_latest_quotes(symbols)
        quotes = self._format_quotes(cached_quotes)
        
        # Determine if update needed
        should_update = (
            force_refresh or
            self.last_update_time is None or
            (now - self.last_update_time).total_seconds() >= self.update_interval or
            len(quotes) < len(symbols) * 0.5
        )
        
        if should_update:
            try:
                fresh_quotes = self.fetcher.get_multiple_quotes(symbols)
                if fresh_quotes:
                    quotes = fresh_quotes
                    self.last_update_time = now
                    logger.info(f"Updated {len(fresh_quotes)} stock quotes")
            except Exception as e:
                logger.error(f"Error fetching fresh quotes: {e}")
        
        # Sort quotes
        quotes = self._sort_quotes(quotes, sort_by)
        
        return {
            'stocks': quotes,
            'last_updated': now.strftime('%Y-%m-%d %H:%M:%S'),
            'next_update': (now + timedelta(seconds=self.update_interval)).strftime('%Y-%m-%d %H:%M:%S'),
            'from_cache': not should_update
        }
    
    def get_cached_stocks(self, limit=100):
        """Get cached stocks from database"""
        cached_quotes = self.db.get_latest_quotes(limit=limit)
        quotes = self._format_quotes(cached_quotes)
        
        return {
            'stocks': quotes,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def get_realtime_quotes(self, symbols):
        """Get real-time quotes for specific symbols"""
        return self.fetcher.get_multiple_quotes(symbols)
    
    def get_stock_detail(self, symbol, period='1m'):
        """Get detailed stock information"""
        # Map period to yfinance period
        period_map = {
            '1d': '5d',
            '1w': '1mo',
            '1m': '3mo',
            '3m': '6mo',
            '6m': '1y',
            '1y': '2y'
        }
        yf_period = period_map.get(period, '3mo')
        
        # Get all data
        df = self.fetcher.get_stock_data(symbol, period=yf_period)
        quote = self.fetcher.get_quote(symbol)
        name = self.fetcher.get_company_name(symbol)
        fundamentals = self.fetcher.get_stock_fundamentals(symbol)
        realtime = self.fetcher.get_realtime_metrics(symbol)
        news = self.fetcher.get_stock_news(symbol)
        
        # Prepare daily data
        daily_data = []
        if not df.empty:
            for date, row in df.iterrows():
                daily_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': int(row['volume'])
                })
        
        return {
            'symbol': symbol,
            'name': name,
            'quote': quote,
            'daily_data': daily_data,
            'fundamentals': fundamentals,
            'realtime_metrics': realtime,
            'news': news,
            'period': period,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def search_stocks(self, query):
        """Search for stocks"""
        results = self.fetcher.search_stocks(query)
        
        # Get basic info for each result
        search_results = []
        for symbol in results:
            try:
                quote = self.fetcher.get_quote(symbol)
                if quote:
                    search_results.append({
                        'symbol': symbol,
                        'name': quote.get('name', symbol),
                        'price': quote['price'],
                        'change': quote['change'],
                        'change_percent': quote['change_percent']
                    })
            except:
                pass
        
        return search_results
    
    def get_categories(self):
        """Get available stock categories"""
        return list(self.fetcher.STOCK_CATEGORIES.keys())
    
    def _format_quotes(self, cached_quotes):
        """Format cached quotes to standard format"""
        quotes = []
        for quote in cached_quotes:
            formatted_quote = {
                'symbol': quote['symbol'],
                'name': self.fetcher.get_company_name(quote['symbol']),
                'price': quote['price'],
                'change': quote['change'],
                'change_percent': quote['change_percent'],
                'volume': quote['volume'],
                'last_updated': quote['timestamp'],
                'from_cache': True
            }
            quotes.append(formatted_quote)
        return quotes
    
    def _sort_quotes(self, quotes, sort_by):
        """Sort quotes by specified criteria"""
        if sort_by == 'change':
            quotes.sort(key=lambda x: x.get('change', 0), reverse=True)
        elif sort_by == 'volume':
            quotes.sort(key=lambda x: x.get('volume', 0), reverse=True)
        elif sort_by == 'sentiment':
            quotes.sort(key=lambda x: x.get('sentiment', 0.5), reverse=True)
        
        return quotes