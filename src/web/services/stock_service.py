# src/web/services/stock_service.py
"""
Stock data service layer with intelligent caching and update management
"""
from datetime import datetime, timedelta
import logging
import time
from collections import defaultdict

logger = logging.getLogger(__name__)

class StockService:
    """Service for handling stock data operations with smart caching"""
    
    def __init__(self):
        # Update intervals based on market hours
        self.market_hours_interval = 60  # 1 minute during market hours
        self.after_hours_interval = 300  # 5 minutes after hours
        self.weekend_interval = 3600  # 1 hour on weekends
        
        # Last update tracking
        self.last_full_update = None
        self.last_update_by_category = defaultdict(lambda: None)
        self.update_in_progress = False
        
        # Cache
        self.stocks_cache = {}
        self.cache_timestamp = None
        
        self._db = None
        self._fetcher = None
    
    def init_app(self, db, fetcher):
        """Initialize with database and fetcher instances"""
        self._db = db
        self._fetcher = fetcher
        
        # Pre-load cache from database
        self._load_initial_cache()
    
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
    
    def _load_initial_cache(self):
        """Load initial cache from database"""
        try:
            logger.info("Loading initial cache from database...")
            cached_quotes = self.db.get_latest_quotes(limit=100)
            
            for quote in cached_quotes:
                self.stocks_cache[quote['symbol']] = {
                    'symbol': quote['symbol'],
                    'name': self.fetcher.get_company_name(quote['symbol']),
                    'price': quote['price'],
                    'change': quote['change'],
                    'change_percent': quote['change_percent'],
                    'volume': quote['volume'],
                    'last_updated': quote['timestamp']
                }
            
            self.cache_timestamp = datetime.now()
            logger.info(f"Loaded {len(self.stocks_cache)} stocks into cache")
            
        except Exception as e:
            logger.error(f"Error loading initial cache: {e}")
    
    def _get_update_interval(self):
        """Get appropriate update interval based on market status"""
        now = datetime.now()
        
        # Simple market hours check (9:30 AM - 4:00 PM ET, Mon-Fri)
        # In production, this should account for holidays and timezone
        if now.weekday() >= 5:  # Weekend
            return self.weekend_interval
        
        hour = now.hour
        if 9 <= hour < 16:  # Rough market hours (adjust for timezone)
            return self.market_hours_interval
        else:
            return self.after_hours_interval
    
    def _should_update(self, category='all', force=False):
        """Determine if we should update stocks"""
        if force:
            return True
        
        if self.update_in_progress:
            return False
        
        now = datetime.now()
        interval = self._get_update_interval()
        
        # Check last update time for category
        last_update = self.last_update_by_category.get(category)
        if last_update is None:
            return True
        
        time_since_update = (now - last_update).total_seconds()
        return time_since_update >= interval
    
    def get_stocks(self, category='all', force_refresh=False, sort_by='market_cap'):
        """Get stocks with intelligent caching and updating"""
        now = datetime.now()
        
        # Return cache immediately if available
        if self.stocks_cache and not force_refresh:
            # Get symbols for category
            if category == 'all':
                symbols = list(self.stocks_cache.keys())[:50]
            else:
                symbols = self.fetcher.get_stocks_by_category(category)
            
            # Filter cached stocks
            cached_stocks = [
                self.stocks_cache[symbol] for symbol in symbols 
                if symbol in self.stocks_cache
            ]
            
            # Sort
            cached_stocks = self._sort_quotes(cached_stocks, sort_by)
            
            # Determine if background update needed
            should_update = self._should_update(category, force_refresh)
            
            response = {
                'stocks': cached_stocks,
                'last_updated': self.cache_timestamp.isoformat() if self.cache_timestamp else now.isoformat(),
                'next_update': (now + timedelta(seconds=self._get_update_interval())).isoformat(),
                'from_cache': True,
                'update_scheduled': should_update
            }
            
            # Schedule background update if needed
            if should_update and not self.update_in_progress:
                # In production, this would be done with a background task queue
                logger.info(f"Background update needed for {category}")
                self._schedule_background_update(category)
            
            return response
        
        # No cache available, must fetch
        return self._fetch_and_update_stocks(category, sort_by)
    
    def _fetch_and_update_stocks(self, category='all', sort_by='market_cap'):
        """Fetch fresh stock data and update cache"""
        if self.update_in_progress:
            # Return stale cache if update is already running
            return self.get_cached_stocks()
        
        try:
            self.update_in_progress = True
            now = datetime.now()
            
            # Get symbols based on category
            if category == 'all':
                symbols = self.fetcher.get_all_tracked_stocks()[:50]
            elif category == 'trending':
                symbols = self.fetcher.get_top_stocks()
            else:
                symbols = self.fetcher.get_stocks_by_category(category)
            
            # Fetch quotes with rate limiting handled by fetcher
            logger.info(f"Fetching quotes for {len(symbols)} stocks in category: {category}")
            fresh_quotes = self.fetcher.get_multiple_quotes(symbols)
            
            if fresh_quotes:
                # Update cache
                for quote in fresh_quotes:
                    self.stocks_cache[quote['symbol']] = quote
                
                self.cache_timestamp = now
                self.last_update_by_category[category] = now
                
                logger.info(f"Updated {len(fresh_quotes)} stock quotes")
            
            # Sort quotes
            quotes = self._sort_quotes(fresh_quotes, sort_by)
            
            return {
                'stocks': quotes,
                'last_updated': now.isoformat(),
                'next_update': (now + timedelta(seconds=self._get_update_interval())).isoformat(),
                'from_cache': False
            }
            
        except Exception as e:
            logger.error(f"Error fetching fresh quotes: {e}")
            # Return cached data on error
            return self.get_cached_stocks()
        finally:
            self.update_in_progress = False
    
    def _schedule_background_update(self, category):
        """Schedule a background update (simplified version)"""
        # In production, this would use Celery or similar task queue
        # For now, we'll just mark that an update is needed
        logger.info(f"Background update scheduled for {category}")
        
        # You could implement a simple thread here, but be careful with
        # Flask's request context if you do
        pass
    
    def get_cached_stocks(self, limit=100):
        """Get cached stocks from memory and database"""
        # First try memory cache
        if self.stocks_cache:
            stocks = list(self.stocks_cache.values())[:limit]
            return {
                'stocks': stocks,
                'last_updated': self.cache_timestamp.isoformat() if self.cache_timestamp else datetime.now().isoformat(),
                'from_cache': True
            }
        
        # Fall back to database
        cached_quotes = self.db.get_latest_quotes(limit=limit)
        quotes = self._format_quotes(cached_quotes)
        
        return {
            'stocks': quotes,
            'last_updated': datetime.now().isoformat(),
            'from_cache': True
        }
    
    def get_realtime_quotes(self, symbols):
        """Get real-time quotes for specific symbols"""
        # First check cache
        cached_quotes = []
        symbols_to_fetch = []
        
        for symbol in symbols:
            if symbol in self.stocks_cache:
                # Check if cache is fresh enough (1 minute)
                cache_age = (datetime.now() - datetime.fromisoformat(
                    self.stocks_cache[symbol].get('last_updated', '2000-01-01')
                )).total_seconds()
                
                if cache_age < 60:
                    cached_quotes.append(self.stocks_cache[symbol])
                else:
                    symbols_to_fetch.append(symbol)
            else:
                symbols_to_fetch.append(symbol)
        
        # Fetch any missing quotes
        if symbols_to_fetch:
            fresh_quotes = self.fetcher.get_multiple_quotes(symbols_to_fetch)
            
            # Update cache
            for quote in fresh_quotes:
                self.stocks_cache[quote['symbol']] = quote
            
            return cached_quotes + fresh_quotes
        
        return cached_quotes
    
    def get_stock_detail(self, symbol, period='1m'):
        """Get detailed stock information with caching"""
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
        
        # Check if we have recent quote in cache
        quote = None
        if symbol in self.stocks_cache:
            cache_age = (datetime.now() - datetime.fromisoformat(
                self.stocks_cache[symbol].get('last_updated', '2000-01-01')
            )).total_seconds()
            
            if cache_age < 300:  # 5 minutes
                quote = self.stocks_cache[symbol]
        
        if not quote:
            quote = self.fetcher.get_quote(symbol)
            if quote:
                self.stocks_cache[symbol] = quote
        
        # Get other data (these have their own caching in fetcher)
        df = self.fetcher.get_stock_data(symbol, period=yf_period)
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
            'last_updated': datetime.now().isoformat()
        }
    
    def search_stocks(self, query):
        """Search for stocks"""
        results = self.fetcher.search_stocks(query)
        
        # Get basic info for each result
        search_results = []
        for symbol in results:
            # Check cache first
            if symbol in self.stocks_cache:
                quote = self.stocks_cache[symbol]
                search_results.append({
                    'symbol': symbol,
                    'name': quote.get('name', symbol),
                    'price': quote['price'],
                    'change': quote['change'],
                    'change_percent': quote['change_percent']
                })
            else:
                # Fetch if not in cache (rate limited by fetcher)
                try:
                    quote = self.fetcher.get_quote(symbol)
                    if quote:
                        self.stocks_cache[symbol] = quote
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
        if not quotes:
            return quotes
        
        if sort_by == 'change':
            quotes.sort(key=lambda x: x.get('change', 0), reverse=True)
        elif sort_by == 'volume':
            quotes.sort(key=lambda x: x.get('volume', 0), reverse=True)
        elif sort_by == 'alpha':
            quotes.sort(key=lambda x: x.get('symbol', ''))
        elif sort_by == 'price':
            quotes.sort(key=lambda x: x.get('price', 0), reverse=True)
        
        return quotes
    
    def refresh_cache(self):
        """Force refresh the entire cache (use sparingly)"""
        try:
            logger.info("Force refreshing entire stock cache...")
            symbols = self.fetcher.get_all_tracked_stocks()[:50]
            
            # Fetch in small batches
            batch_size = 10
            all_quotes = []
            
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i+batch_size]
                quotes = self.fetcher.get_multiple_quotes(batch)
                all_quotes.extend(quotes)
                
                # Update cache as we go
                for quote in quotes:
                    self.stocks_cache[quote['symbol']] = quote
                
                # Small delay between batches
                if i + batch_size < len(symbols):
                    time.sleep(2)
            
            self.cache_timestamp = datetime.now()
            self.last_full_update = datetime.now()
            
            logger.info(f"Cache refresh complete. Updated {len(all_quotes)} stocks")
            
        except Exception as e:
            logger.error(f"Error refreshing cache: {e}")