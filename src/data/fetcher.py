# src/data/fetcher.py
import yfinance as yf
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from threading import Lock
import time
import requests
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiter for API calls"""
    def __init__(self, calls_per_minute=30):
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute
        self.last_call = defaultdict(float)
        self.lock = Lock()
    
    def wait_if_needed(self, key='default'):
        """Wait if necessary to respect rate limit"""
        with self.lock:
            now = time.time()
            time_since_last = now - self.last_call[key]
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                time.sleep(sleep_time)
            self.last_call[key] = time.time()

class DataCache:
    """Simple cache with TTL"""
    def __init__(self, default_ttl=300):  # 5 minutes default
        self.cache = {}
        self.timestamps = {}
        self.default_ttl = default_ttl
        self.lock = Lock()
    
    def get(self, key):
        """Get item from cache if not expired"""
        with self.lock:
            if key in self.cache:
                if time.time() - self.timestamps[key] < self.default_ttl:
                    return self.cache[key]
                else:
                    # Expired
                    del self.cache[key]
                    del self.timestamps[key]
            return None
    
    def set(self, key, value):
        """Set item in cache"""
        with self.lock:
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def clear_expired(self):
        """Clear expired items"""
        with self.lock:
            now = time.time()
            expired_keys = [k for k, t in self.timestamps.items() if now - t >= self.default_ttl]
            for key in expired_keys:
                del self.cache[key]
                del self.timestamps[key]

class StockDataFetcher:
    """
    Enhanced stock data fetcher using Yahoo Finance with rate limiting and caching.
    """
    
    # Stock categories for organization
    STOCK_CATEGORIES = {
        'mega_cap': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B'],
        'tech': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'TSLA', 'ORCL', 'CRM', 'ADBE', 'INTC', 'AMD', 'CSCO'],
        'finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'V', 'MA', 'PYPL'],
        'healthcare': ['UNH', 'JNJ', 'PFE', 'LLY', 'ABBV', 'MRK', 'TMO', 'ABT', 'CVS', 'AMGN', 'GILD', 'BMY'],
        'consumer': ['AMZN', 'WMT', 'HD', 'PG', 'KO', 'PEP', 'COST', 'NKE', 'MCD', 'SBUX', 'TGT', 'DIS'],
        'energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'MPC', 'PSX', 'VLO', 'OXY'],
        'etfs': ['SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'VOO', 'GLD', 'SLV', 'TLT', 'HYG'],
        'crypto': ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'DOGE-USD']
    }
    
    # Popular stocks with names (simplified list)
    STOCK_INFO = {
        'AAPL': 'Apple Inc.',
        'MSFT': 'Microsoft Corporation',
        'GOOGL': 'Alphabet Inc.',
        'AMZN': 'Amazon.com Inc.',
        'META': 'Meta Platforms Inc.',
        'NVDA': 'NVIDIA Corporation',
        'TSLA': 'Tesla Inc.',
        'JPM': 'JPMorgan Chase & Co.',
        'BAC': 'Bank of America Corp.',
        'JNJ': 'Johnson & Johnson',
        'UNH': 'UnitedHealth Group Inc.',
        'PFE': 'Pfizer Inc.',
        'WMT': 'Walmart Inc.',
        'PG': 'Procter & Gamble Co.',
        'KO': 'The Coca-Cola Company',
        'DIS': 'Walt Disney Company',
        'HD': 'Home Depot Inc.',
        'V': 'Visa Inc.',
        'MA': 'Mastercard Inc.',
        'NFLX': 'Netflix Inc.',
        'ADBE': 'Adobe Inc.',
        'CRM': 'Salesforce Inc.',
        'PYPL': 'PayPal Holdings Inc.',
        'INTC': 'Intel Corporation',
        'AMD': 'Advanced Micro Devices Inc.',
        'CSCO': 'Cisco Systems Inc.',
        'PEP': 'PepsiCo Inc.',
        'TMO': 'Thermo Fisher Scientific Inc.',
        'ABBV': 'AbbVie Inc.',
        'CVX': 'Chevron Corporation',
        'MRK': 'Merck & Co. Inc.',
        'NKE': 'Nike Inc.',
        'LLY': 'Eli Lilly and Company',
        'XOM': 'Exxon Mobil Corporation',
        'SPY': 'SPDR S&P 500 ETF Trust',
        'QQQ': 'Invesco QQQ Trust',
        'IWM': 'iShares Russell 2000 ETF',
        'DIA': 'SPDR Dow Jones Industrial Average ETF',
        'VTI': 'Vanguard Total Stock Market ETF',
        'GLD': 'SPDR Gold Trust',
        'ORCL': 'Oracle Corporation',
        'COST': 'Costco Wholesale Corporation',
        'BA': 'Boeing Company',
        'CAT': 'Caterpillar Inc.',
        'IBM': 'IBM Corporation',
        'GS': 'Goldman Sachs Group Inc.',
        'MS': 'Morgan Stanley',
        'C': 'Citigroup Inc.',
        'AMGN': 'Amgen Inc.',
        'HON': 'Honeywell International Inc.'
    }
    
    def __init__(self, db=None):
        """Initialize with database connection."""
        self.db = db
        self.lock = Lock()
        
        # Rate limiting
        self.rate_limiter = RateLimiter(calls_per_minute=30)  # Yahoo Finance unofficial limit
        
        # Caching
        self.quote_cache = DataCache(default_ttl=60)  # 1 minute for quotes
        self.data_cache = DataCache(default_ttl=300)  # 5 minutes for historical data
        self.info_cache = DataCache(default_ttl=3600)  # 1 hour for company info
        
        # Batch fetching state
        self.last_batch_fetch = None
        self.batch_fetch_interval = 60  # Minimum seconds between batch fetches
        
        logger.info("Enhanced Stock Data Fetcher initialized with rate limiting and caching")
    
    def get_stocks_by_category(self, category):
        """Get stocks for a specific category"""
        if category in self.STOCK_CATEGORIES:
            return self.STOCK_CATEGORIES[category]
        return []
    
    def get_company_name(self, symbol):
        """Get company name for a symbol with caching"""
        # First check our known stocks
        if symbol in self.STOCK_INFO:
            return self.STOCK_INFO[symbol]
        
        # Check cache
        cache_key = f"name_{symbol}"
        cached_name = self.info_cache.get(cache_key)
        if cached_name:
            return cached_name
        
        # Try to get from Yahoo Finance with rate limiting
        try:
            self.rate_limiter.wait_if_needed('info')
            ticker = yf.Ticker(symbol)
            info = ticker.info
            name = info.get('longName', info.get('shortName', symbol))
            self.info_cache.set(cache_key, name)
            return name
        except Exception as e:
            logger.debug(f"Could not get name for {symbol}: {e}")
            return symbol
    
    def get_all_tracked_stocks(self):
        """Get all stocks we're tracking"""
        return list(self.STOCK_INFO.keys())
    
    def search_stocks(self, query, limit=10):
        """Search for stocks by symbol or company name"""
        try:
            query = query.upper()
            all_stocks = self.get_all_tracked_stocks()
            
            # Filter stocks that match the query
            matches = []
            
            # Check symbols
            for symbol in all_stocks:
                if query in symbol:
                    matches.append(symbol)
            
            # Check company names
            for symbol, name in self.STOCK_INFO.items():
                if query.lower() in name.lower() and symbol not in matches:
                    matches.append(symbol)
            
            # If no matches in our tracked stocks, try Yahoo Finance
            if not matches and len(query) >= 1:
                try:
                    # Only check if it's a valid symbol with rate limiting
                    self.rate_limiter.wait_if_needed('search')
                    ticker = yf.Ticker(query)
                    info = ticker.info
                    if 'symbol' in info:
                        matches = [query]
                except:
                    pass
            
            return matches[:limit]
        except Exception as e:
            logger.error(f"Error searching stocks: {e}")
            return []
    
    def get_market_overview(self):
        """Get overall market statistics and indices with caching"""
        # Check cache first
        cached_overview = self.data_cache.get('market_overview')
        if cached_overview:
            logger.info("Returning cached market overview")
            return cached_overview
        
        try:
            indices = {
                '^GSPC': 'S&P 500',
                '^DJI': 'Dow Jones',
                '^IXIC': 'NASDAQ',
                '^VIX': 'VIX (Fear Index)',
                '^TNX': '10-Year Treasury'
            }
            
            overview = {}
            
            # Fetch with rate limiting
            for symbol, name in indices.items():
                try:
                    self.rate_limiter.wait_if_needed('market')
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="2d")
                    if not hist.empty and len(hist) >= 2:
                        current = hist['Close'].iloc[-1]
                        previous = hist['Close'].iloc[-2]
                        change = current - previous
                        change_pct = (change / previous) * 100
                        
                        overview[name] = {
                            'symbol': symbol,
                            'value': float(current),
                            'change': float(change),
                            'change_percent': float(change_pct)
                        }
                except Exception as e:
                    logger.error(f"Error fetching {symbol}: {e}")
            
            # Calculate Fear & Greed Index based on VIX
            vix_value = overview.get('VIX (Fear Index)', {}).get('value', 20)
            if vix_value < 12:
                fear_greed = 80  # Extreme Greed
            elif vix_value < 20:
                fear_greed = 60  # Greed
            elif vix_value < 30:
                fear_greed = 40  # Fear
            else:
                fear_greed = 20  # Extreme Fear
                
            overview['fear_greed_index'] = fear_greed
            
            # Cache the result
            self.data_cache.set('market_overview', overview)
            
            return overview
            
        except Exception as e:
            logger.error(f"Error getting market overview: {e}")
            return {}
    
    def get_stock_data(self, symbol, period="3mo"):
        """Get daily stock data for a symbol using Yahoo Finance with caching."""
        # Check cache first
        cache_key = f"data_{symbol}_{period}"
        cached_data = self.data_cache.get(cache_key)
        if cached_data is not None:
            logger.info(f"Returning cached data for {symbol}")
            return cached_data
        
        # Check database for recent data
        if self.db:
            db_data = self.db.get_historical_data(symbol, days=100)
            if not db_data.empty and len(db_data) > 0:
                latest_date = db_data.index[0]
                if (datetime.now() - pd.to_datetime(latest_date)).days < 1:
                    logger.info(f"Using recent database data for {symbol}")
                    self.data_cache.set(cache_key, db_data)
                    return db_data
        
        try:
            # Rate limit the request
            self.rate_limiter.wait_if_needed('data')
            
            logger.info(f"Fetching data for {symbol} from Yahoo Finance")
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            
            if df.empty:
                logger.error(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df = df.sort_index(ascending=False)
            
            # Cache the result
            self.data_cache.set(cache_key, df)
            
            # Store in database
            if self.db and not df.empty:
                self.db.store_historical_data(symbol, df)
                logger.info(f"Stored {len(df)} days of data for {symbol} in database")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            # Try to return cached data from database
            if self.db:
                cached_data = self.db.get_historical_data(symbol, days=100)
                if not cached_data.empty:
                    logger.info(f"Returning database cached data for {symbol}")
                    return cached_data
            return pd.DataFrame()
    
    def get_quote(self, symbol):
        """Get real-time quote for a symbol with aggressive caching."""
        # Check memory cache first
        cache_key = f"quote_{symbol}"
        cached_quote = self.quote_cache.get(cache_key)
        if cached_quote:
            logger.info(f"Returning memory cached quote for {symbol}")
            return cached_quote
        
        # Check database cache
        if self.db:
            cached_quotes = self.db.get_latest_quotes([symbol])
            if cached_quotes:
                quote = cached_quotes[0]
                quote_time = datetime.fromisoformat(quote['timestamp'])
                if (datetime.now() - quote_time).seconds < 60:
                    logger.info(f"Using database cached quote for {symbol}")
                    result = {
                        'symbol': quote['symbol'],
                        'name': self.get_company_name(symbol),
                        'price': quote['price'],
                        'change': quote['change'],
                        'change_percent': quote['change_percent'],
                        'volume': quote['volume'],
                        'last_updated': quote['timestamp'],
                        'from_cache': True
                    }
                    # Cache in memory
                    self.quote_cache.set(cache_key, result)
                    return result
        
        try:
            # Rate limit the request
            self.rate_limiter.wait_if_needed('quote')
            
            logger.info(f"Fetching quote for {symbol} from Yahoo Finance")
            ticker = yf.Ticker(symbol)
            
            # Try to get from history first (more reliable)
            hist = ticker.history(period="5d")
            if not hist.empty and len(hist) >= 2:
                current_price = float(hist['Close'].iloc[-1])
                prev_close = float(hist['Close'].iloc[-2])
                volume = int(hist['Volume'].iloc[-1])
                
                change = current_price - prev_close
                change_percent = f"{(change / prev_close * 100):.2f}%"
                
                result = {
                    'symbol': symbol.upper(),
                    'name': self.get_company_name(symbol),
                    'price': current_price,
                    'change': change,
                    'change_percent': change_percent,
                    'volume': volume,
                    'last_updated': datetime.now().isoformat(),
                    'from_cache': False
                }
                
                # Cache the result
                self.quote_cache.set(cache_key, result)
                
                # Store in database
                if self.db:
                    self.db.store_quote(result)
                
                return result
            else:
                logger.error(f"No history data for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {str(e)}")
            # Try to get from database
            if self.db:
                historical = self.db.get_historical_data(symbol, days=2)
                if not historical.empty:
                    latest = historical.iloc[0]
                    previous = historical.iloc[1] if len(historical) > 1 else latest
                    
                    change = latest['close'] - previous['close']
                    change_percent = f"{(change / previous['close'] * 100):.2f}%"
                    
                    result = {
                        'symbol': symbol,
                        'name': self.get_company_name(symbol),
                        'price': float(latest['close']),
                        'change': float(change),
                        'change_percent': change_percent,
                        'volume': int(latest['volume']),
                        'last_updated': historical.index[0].isoformat(),
                        'from_cache': True
                    }
                    
                    # Cache in memory
                    self.quote_cache.set(cache_key, result)
                    return result
            
            return None
    
    def get_top_stocks(self):
        """Return top stocks to display"""
        return list(self.STOCK_INFO.keys())[:15]
    
    def get_multiple_quotes(self, symbols):
        """Get quotes for multiple symbols with batch fetching and caching."""
        quotes = []
        symbols_to_fetch = []
        
        # First, check cache for all symbols
        for symbol in symbols:
            cached_quote = self.get_quote_from_cache(symbol)
            if cached_quote:
                quotes.append(cached_quote)
            else:
                symbols_to_fetch.append(symbol)
        
        # If we need to fetch any symbols
        if symbols_to_fetch:
            # Check if we can do a batch fetch
            now = time.time()
            if (self.last_batch_fetch is None or 
                now - self.last_batch_fetch > self.batch_fetch_interval):
                
                logger.info(f"Batch fetching {len(symbols_to_fetch)} symbols")
                self.last_batch_fetch = now
                
                # Use yfinance download for batch fetching with rate limiting
                try:
                    # Process in smaller batches to avoid overwhelming the API
                    batch_size = 10
                    for i in range(0, len(symbols_to_fetch), batch_size):
                        batch = symbols_to_fetch[i:i+batch_size]
                        
                        # Rate limit between batches
                        if i > 0:
                            time.sleep(2)  # Wait 2 seconds between batches
                        
                        self.rate_limiter.wait_if_needed('batch')
                        
                        # Use download for batch fetching
                        batch_str = ' '.join(batch)
                        data = yf.download(batch_str, period='5d', group_by='ticker',
                                         auto_adjust=True, prepost=True, threads=False,
                                         progress=False)
                        
                        # Process each symbol in the batch
                        for symbol in batch:
                            try:
                                if len(batch) == 1:
                                    df = data
                                else:
                                    df = data[symbol] if symbol in data.columns.levels[0] else None
                                
                                if df is not None and not df.empty and len(df) >= 2:
                                    current_price = float(df['Close'].iloc[-1])
                                    prev_close = float(df['Close'].iloc[-2])
                                    volume = int(df['Volume'].iloc[-1])
                                    
                                    change = current_price - prev_close
                                    change_percent = f"{(change / prev_close * 100):.2f}%"
                                    
                                    quote = {
                                        'symbol': symbol.upper(),
                                        'name': self.get_company_name(symbol),
                                        'price': current_price,
                                        'change': change,
                                        'change_percent': change_percent,
                                        'volume': volume,
                                        'last_updated': datetime.now().isoformat(),
                                        'from_cache': False
                                    }
                                    
                                    quotes.append(quote)
                                    
                                    # Cache the quote
                                    self.quote_cache.set(f"quote_{symbol}", quote)
                                    
                                    # Store in database
                                    if self.db:
                                        self.db.store_quote(quote)
                                else:
                                    # Try individual fetch as fallback
                                    individual_quote = self.get_quote(symbol)
                                    if individual_quote:
                                        quotes.append(individual_quote)
                            except Exception as e:
                                logger.error(f"Error processing {symbol} in batch: {e}")
                                # Try individual fetch as fallback
                                individual_quote = self.get_quote(symbol)
                                if individual_quote:
                                    quotes.append(individual_quote)
                        
                        # Clear expired cache entries periodically
                        self.quote_cache.clear_expired()
                        
                except Exception as e:
                    logger.error(f"Batch quote fetch failed: {e}")
                    # Fall back to individual fetches with rate limiting
                    for symbol in symbols_to_fetch[:5]:  # Limit fallback fetches
                        quote = self.get_quote(symbol)
                        if quote:
                            quotes.append(quote)
                        time.sleep(1)  # Wait between individual fetches
            else:
                # Too soon for batch fetch, use cached data or limited individual fetches
                logger.info("Using cached data - too soon for batch fetch")
                for symbol in symbols_to_fetch[:3]:  # Very limited fetches
                    quote = self.get_quote(symbol)
                    if quote:
                        quotes.append(quote)
        
        logger.info(f"Retrieved {len(quotes)} quotes")
        return quotes
    
    def get_quote_from_cache(self, symbol):
        """Try to get quote from various cache layers"""
        # Check memory cache
        cache_key = f"quote_{symbol}"
        cached_quote = self.quote_cache.get(cache_key)
        if cached_quote:
            return cached_quote
        
        # Check database
        if self.db:
            cached_quotes = self.db.get_latest_quotes([symbol])
            if cached_quotes:
                quote = cached_quotes[0]
                quote_time = datetime.fromisoformat(quote['timestamp'])
                if (datetime.now() - quote_time).seconds < 300:  # 5 minutes
                    result = {
                        'symbol': quote['symbol'],
                        'name': self.get_company_name(symbol),
                        'price': quote['price'],
                        'change': quote['change'],
                        'change_percent': quote['change_percent'],
                        'volume': quote['volume'],
                        'last_updated': quote['timestamp'],
                        'from_cache': True
                    }
                    # Cache in memory
                    self.quote_cache.set(cache_key, result)
                    return result
        
        return None
    
    def get_stocks_for_analysis(self, symbols=None):
        """Get stock data prepared for technical analysis with rate limiting."""
        if symbols is None:
            symbols = self.get_all_tracked_stocks()
        
        stocks_data = {}
        
        try:
            logger.info(f"Downloading historical data for {len(symbols)} stocks...")
            
            # Process in small batches with rate limiting
            batch_size = 5  # Smaller batch size to avoid rate limits
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i+batch_size]
                
                # Rate limit between batches
                if i > 0:
                    time.sleep(3)  # Wait 3 seconds between batches
                
                # Check cache first
                cached_batch = []
                uncached_batch = []
                
                for symbol in batch:
                    cache_key = f"analysis_{symbol}"
                    cached_data = self.data_cache.get(cache_key)
                    if cached_data is not None:
                        stocks_data[symbol] = cached_data
                        cached_batch.append(symbol)
                    else:
                        uncached_batch.append(symbol)
                
                if cached_batch:
                    logger.info(f"Using cached data for: {cached_batch}")
                
                # Fetch uncached symbols
                if uncached_batch:
                    self.rate_limiter.wait_if_needed('analysis')
                    
                    try:
                        tickers_str = ' '.join(uncached_batch)
                        data = yf.download(tickers_str, period="3mo", group_by='ticker',
                                         auto_adjust=True, prepost=True, threads=False,
                                         progress=False)
                        
                        for symbol in uncached_batch:
                            try:
                                if len(uncached_batch) == 1:
                                    df = data
                                else:
                                    df = data[symbol] if symbol in data.columns.levels[0] else None
                                
                                if df is not None and not df.empty and len(df) >= 30:
                                    df = df.dropna()
                                    df = df.sort_index(ascending=True)
                                    
                                    stocks_data[symbol] = df
                                    
                                    # Cache the data
                                    self.data_cache.set(f"analysis_{symbol}", df)
                                    
                                    logger.info(f"Loaded {len(df)} days of data for {symbol}")
                                    
                                    if self.db:
                                        # For database storage, convert to lowercase
                                        db_df = df.rename(columns={
                                            'Open': 'open',
                                            'High': 'high',
                                            'Low': 'low',
                                            'Close': 'close',
                                            'Volume': 'volume'
                                        })
                                        db_df = db_df[['open', 'high', 'low', 'close', 'volume']]
                                        db_df = db_df.sort_index(ascending=False)
                                        self.db.store_historical_data(symbol, db_df)
                                        
                            except Exception as e:
                                logger.error(f"Error processing {symbol}: {e}")
                    
                    except Exception as e:
                        logger.error(f"Batch download failed for {uncached_batch}: {e}")
                        # Try individual downloads with more aggressive rate limiting
                        for symbol in uncached_batch[:2]:  # Only try first 2
                            try:
                                time.sleep(2)
                                df = self.get_stock_data(symbol)
                                if not df.empty:
                                    stocks_data[symbol] = df
                            except Exception as e2:
                                logger.error(f"Individual download failed for {symbol}: {e2}")
                
                # Clear expired cache entries
                self.data_cache.clear_expired()
                
        except Exception as e:
            logger.error(f"Analysis data fetch failed: {e}")
        
        logger.info(f"Successfully loaded data for {len(stocks_data)} stocks")
        return stocks_data
    
    def get_stock_fundamentals(self, symbol):
        """Get fundamental analysis data for a stock with caching"""
        cache_key = f"fundamentals_{symbol}"
        cached_data = self.info_cache.get(cache_key)
        if cached_data:
            return cached_data
        
        try:
            self.rate_limiter.wait_if_needed('fundamentals')
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            fundamentals = {
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                'profit_margin': info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 0,
                'revenue_growth': info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 0,
                'debt_to_equity': info.get('debtToEquity', 0),
                'roe': info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0,
                'beta': info.get('beta', 1),
                'analyst_rating': info.get('recommendationKey', 'none'),
                'price_target': info.get('targetMeanPrice', 0),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0),
                'shares_outstanding': info.get('sharesOutstanding', 0),
                'institutional_holders': info.get('institutionalHoldersPercentage', 0) * 100 if info.get('institutionalHoldersPercentage') else 0
            }
            
            self.info_cache.set(cache_key, fundamentals)
            return fundamentals
            
        except Exception as e:
            logger.error(f"Error getting fundamentals for {symbol}: {e}")
            return None
    
    def get_stock_news(self, symbol):
        """Get recent news for a stock with caching"""
        cache_key = f"news_{symbol}"
        cached_news = self.info_cache.get(cache_key)
        if cached_news:
            return cached_news
        
        try:
            self.rate_limiter.wait_if_needed('news')
            ticker = yf.Ticker(symbol)
            news = ticker.news[:5]  # Get latest 5 news items
            
            news_items = []
            for item in news:
                news_items.append({
                    'title': item.get('title', ''),
                    'publisher': item.get('publisher', ''),
                    'link': item.get('link', ''),
                    'timestamp': item.get('providerPublishTime', 0)
                })
            
            self.info_cache.set(cache_key, news_items)
            return news_items
        except Exception as e:
            logger.error(f"Error getting news for {symbol}: {e}")
            return []
    
    def get_realtime_metrics(self, symbol):
        """Get real-time metrics for a stock with caching"""
        cache_key = f"metrics_{symbol}"
        cached_metrics = self.quote_cache.get(cache_key)
        if cached_metrics:
            return cached_metrics
        
        try:
            self.rate_limiter.wait_if_needed('metrics')
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            metrics = {
                'bid': info.get('bid', 0),
                'ask': info.get('ask', 0),
                'bid_size': info.get('bidSize', 0),
                'ask_size': info.get('askSize', 0),
                'day_high': info.get('dayHigh', 0),
                'day_low': info.get('dayLow', 0),
                'open': info.get('open', 0),
                'previous_close': info.get('previousClose', 0),
                'fifty_day_average': info.get('fiftyDayAverage', 0),
                'two_hundred_day_average': info.get('twoHundredDayAverage', 0),
                'average_volume': info.get('averageVolume', 0),
                'average_volume_10days': info.get('averageVolume10days', 0),
                'exchange': info.get('exchange', ''),
                'quote_type': info.get('quoteType', ''),
                'currency': info.get('currency', 'USD')
            }
            
            self.quote_cache.set(cache_key, metrics)
            return metrics
        except Exception as e:
            logger.error(f"Error getting real-time metrics for {symbol}: {e}")
            return None