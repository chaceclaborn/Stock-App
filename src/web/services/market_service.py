# src/web/services/market_service.py
"""
Market data service layer
"""
import logging
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd

logger = logging.getLogger(__name__)

class MarketService:
    """Service for market-wide data operations"""
    
    def __init__(self):
        self._db = None
        self._fetcher = None
        self.market_cache = None
        self.market_cache_time = None
        self.cache_duration = 60  # seconds
    
    def init_app(self, db, fetcher):
        """Initialize with dependencies"""
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
    
    def get_market_overview(self):
        """Get overall market statistics and indices"""
        now = datetime.now()
        
        # Check cache
        if (self.market_cache is not None and 
            self.market_cache_time is not None and
            (now - self.market_cache_time).total_seconds() < self.cache_duration):
            
            return {
                **self.market_cache,
                'from_cache': True
            }
        
        try:
            # Get market indices
            indices_data = self.fetcher.get_market_overview()
            
            # Determine market status
            market_status = self._get_market_status()
            
            # Cache the result
            self.market_cache = {
                'indices': indices_data,
                'market_status': market_status,
                'timestamp': now.isoformat()
            }
            self.market_cache_time = now
            
            return {
                **self.market_cache,
                'from_cache': False
            }
            
        except Exception as e:
            logger.error(f"Error getting market overview: {e}")
            return None
    
    def get_sector_performance(self):
        """Get performance data for market sectors"""
        try:
            # Define sector ETFs
            sector_etfs = {
                'Technology': 'XLK',
                'Healthcare': 'XLV',
                'Financials': 'XLF',
                'Consumer Discretionary': 'XLY',
                'Consumer Staples': 'XLP',
                'Energy': 'XLE',
                'Industrials': 'XLI',
                'Materials': 'XLB',
                'Real Estate': 'XLRE',
                'Utilities': 'XLU',
                'Communications': 'XLC'
            }
            
            sector_data = {}
            
            for sector, etf in sector_etfs.items():
                try:
                    ticker = yf.Ticker(etf)
                    hist = ticker.history(period="5d")
                    
                    if not hist.empty and len(hist) >= 2:
                        current = hist['Close'].iloc[-1]
                        previous = hist['Close'].iloc[-2]
                        change = current - previous
                        change_pct = (change / previous) * 100
                        
                        # 5-day performance
                        five_day_ago = hist['Close'].iloc[0]
                        five_day_change = ((current - five_day_ago) / five_day_ago) * 100
                        
                        sector_data[sector] = {
                            'symbol': etf,
                            'price': float(current),
                            'change': float(change),
                            'change_percent': float(change_pct),
                            'five_day_change': float(five_day_change),
                            'volume': int(hist['Volume'].iloc[-1])
                        }
                        
                except Exception as e:
                    logger.error(f"Error getting sector data for {etf}: {e}")
            
            # Sort by performance
            sorted_sectors = sorted(
                sector_data.items(), 
                key=lambda x: x[1]['change_percent'], 
                reverse=True
            )
            
            return {
                'sectors': dict(sorted_sectors),
                'best_performer': sorted_sectors[0][0] if sorted_sectors else None,
                'worst_performer': sorted_sectors[-1][0] if sorted_sectors else None,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting sector performance: {e}")
            return None
    
    def get_market_movers(self):
        """Get top gainers, losers, and most active stocks"""
        try:
            all_symbols = self.fetcher.get_all_tracked_stocks()
            
            # Get quotes for all stocks
            quotes = self.fetcher.get_multiple_quotes(all_symbols)
            
            if not quotes:
                return None
            
            # Filter valid quotes
            valid_quotes = [
                q for q in quotes 
                if q.get('change') is not None and q.get('volume', 0) > 0
            ]
            
            # Sort for different categories
            gainers = sorted(
                [q for q in valid_quotes if q['change'] > 0],
                key=lambda x: x['change'],
                reverse=True
            )[:10]
            
            losers = sorted(
                [q for q in valid_quotes if q['change'] < 0],
                key=lambda x: x['change']
            )[:10]
            
            most_active = sorted(
                valid_quotes,
                key=lambda x: x.get('volume', 0),
                reverse=True
            )[:10]
            
            return {
                'gainers': gainers,
                'losers': losers,
                'most_active': most_active,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting market movers: {e}")
            return None
    
    def get_market_calendar(self):
        """Get upcoming market events"""
        try:
            # This is a placeholder - in production, you'd integrate with
            # economic calendar APIs, earnings calendar, etc.
            
            calendar_events = {
                'market_holidays': self._get_market_holidays(),
                'earnings_today': [],  # TODO: Integrate earnings calendar
                'economic_events': [],  # TODO: Integrate economic calendar
                'dividend_dates': [],   # TODO: Get dividend calendars
                'ipo_calendar': []      # TODO: Get IPO calendar
            }
            
            return calendar_events
            
        except Exception as e:
            logger.error(f"Error getting market calendar: {e}")
            return None
    
    def get_market_breadth(self):
        """Get market breadth indicators"""
        try:
            # Get data for major indices components
            # This is simplified - in production, you'd track all S&P 500 components
            
            sample_stocks = self.fetcher.get_all_tracked_stocks()[:50]
            quotes = self.fetcher.get_multiple_quotes(sample_stocks)
            
            if not quotes:
                return None
            
            # Calculate breadth metrics
            advances = sum(1 for q in quotes if q.get('change', 0) > 0)
            declines = sum(1 for q in quotes if q.get('change', 0) < 0)
            unchanged = sum(1 for q in quotes if q.get('change', 0) == 0)
            
            total = len(quotes)
            
            # Calculate indicators
            advance_decline_ratio = advances / declines if declines > 0 else float('inf')
            percent_above_50ma = self._calculate_percent_above_ma(sample_stocks, 50)
            percent_above_200ma = self._calculate_percent_above_ma(sample_stocks, 200)
            
            # New highs/lows (simplified)
            new_highs = sum(1 for q in quotes if self._is_new_high(q['symbol']))
            new_lows = sum(1 for q in quotes if self._is_new_low(q['symbol']))
            
            return {
                'advances': advances,
                'declines': declines,
                'unchanged': unchanged,
                'advance_decline_ratio': advance_decline_ratio,
                'advance_decline_line': advances - declines,
                'percent_positive': (advances / total * 100) if total > 0 else 0,
                'percent_above_50ma': percent_above_50ma,
                'percent_above_200ma': percent_above_200ma,
                'new_highs': new_highs,
                'new_lows': new_lows,
                'mcclellan_oscillator': self._calculate_mcclellan(advances, declines),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting market breadth: {e}")
            return None
    
    def _get_market_status(self):
        """Determine if market is open"""
        now = datetime.now()
        
        # Simple check - NYSE hours (9:30 AM - 4:00 PM ET)
        # This should account for timezone and holidays in production
        if now.weekday() >= 5:  # Weekend
            return 'closed'
        
        # Convert to ET (simplified)
        hour = now.hour
        minute = now.minute
        
        if hour < 9 or (hour == 9 and minute < 30):
            return 'pre-market'
        elif hour >= 16:
            return 'after-hours'
        else:
            return 'open'
    
    def _get_market_holidays(self):
        """Get upcoming market holidays"""
        # Simplified list of 2025 market holidays
        holidays = [
            {'date': '2025-01-01', 'holiday': "New Year's Day"},
            {'date': '2025-01-20', 'holiday': 'Martin Luther King Jr. Day'},
            {'date': '2025-02-17', 'holiday': "Presidents' Day"},
            {'date': '2025-04-18', 'holiday': 'Good Friday'},
            {'date': '2025-05-26', 'holiday': 'Memorial Day'},
            {'date': '2025-06-19', 'holiday': 'Juneteenth'},
            {'date': '2025-07-04', 'holiday': 'Independence Day'},
            {'date': '2025-09-01', 'holiday': 'Labor Day'},
            {'date': '2025-11-27', 'holiday': 'Thanksgiving Day'},
            {'date': '2025-12-25', 'holiday': 'Christmas Day'}
        ]
        
        # Filter to upcoming holidays
        today = datetime.now().date()
        upcoming = [
            h for h in holidays 
            if datetime.strptime(h['date'], '%Y-%m-%d').date() >= today
        ][:3]  # Next 3 holidays
        
        return upcoming
    
    def _calculate_percent_above_ma(self, symbols, period):
        """Calculate percentage of stocks above moving average"""
        above_ma = 0
        total = 0
        
        for symbol in symbols[:20]:  # Sample for performance
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=f"{period+10}d")
                
                if len(hist) >= period:
                    current_price = hist['Close'].iloc[-1]
                    ma = hist['Close'].tail(period).mean()
                    
                    if current_price > ma:
                        above_ma += 1
                    total += 1
                    
            except:
                continue
        
        return (above_ma / total * 100) if total > 0 else 50
    
    def _is_new_high(self, symbol):
        """Check if stock is at 52-week high"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            current = info.get('regularMarketPrice', 0)
            high_52w = info.get('fiftyTwoWeekHigh', 0)
            
            return current >= high_52w * 0.98  # Within 2% of 52w high
            
        except:
            return False
    
    def _is_new_low(self, symbol):
        """Check if stock is at 52-week low"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            current = info.get('regularMarketPrice', 0)
            low_52w = info.get('fiftyTwoWeekLow', float('inf'))
            
            return current <= low_52w * 1.02  # Within 2% of 52w low
            
        except:
            return False
    
    def _calculate_mcclellan(self, advances, declines):
        """Simplified McClellan Oscillator calculation"""
        # This is a simplified version
        # Real calculation requires historical advance/decline data
        net_advances = advances - declines
        total = advances + declines
        
        if total == 0:
            return 0
        
        ratio = net_advances / total
        return ratio * 1000  # Scaled value