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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockDataFetcher:
    """
    Enhanced stock data fetcher using Yahoo Finance with real-time capabilities.
    Includes company names, news, and enhanced analysis features.
    """
    
    # Popular stock categories with names
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
        'GS': 'Goldman Sachs Group Inc.',
        'MS': 'Morgan Stanley',
        'WFC': 'Wells Fargo & Company',
        'C': 'Citigroup Inc.',
        'BLK': 'BlackRock Inc.',
        'SCHW': 'Charles Schwab Corp.',
        'JNJ': 'Johnson & Johnson',
        'UNH': 'UnitedHealth Group Inc.',
        'PFE': 'Pfizer Inc.',
        'ABBV': 'AbbVie Inc.',
        'TMO': 'Thermo Fisher Scientific Inc.',
        'MRK': 'Merck & Co. Inc.',
        'ABT': 'Abbott Laboratories',
        'CVS': 'CVS Health Corporation',
        'WMT': 'Walmart Inc.',
        'PG': 'Procter & Gamble Co.',
        'KO': 'The Coca-Cola Company',
        'PEP': 'PepsiCo Inc.',
        'MCD': "McDonald's Corporation",
        'NKE': 'Nike Inc.',
        'SBUX': 'Starbucks Corporation',
        'TGT': 'Target Corporation',
        'XOM': 'Exxon Mobil Corporation',
        'CVX': 'Chevron Corporation',
        'COP': 'ConocoPhillips',
        'SLB': 'Schlumberger NV',
        'EOG': 'EOG Resources Inc.',
        'MPC': 'Marathon Petroleum Corp.',
        'PSX': 'Phillips 66',
        'VLO': 'Valero Energy Corporation',
        'BA': 'Boeing Company',
        'CAT': 'Caterpillar Inc.',
        'GE': 'General Electric Company',
        'MMM': '3M Company',
        'HON': 'Honeywell International Inc.',
        'UPS': 'United Parcel Service Inc.',
        'RTX': 'Raytheon Technologies Corp.',
        'LMT': 'Lockheed Martin Corporation',
        'SPY': 'SPDR S&P 500 ETF Trust',
        'QQQ': 'Invesco QQQ Trust',
        'IWM': 'iShares Russell 2000 ETF',
        'DIA': 'SPDR Dow Jones Industrial Average ETF',
        'VTI': 'Vanguard Total Stock Market ETF',
        'VOO': 'Vanguard S&P 500 ETF',
        'EEM': 'iShares MSCI Emerging Markets ETF',
        'GLD': 'SPDR Gold Trust',
        'COIN': 'Coinbase Global Inc.',
        'MARA': 'Marathon Digital Holdings Inc.',
        'RIOT': 'Riot Platforms Inc.',
        'MSTR': 'MicroStrategy Inc.',
        'SQ': 'Block Inc.',
        'PYPL': 'PayPal Holdings Inc.',
        'GME': 'GameStop Corp.',
        'AMC': 'AMC Entertainment Holdings Inc.',
        'BB': 'BlackBerry Ltd.',
        'NOK': 'Nokia Corporation',
        'BBBY': 'Bed Bath & Beyond Inc.',
        'WISH': 'ContextLogic Inc.',
        'CLOV': 'Clover Health Investments Corp.',
        'ARKK': 'ARK Innovation ETF',
        'ARKQ': 'ARK Autonomous Technology & Robotics ETF',
        'ARKW': 'ARK Next Generation Internet ETF',
        'ARKG': 'ARK Genomic Revolution ETF',
        'ARKF': 'ARK Fintech Innovation ETF',
        'AMD': 'Advanced Micro Devices Inc.',
        'INTC': 'Intel Corporation',
        'MU': 'Micron Technology Inc.',
        'QCOM': 'QUALCOMM Inc.',
        'AVGO': 'Broadcom Inc.',
        'TXN': 'Texas Instruments Inc.',
        'ADI': 'Analog Devices Inc.',
        'MRVL': 'Marvell Technology Inc.',
        'RIVN': 'Rivian Automotive Inc.',
        'LCID': 'Lucid Group Inc.',
        'NIO': 'NIO Inc.',
        'XPEV': 'XPeng Inc.',
        'LI': 'Li Auto Inc.',
        'FSR': 'Fisker Inc.',
        'GOEV': 'Canoo Inc.',
        'PLTR': 'Palantir Technologies Inc.',
        'C3AI': 'C3.ai Inc.',
        'SNOW': 'Snowflake Inc.',
        'PATH': 'UiPath Inc.',
        'DDOG': 'Datadog Inc.',
        'NET': 'Cloudflare Inc.',
        'CRWD': 'CrowdStrike Holdings Inc.'
    }
    
    STOCK_CATEGORIES = {
        'mega_tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA'],
        'finance': ['JPM', 'BAC', 'GS', 'MS', 'WFC', 'C', 'BLK', 'SCHW'],
        'healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'MRK', 'ABT', 'CVS'],
        'consumer': ['WMT', 'PG', 'KO', 'PEP', 'MCD', 'NKE', 'SBUX', 'TGT'],
        'energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO'],
        'industrials': ['BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'RTX', 'LMT'],
        'etfs': ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'EEM', 'GLD'],
        'crypto_stocks': ['COIN', 'MARA', 'RIOT', 'MSTR', 'SQ', 'PYPL'],
        'meme_stocks': ['GME', 'AMC', 'BB', 'NOK', 'BBBY', 'WISH', 'CLOV'],
        'ark_invest': ['ARKK', 'ARKQ', 'ARKW', 'ARKG', 'ARKF'],
        'semiconductor': ['AMD', 'INTC', 'MU', 'QCOM', 'AVGO', 'TXN', 'ADI', 'MRVL'],
        'ev_stocks': ['RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'FSR', 'GOEV'],
        'ai_stocks': ['PLTR', 'C3AI', 'SNOW', 'PATH', 'DDOG', 'NET', 'CRWD']
    }
    
    def __init__(self, db=None):
        """Initialize with database connection."""
        self.db = db
        self.lock = Lock()
        logger.info("Enhanced Stock Data Fetcher initialized with real-time capabilities")
    
    def get_company_name(self, symbol):
        """Get company name for a symbol"""
        # First check our known stocks
        if symbol in self.STOCK_INFO:
            return self.STOCK_INFO[symbol]
        
        # Try to get from Yahoo Finance
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            if 'longName' in info:
                return info['longName']
            elif 'shortName' in info:
                return info['shortName']
        except:
            pass
        
        return symbol  # Return symbol if name not found
    
    def get_all_tracked_stocks(self):
        """Get all stocks we're tracking across all categories"""
        all_stocks = []
        for category, stocks in self.STOCK_CATEGORIES.items():
            all_stocks.extend(stocks)
        # Remove duplicates while preserving order
        seen = set()
        unique_stocks = []
        for stock in all_stocks:
            if stock not in seen:
                seen.add(stock)
                unique_stocks.append(stock)
        return unique_stocks
    
    def get_stocks_by_category(self, category):
        """Get stocks for a specific category"""
        return self.STOCK_CATEGORIES.get(category, [])
    
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
        """Get overall market statistics and indices"""
        try:
            indices = {
                '^GSPC': 'S&P 500',
                '^DJI': 'Dow Jones',
                '^IXIC': 'NASDAQ',
                '^VIX': 'VIX (Fear Index)',
                '^TNX': '10-Year Treasury'
            }
            
            overview = {}
            for symbol, name in indices.items():
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d")
                if not hist.empty:
                    current = hist['Close'].iloc[-1]
                    previous = hist['Close'].iloc[-2] if len(hist) > 1 else current
                    change = current - previous
                    change_pct = (change / previous) * 100
                    
                    overview[name] = {
                        'symbol': symbol,
                        'value': float(current),
                        'change': float(change),
                        'change_percent': float(change_pct)
                    }
            
            # Fear & Greed Index based on VIX
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
            
            return overview
            
        except Exception as e:
            logger.error(f"Error getting market overview: {e}")
            return {}
    
    def get_stock_fundamentals(self, symbol):
        """Get fundamental analysis data for a stock"""
        try:
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
            
            return fundamentals
            
        except Exception as e:
            logger.error(f"Error getting fundamentals for {symbol}: {e}")
            return None
    
    def get_stock_news(self, symbol):
        """Get recent news for a stock"""
        try:
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
            
            return news_items
        except Exception as e:
            logger.error(f"Error getting news for {symbol}: {e}")
            return []
    
    def get_realtime_metrics(self, symbol):
        """Get real-time metrics for a stock"""
        try:
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
            
            return metrics
        except Exception as e:
            logger.error(f"Error getting real-time metrics for {symbol}: {e}")
            return None
    
    def get_stock_data(self, symbol, period="3mo"):
        """Get daily stock data for a symbol using Yahoo Finance."""
        if self.db:
            db_data = self.db.get_historical_data(symbol, days=100)
            if not db_data.empty and len(db_data) > 0:
                latest_date = db_data.index[0]
                if (datetime.now() - pd.to_datetime(latest_date)).days < 1:
                    logger.info(f"Using recent cached data for {symbol}")
                    return db_data
        
        try:
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
            
            if self.db and not df.empty:
                self.db.store_historical_data(symbol, df)
                logger.info(f"Stored {len(df)} days of data for {symbol} in database")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            if self.db:
                cached_data = self.db.get_historical_data(symbol, days=100)
                if not cached_data.empty:
                    logger.info(f"Returning cached data for {symbol}")
                    return cached_data
            return pd.DataFrame()
    
    def get_quote(self, symbol):
        """Get real-time quote for a symbol using Yahoo Finance."""
        if self.db:
            cached_quotes = self.db.get_latest_quotes([symbol])
            if cached_quotes:
                quote = cached_quotes[0]
                quote_time = datetime.fromisoformat(quote['timestamp'])
                if (datetime.now() - quote_time).seconds < 60:
                    logger.info(f"Using cached quote for {symbol}")
                    result = {
                        'symbol': quote['symbol'],
                        'price': quote['price'],
                        'change': quote['change'],
                        'change_percent': quote['change_percent'],
                        'volume': quote['volume'],
                        'last_updated': quote['timestamp'],
                        'from_cache': True
                    }
                    # Add company name
                    result['name'] = self.get_company_name(symbol)
                    return result
        
        try:
            logger.info(f"Fetching real-time quote for {symbol} from Yahoo Finance")
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            current_price = info.get('regularMarketPrice') or info.get('currentPrice') or info.get('price')
            prev_close = info.get('regularMarketPreviousClose') or info.get('previousClose')
            volume = info.get('regularMarketVolume') or info.get('volume') or 0
            
            if current_price is None:
                hist = ticker.history(period="5d")
                if not hist.empty:
                    current_price = float(hist['Close'].iloc[-1])
                    volume = int(hist['Volume'].iloc[-1])
                    if len(hist) > 1:
                        prev_close = float(hist['Close'].iloc[-2])
                    else:
                        prev_close = current_price
                else:
                    logger.error(f"Could not get price for {symbol}")
                    return None
            
            if prev_close:
                change = current_price - prev_close
                change_percent = f"{(change / prev_close * 100):.2f}%"
            else:
                change = 0
                change_percent = "0.00%"
            
            result = {
                'symbol': symbol.upper(),
                'name': self.get_company_name(symbol),
                'price': float(current_price),
                'change': float(change),
                'change_percent': change_percent,
                'volume': int(volume),
                'last_updated': datetime.now().isoformat(),
                'from_cache': False
            }
            
            if self.db:
                self.db.store_quote(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {str(e)}")
            return self._get_quote_from_historical(symbol)
    
    def _get_quote_from_historical(self, symbol):
        """Get latest price from historical data when real-time fails"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            
            if not hist.empty:
                latest = hist.iloc[-1]
                prev = hist.iloc[-2] if len(hist) > 1 else latest
                
                change = latest['Close'] - prev['Close']
                change_percent = f"{(change / prev['Close'] * 100):.2f}%"
                
                return {
                    'symbol': symbol.upper(),
                    'name': self.get_company_name(symbol),
                    'price': float(latest['Close']),
                    'change': float(change),
                    'change_percent': change_percent,
                    'volume': int(latest['Volume']),
                    'last_updated': hist.index[-1].isoformat(),
                    'from_cache': True
                }
        except:
            pass
        
        if self.db:
            historical = self.db.get_historical_data(symbol, days=2)
            if not historical.empty:
                latest = historical.iloc[0]
                previous = historical.iloc[1] if len(historical) > 1 else None
                
                if previous is not None:
                    change = latest['close'] - previous['close']
                    change_percent = f"{(change / previous['close'] * 100):.2f}%"
                else:
                    change = 0
                    change_percent = "0.00%"
                
                return {
                    'symbol': symbol,
                    'name': self.get_company_name(symbol),
                    'price': float(latest['close']),
                    'change': float(change),
                    'change_percent': change_percent,
                    'volume': int(latest['volume']),
                    'last_updated': historical.index[0].isoformat(),
                    'from_cache': True
                }
        
        logger.error(f"Unable to get any data for {symbol}")
        return None
    
    def get_top_stocks(self):
        """Return top stocks from multiple categories"""
        top_stocks = []
        # Get a mix from different categories
        categories_to_include = ['mega_tech', 'finance', 'healthcare', 'etfs', 'meme_stocks']
        for category in categories_to_include:
            stocks = self.STOCK_CATEGORIES.get(category, [])
            top_stocks.extend(stocks[:3])  # Top 3 from each category
        
        return top_stocks[:15]  # Return top 15
    
    def get_multiple_quotes(self, symbols):
        """Get quotes for multiple symbols efficiently."""
        quotes = []
        
        try:
            logger.info(f"Fetching batch quotes for {len(symbols)} stocks")
            tickers_str = ' '.join(symbols)
            tickers = yf.Tickers(tickers_str)
            
            for symbol in symbols:
                try:
                    ticker = tickers.tickers[symbol.upper()]
                    info = ticker.info
                    
                    current_price = info.get('regularMarketPrice') or info.get('currentPrice')
                    prev_close = info.get('regularMarketPreviousClose') or info.get('previousClose')
                    volume = info.get('regularMarketVolume') or info.get('volume', 0)
                    
                    if current_price and prev_close:
                        change = current_price - prev_close
                        change_percent = f"{(change / prev_close * 100):.2f}%"
                        
                        quote = {
                            'symbol': symbol.upper(),
                            'name': self.get_company_name(symbol),
                            'price': float(current_price),
                            'change': float(change),
                            'change_percent': change_percent,
                            'volume': int(volume),
                            'last_updated': datetime.now().isoformat(),
                            'from_cache': False
                        }
                        
                        quotes.append(quote)
                        
                        if self.db:
                            self.db.store_quote(quote)
                except Exception as e:
                    logger.error(f"Error getting quote for {symbol}: {e}")
                    quote = self.get_quote(symbol)
                    if quote:
                        quotes.append(quote)
            
        except Exception as e:
            logger.error(f"Batch quote fetch failed: {e}")
            for symbol in symbols:
                quote = self.get_quote(symbol)
                if quote:
                    quotes.append(quote)
        
        logger.info(f"Retrieved {len(quotes)} quotes")
        return quotes
    
    def get_stocks_for_analysis(self, symbols=None):
        """Get stock data prepared for technical analysis."""
        if symbols is None:
            symbols = self.get_all_tracked_stocks()[:30]  # Limit to 30 for performance
        
        stocks_data = {}
        
        try:
            logger.info(f"Downloading historical data for {len(symbols)} stocks...")
            
            # Process in batches to avoid overwhelming the API
            batch_size = 10
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i+batch_size]
                tickers_str = ' '.join(batch)
                
                try:
                    data = yf.download(tickers_str, period="3mo", group_by='ticker', 
                                     auto_adjust=True, prepost=True, threads=True)
                    
                    for symbol in batch:
                        try:
                            if len(batch) == 1:
                                df = data
                            else:
                                df = data[symbol]
                            
                            if not df.empty and len(df) >= 30:
                                df = df.rename(columns={
                                    'Open': 'open',
                                    'High': 'high',
                                    'Low': 'low',
                                    'Close': 'close',
                                    'Volume': 'volume'
                                })
                                
                                df = df[['open', 'high', 'low', 'close', 'volume']]
                                df = df.dropna()
                                df = df.sort_index(ascending=True)
                                
                                stocks_data[symbol] = df
                                logger.info(f"Loaded {len(df)} days of data for {symbol}")
                                
                                if self.db:
                                    db_df = df.sort_index(ascending=False)
                                    self.db.store_historical_data(symbol, db_df)
                                    
                        except Exception as e:
                            logger.error(f"Error processing {symbol}: {e}")
                
                except Exception as e:
                    logger.error(f"Batch download failed for {batch}: {e}")
                    
        except Exception as e:
            logger.error(f"Analysis data fetch failed: {e}")
        
        logger.info(f"Successfully loaded data for {len(stocks_data)} stocks")
        return stocks_data