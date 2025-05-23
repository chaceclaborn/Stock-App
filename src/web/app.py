# src/web/app.py
import os
import sys
import logging
import json
from flask import Flask, render_template, jsonify, request
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import re
from textblob import TextBlob
import yfinance as yf

# Add parent directory to path so we can import our modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from data.database import StockDatabase
    from data.fetcher import StockDataFetcher
    from models.predictor import DayTradePredictor
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create the Flask app
app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'))

# Set the database path
db_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
os.makedirs(db_dir, exist_ok=True)
db_path = os.path.join(db_dir, 'stock_data.db')

# Create thread-safe database
db = StockDatabase(db_path=db_path)

# Create fetcher instance with database
fetcher = StockDataFetcher(db=db)

# Create predictor instance
predictor = DayTradePredictor()

# Last update time tracking
last_update_time = None
update_interval = 30  # seconds

# Cache for various data
market_overview_cache = None
market_overview_last_update = None
predictions_cache = None
predictions_last_update = None
predictions_interval = 300  # 5 minutes
sentiment_cache = {}
sentiment_last_update = {}

class MarketSentimentAnalyzer:
    """Analyze market sentiment from various sources"""
    
    def __init__(self):
        self.sentiment_weights = {
            'news': 0.3,
            'technical': 0.4,
            'volume': 0.2,
            'social': 0.1
        }
    
    def analyze_news_sentiment(self, news_items):
        """Analyze sentiment from news headlines"""
        if not news_items:
            return 0.5  # Neutral
        
        sentiments = []
        for item in news_items:
            try:
                # Clean the text
                text = item.get('title', '') + ' ' + item.get('summary', '')
                text = re.sub(r'[^a-zA-Z\s]', '', text)
                
                # Analyze sentiment
                blob = TextBlob(text)
                # Convert from [-1, 1] to [0, 1]
                sentiment = (blob.sentiment.polarity + 1) / 2
                sentiments.append(sentiment)
            except:
                continue
        
        return np.mean(sentiments) if sentiments else 0.5
    
    def analyze_technical_sentiment(self, symbol, data):
        """Analyze sentiment from technical indicators"""
        try:
            if data.empty or len(data) < 20:
                return 0.5
            
            sentiment_score = 0
            factors = 0
            
            # Price momentum
            returns_5d = (data['close'].iloc[-1] - data['close'].iloc[-6]) / data['close'].iloc[-6]
            returns_20d = (data['close'].iloc[-1] - data['close'].iloc[-20]) / data['close'].iloc[-20]
            
            if returns_5d > 0.02:
                sentiment_score += 0.7
            elif returns_5d < -0.02:
                sentiment_score += 0.3
            else:
                sentiment_score += 0.5
            factors += 1
            
            if returns_20d > 0.05:
                sentiment_score += 0.8
            elif returns_20d < -0.05:
                sentiment_score += 0.2
            else:
                sentiment_score += 0.5
            factors += 1
            
            # Volume analysis
            avg_volume = data['volume'].iloc[-20:].mean()
            recent_volume = data['volume'].iloc[-5:].mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
            
            if volume_ratio > 1.5 and returns_5d > 0:
                sentiment_score += 0.8
            elif volume_ratio > 1.5 and returns_5d < 0:
                sentiment_score += 0.2
            else:
                sentiment_score += 0.5
            factors += 1
            
            # Moving average position
            ma_20 = data['close'].iloc[-20:].mean()
            current_price = data['close'].iloc[-1]
            
            if current_price > ma_20 * 1.02:
                sentiment_score += 0.7
            elif current_price < ma_20 * 0.98:
                sentiment_score += 0.3
            else:
                sentiment_score += 0.5
            factors += 1
            
            return sentiment_score / factors if factors > 0 else 0.5
            
        except Exception as e:
            logger.error(f"Error in technical sentiment analysis: {e}")
            return 0.5
    
    def analyze_volume_sentiment(self, volume_data):
        """Analyze sentiment from volume patterns"""
        try:
            if len(volume_data) < 10:
                return 0.5
            
            # Calculate volume trend
            recent_avg = np.mean(volume_data[-5:])
            historical_avg = np.mean(volume_data[-20:-5])
            
            if historical_avg == 0:
                return 0.5
            
            volume_ratio = recent_avg / historical_avg
            
            # Map to sentiment score
            if volume_ratio > 2.0:
                return 0.9
            elif volume_ratio > 1.5:
                return 0.7
            elif volume_ratio > 1.2:
                return 0.6
            elif volume_ratio < 0.5:
                return 0.1
            elif volume_ratio < 0.8:
                return 0.3
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error in volume sentiment analysis: {e}")
            return 0.5
    
    def calculate_fear_greed_index(self, market_data):
        """Calculate a fear and greed index based on multiple factors"""
        try:
            factors = []
            weights = []
            
            # VIX level (inverted)
            if 'VIX (Fear Index)' in market_data:
                vix = market_data['VIX (Fear Index)']['value']
                if vix < 12:
                    factors.append(0.9)  # Extreme greed
                elif vix < 16:
                    factors.append(0.7)  # Greed
                elif vix < 20:
                    factors.append(0.5)  # Neutral
                elif vix < 28:
                    factors.append(0.3)  # Fear
                else:
                    factors.append(0.1)  # Extreme fear
                weights.append(0.25)
            
            # Market momentum (S&P 500)
            if 'S&P 500' in market_data:
                sp_change = market_data['S&P 500']['change_percent']
                if sp_change > 1:
                    factors.append(0.8)
                elif sp_change > 0:
                    factors.append(0.6)
                elif sp_change > -1:
                    factors.append(0.4)
                else:
                    factors.append(0.2)
                weights.append(0.25)
            
            # Market breadth (simulated)
            # In production, this would use advance/decline data
            breadth = np.random.normal(0.5, 0.1)
            breadth = max(0, min(1, breadth))
            factors.append(breadth)
            weights.append(0.25)
            
            # Put/Call ratio (simulated)
            # In production, this would use real options data
            put_call = np.random.normal(0.5, 0.1)
            put_call = max(0, min(1, put_call))
            factors.append(1 - put_call)  # Invert since high P/C = fear
            weights.append(0.25)
            
            # Calculate weighted average
            if factors and weights:
                fear_greed = np.average(factors, weights=weights) * 100
                return int(fear_greed)
            else:
                return 50
                
        except Exception as e:
            logger.error(f"Error calculating fear/greed index: {e}")
            return 50
    
    def get_market_sentiment(self, symbol, include_social=True):
        """Get comprehensive market sentiment for a symbol"""
        global sentiment_cache, sentiment_last_update
        
        # Check cache
        now = datetime.now()
        if symbol in sentiment_cache and symbol in sentiment_last_update:
            if (now - sentiment_last_update[symbol]).seconds < 300:  # 5 minute cache
                return sentiment_cache[symbol]
        
        try:
            sentiment = {
                'overall': 0.5,
                'components': {},
                'description': 'Neutral',
                'confidence': 'Medium'
            }
            
            # Get stock data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1mo")
            
            if not hist.empty:
                # Technical sentiment
                tech_sentiment = self.analyze_technical_sentiment(symbol, hist.rename(columns={
                    'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
                }))
                sentiment['components']['technical'] = tech_sentiment
                
                # Volume sentiment
                volume_sentiment = self.analyze_volume_sentiment(hist['Volume'].values)
                sentiment['components']['volume'] = volume_sentiment
            
            # News sentiment
            news = ticker.news[:10] if hasattr(ticker, 'news') else []
            if news:
                news_sentiment = self.analyze_news_sentiment(news)
                sentiment['components']['news'] = news_sentiment
            
            # Social sentiment (simulated for now)
            if include_social:
                # In production, this would use real social media APIs
                social_sentiment = np.random.normal(0.5, 0.15)
                social_sentiment = max(0, min(1, social_sentiment))
                sentiment['components']['social'] = social_sentiment
            
            # Calculate overall sentiment
            total_weight = 0
            weighted_sum = 0
            
            for component, score in sentiment['components'].items():
                weight = self.sentiment_weights.get(component, 0.1)
                weighted_sum += score * weight
                total_weight += weight
            
            if total_weight > 0:
                sentiment['overall'] = weighted_sum / total_weight
            
            # Determine description
            overall = sentiment['overall']
            if overall >= 0.8:
                sentiment['description'] = 'Very Bullish'
                sentiment['confidence'] = 'High'
            elif overall >= 0.65:
                sentiment['description'] = 'Bullish'
                sentiment['confidence'] = 'High'
            elif overall >= 0.55:
                sentiment['description'] = 'Slightly Bullish'
                sentiment['confidence'] = 'Medium'
            elif overall >= 0.45:
                sentiment['description'] = 'Neutral'
                sentiment['confidence'] = 'Medium'
            elif overall >= 0.35:
                sentiment['description'] = 'Slightly Bearish'
                sentiment['confidence'] = 'Medium'
            elif overall >= 0.2:
                sentiment['description'] = 'Bearish'
                sentiment['confidence'] = 'High'
            else:
                sentiment['description'] = 'Very Bearish'
                sentiment['confidence'] = 'High'
            
            # Cache the result
            sentiment_cache[symbol] = sentiment
            sentiment_last_update[symbol] = now
            
            return sentiment
            
        except Exception as e:
            logger.error(f"Error getting market sentiment for {symbol}: {e}")
            return {
                'overall': 0.5,
                'components': {},
                'description': 'Unknown',
                'confidence': 'Low'
            }

# Initialize sentiment analyzer
sentiment_analyzer = MarketSentimentAnalyzer()

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/api/stocks')
def get_stocks():
    """API endpoint to get current stock data with status info."""
    global last_update_time
    
    # Get parameters
    force_refresh = request.args.get('force', 'false').lower() == 'true'
    category = request.args.get('category', 'all')
    sort_by = request.args.get('sort', 'market_cap')  # market_cap, change, volume, sentiment
    
    try:
        now = datetime.now()
        
        # Only update if it's been more than update_interval seconds or forced
        if (last_update_time is None or
            (now - last_update_time).total_seconds() >= update_interval or
            force_refresh):
            
            # Get stocks based on category
            if category == 'all':
                symbols = fetcher.get_all_tracked_stocks()[:50]  # Limit to 50
            elif category == 'trending':
                # Get stocks with high volume or big moves
                symbols = fetcher.get_top_stocks()
            else:
                symbols = fetcher.get_stocks_by_category(category)
            
            # Get quotes for the stocks
            quotes = fetcher.get_multiple_quotes(symbols)
            
            # Add sentiment data for each stock
            for quote in quotes:
                sentiment = sentiment_analyzer.get_market_sentiment(quote['symbol'], include_social=False)
                quote['sentiment'] = sentiment['overall']
                quote['sentiment_description'] = sentiment['description']
            
            # Sort based on criteria
            if sort_by == 'change':
                quotes.sort(key=lambda x: x.get('change', 0), reverse=True)
            elif sort_by == 'volume':
                quotes.sort(key=lambda x: x.get('volume', 0), reverse=True)
            elif sort_by == 'sentiment':
                quotes.sort(key=lambda x: x.get('sentiment', 0.5), reverse=True)
            
            # Update last update time
            last_update_time = now
            
            # Get stock categories for UI
            categories = list(fetcher.STOCK_CATEGORIES.keys())
            
            return jsonify({
                'stocks': quotes,
                'categories': categories,
                'last_updated': now.strftime('%Y-%m-%d %H:%M:%S'),
                'next_update': (now + timedelta(seconds=update_interval)).strftime('%Y-%m-%d %H:%M:%S'),
                'status': 'ok',
                'from_cache': False
            })
        else:
            # Return cached result
            cached_quotes = fetcher.get_multiple_quotes(fetcher.get_top_stocks())
            
            # Add cached sentiment
            for quote in cached_quotes:
                if quote['symbol'] in sentiment_cache:
                    quote['sentiment'] = sentiment_cache[quote['symbol']]['overall']
                    quote['sentiment_description'] = sentiment_cache[quote['symbol']]['description']
            
            return jsonify({
                'stocks': cached_quotes,
                'categories': list(fetcher.STOCK_CATEGORIES.keys()),
                'last_updated': last_update_time.strftime('%Y-%m-%d %H:%M:%S'),
                'next_update': (last_update_time + timedelta(seconds=update_interval)).strftime('%Y-%m-%d %H:%M:%S'),
                'status': 'cached',
                'from_cache': True
            })
            
    except Exception as e:
        logger.error(f"Error getting stock data: {str(e)}")
        
        return jsonify({
            'error': f'Error retrieving stock data: {str(e)}',
            'stocks': [],
            'categories': list(fetcher.STOCK_CATEGORIES.keys()),
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'error'
        }), 500

@app.route('/api/market-overview')
def get_market_overview():
    """API endpoint for market overview including indices and sentiment"""
    global market_overview_cache, market_overview_last_update
    
    try:
        now = datetime.now()
        
        # Cache for 1 minute
        if (market_overview_cache is None or 
            market_overview_last_update is None or
            (now - market_overview_last_update).total_seconds() >= 60):
            
            logger.info("Fetching market overview...")
            overview = fetcher.get_market_overview()
            
            # Calculate enhanced fear & greed index
            fear_greed = sentiment_analyzer.calculate_fear_greed_index(overview)
            overview['fear_greed_index'] = fear_greed
            
            # Add market sentiment analysis
            market_sentiment = {
                'spy': sentiment_analyzer.get_market_sentiment('SPY', include_social=False),
                'qqq': sentiment_analyzer.get_market_sentiment('QQQ', include_social=False),
                'vix': sentiment_analyzer.get_market_sentiment('^VIX', include_social=False)
            }
            
            market_overview_cache = {
                'indices': overview,
                'market_sentiment': market_sentiment,
                'timestamp': now.isoformat(),
                'status': 'ok'
            }
            market_overview_last_update = now
        
        return jsonify(market_overview_cache)
        
    except Exception as e:
        logger.error(f"Error getting market overview: {str(e)}")
        return jsonify({
            'error': f'Error getting market overview: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/api/stock/<symbol>')
def get_stock_detail(symbol):
    """API endpoint to get detailed data for a specific stock."""
    try:
        # Get period parameter
        period = request.args.get('period', '1m')
        
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
        
        # Get historical data
        df = fetcher.get_stock_data(symbol.upper(), period=yf_period)
        
        # Get the latest quote
        quote = fetcher.get_quote(symbol.upper())
        
        # Get company name
        name = fetcher.get_company_name(symbol.upper())
        
        # Get fundamentals
        fundamentals = fetcher.get_stock_fundamentals(symbol.upper())
        
        # Get real-time metrics
        realtime = fetcher.get_realtime_metrics(symbol.upper())
        
        # Get recent news
        news = fetcher.get_stock_news(symbol.upper())
        
        # Get sentiment analysis
        sentiment = sentiment_analyzer.get_market_sentiment(symbol.upper())
        
        # Prepare daily data for charting
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
        
        # Return combined data
        return jsonify({
            'symbol': symbol.upper(),
            'name': name,
            'quote': quote,
            'daily_data': daily_data,
            'fundamentals': fundamentals,
            'realtime_metrics': realtime,
            'news': news,
            'sentiment': sentiment,
            'period': period,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'ok'
        })
        
    except Exception as e:
        logger.error(f"Error getting stock detail for {symbol}: {str(e)}")
        return jsonify({
            'error': f'Error retrieving data: {str(e)}'
        }), 500

@app.route('/api/search')
def search_stocks():
    """API endpoint for stock search"""
    try:
        query = request.args.get('q', '')
        if not query:
            return jsonify({'results': []})
        
        results = fetcher.search_stocks(query)
        
        # Get basic info for each result
        search_results = []
        for symbol in results:
            try:
                quote = fetcher.get_quote(symbol)
                if quote:
                    # Add sentiment
                    sentiment = sentiment_analyzer.get_market_sentiment(symbol, include_social=False)
                    
                    search_results.append({
                        'symbol': symbol,
                        'name': quote.get('name', symbol),
                        'price': quote['price'],
                        'change': quote['change'],
                        'change_percent': quote['change_percent'],
                        'sentiment': sentiment['overall'],
                        'sentiment_description': sentiment['description']
                    })
            except:
                pass
        
        return jsonify({'results': search_results})
        
    except Exception as e:
        logger.error(f"Error searching stocks: {str(e)}")
        return jsonify({'results': []})

@app.route('/api/predictions')
def get_predictions():
    """API endpoint to get enhanced trading predictions with AI analysis."""
    global predictions_cache, predictions_last_update
    
    try:
        now = datetime.now()
        force_refresh = request.args.get('force', 'false').lower() == 'true'
        
        # Check if we need to update predictions
        if (predictions_cache is None or 
            predictions_last_update is None or
            (now - predictions_last_update).total_seconds() >= predictions_interval or
            force_refresh):
            
            logger.info("Generating new AI-powered predictions...")
            
            # Get stock data for analysis
            stocks_data = fetcher.get_stocks_for_analysis()
            
            if not stocks_data:
                logger.warning("No stock data available for analysis")
                predictions_cache = {
                    'opportunities': [],
                    'generated_at': now.isoformat(),
                    'status': 'no_data',
                    'message': 'Unable to retrieve sufficient stock data for analysis'
                }
            else:
                logger.info(f"Analyzing {len(stocks_data)} stocks for opportunities")
                
                # Find opportunities
                opportunities_df = predictor.find_opportunities(stocks_data, top_n=20)
                
                if not opportunities_df.empty:
                    # Convert DataFrame to list of dicts
                    opportunities = opportunities_df.to_dict('records')
                    
                    # Add company names and sentiment
                    for opp in opportunities:
                        opp['name'] = fetcher.get_company_name(opp['ticker'])
                        sentiment = sentiment_analyzer.get_market_sentiment(opp['ticker'])
                        opp['sentiment'] = sentiment
                    
                    # Get market context
                    market_overview = fetcher.get_market_overview()
                    
                    # Cache the results
                    predictions_cache = {
                        'opportunities': opportunities,
                        'generated_at': now.isoformat(),
                        'status': 'success',
                        'stocks_analyzed': len(stocks_data),
                        'market_context': {
                            'fear_greed_index': sentiment_analyzer.calculate_fear_greed_index(market_overview),
                            'market_trend': 'bullish' if market_overview.get('S&P 500', {}).get('change', 0) > 0 else 'bearish'
                        }
                    }
                    logger.info(f"Found {len(opportunities)} trading opportunities")
                else:
                    predictions_cache = {
                        'opportunities': [],
                        'generated_at': now.isoformat(),
                        'status': 'no_opportunities',
                        'stocks_analyzed': len(stocks_data),
                        'message': f'Analyzed {len(stocks_data)} stocks but found no strong trading signals'
                    }
                    logger.info("No trading opportunities found")
            
            predictions_last_update = now
            
        return jsonify(predictions_cache)
        
    except Exception as e:
        logger.error(f"Error generating predictions: {str(e)}", exc_info=True)
        return jsonify({
            'error': f'Error generating predictions: {str(e)}',
            'opportunities': [],
            'status': 'error'
        }), 500

@app.route('/api/stock/<symbol>/analysis')
def get_stock_analysis(symbol):
    """API endpoint to get detailed technical analysis with real-time data."""
    try:
        # Get historical data
        df = fetcher.get_stock_data(symbol.upper())
        
        if df.empty:
            return jsonify({
                'error': f'No data available for {symbol}'
            }), 404
        
        # Sort data chronologically for analysis
        df = df.sort_index(ascending=True)
        
        # Analyze the stock
        indicators_df = predictor.analyze_stock(symbol.upper(), df)
        
        if indicators_df is None:
            return jsonify({
                'error': f'Unable to analyze {symbol}'
            }), 500
        
        # Get latest indicators
        latest = indicators_df.iloc[-1].to_dict()
        
        # Get entry/exit suggestions
        suggestions = predictor.get_entry_exit_points(symbol.upper(), indicators_df)
        
        # Get sentiment analysis
        sentiment = sentiment_analyzer.get_market_sentiment(symbol.upper())
        
        # Prepare data for charting (last 60 days for better pattern visibility)
        chart_data = []
        for date, row in indicators_df.tail(60).iterrows():
            chart_point = {
                'date': date.strftime('%Y-%m-%d'),
                'close': float(row['Close']),
                'rsi': float(row['RSI']) if pd.notna(row['RSI']) else None,
                'ma_short': float(row['MA_Short']) if pd.notna(row.get('MA_Short')) else None,
                'ma_long': float(row['MA_Long']) if pd.notna(row.get('MA_Long')) else None,
                'bb_upper': float(row['BB_Upper']) if pd.notna(row.get('BB_Upper')) else None,
                'bb_lower': float(row['BB_Lower']) if pd.notna(row.get('BB_Lower')) else None,
                'macd': float(row['MACD']) if pd.notna(row.get('MACD')) else None,
                'macd_signal': float(row['MACD_Signal']) if pd.notna(row.get('MACD_Signal')) else None,
                'volume': int(row['Volume'])
            }
            chart_data.append(chart_point)
        
        # Score the current opportunity
        score, reasons, signals = predictor.score_opportunity(latest, indicators_df)
        
        # Get company name
        name = fetcher.get_company_name(symbol.upper())
        
        # Get real-time metrics
        realtime = fetcher.get_realtime_metrics(symbol.upper())
        
        return jsonify({
            'symbol': symbol.upper(),
            'name': name,
            'current_indicators': {
                'price': float(latest['Close']),
                'rsi': float(latest['RSI']) if pd.notna(latest.get('RSI')) else None,
                'volatility': float(latest['Volatility']) if pd.notna(latest.get('Volatility')) else None,
                'volume_trend': float(latest['Volume_Trend']) if pd.notna(latest.get('Volume_Trend')) else None,
                'ma_signal': int(latest['Signal']) if pd.notna(latest.get('Signal')) else 0
            },
            'score': score,
            'reasons': reasons,
            'signals': signals,
            'suggestions': suggestions,
            'sentiment': sentiment,
            'realtime_metrics': realtime,
            'chart_data': chart_data,
            'analyzed_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {str(e)}")
        return jsonify({
            'error': f'Error analyzing stock: {str(e)}'
        }), 500

@app.route('/api/stock/<symbol>/sentiment')
def get_stock_sentiment(symbol):
    """API endpoint to get detailed sentiment analysis for a stock"""
    try:
        sentiment = sentiment_analyzer.get_market_sentiment(symbol.upper())
        
        # Get additional context
        ticker = yf.Ticker(symbol.upper())
        info = ticker.info
        
        # Get recent price action for context
        hist = ticker.history(period="1mo")
        if not hist.empty:
            price_change_1w = (hist['Close'].iloc[-1] - hist['Close'].iloc[-6]) / hist['Close'].iloc[-6] * 100
            price_change_1m = (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100
        else:
            price_change_1w = 0
            price_change_1m = 0
        
        return jsonify({
            'symbol': symbol.upper(),
            'name': fetcher.get_company_name(symbol.upper()),
            'sentiment': sentiment,
            'price_context': {
                'current_price': info.get('regularMarketPrice', 0),
                'change_1w': price_change_1w,
                'change_1m': price_change_1m,
                'volume_vs_avg': info.get('regularMarketVolume', 0) / info.get('averageVolume', 1) if info.get('averageVolume', 0) > 0 else 1
            },
            'analyst_data': {
                'recommendation': info.get('recommendationKey', 'none'),
                'target_mean': info.get('targetMeanPrice', 0),
                'number_of_analysts': info.get('numberOfAnalystOpinions', 0)
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting sentiment for {symbol}: {str(e)}")
        return jsonify({
            'error': f'Error getting sentiment: {str(e)}'
        }), 500

@app.errorhandler(404)
def page_not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Server error'}), 500

def run_app(debug=True, port=5000):
    """Run the Flask application."""
    logger.info(f"Starting Enhanced Stock Market Analyzer on port {port}")
    logger.info("Features: Real-time data, Market sentiment analysis, AI predictions")
    logger.info("Using Yahoo Finance for real-time stock data.")
    
    app.run(debug=debug, port=port, threaded=True)

if __name__ == '__main__':
    run_app()