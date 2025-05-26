# src/web/services/sentiment_service.py
"""
Market sentiment analysis service - moved from app.py
"""
import re
import numpy as np
import pandas as pd
import yfinance as yf
from textblob import TextBlob
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class MarketSentimentAnalyzer:
    """Analyze market sentiment from various sources"""
    
    def __init__(self):
        self.sentiment_weights = {
            'news': 0.3,
            'technical': 0.4,
            'volume': 0.2,
            'social': 0.1
        }
        self.sentiment_cache = {}
        self.sentiment_last_update = {}
        self.cache_duration = 300  # 5 minutes
    
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
        # Check cache
        now = datetime.now()
        if symbol in self.sentiment_cache and symbol in self.sentiment_last_update:
            if (now - self.sentiment_last_update[symbol]).seconds < self.cache_duration:
                return self.sentiment_cache[symbol]
        
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
            self.sentiment_cache[symbol] = sentiment
            self.sentiment_last_update[symbol] = now
            
            return sentiment
            
        except Exception as e:
            logger.error(f"Error getting market sentiment for {symbol}: {e}")
            return {
                'overall': 0.5,
                'components': {},
                'description': 'Unknown',
                'confidence': 'Low'
            }