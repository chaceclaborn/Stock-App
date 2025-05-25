# src/analysis/market_analyzer.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

logger = logging.getLogger(__name__)

class MarketAnalyzer:
    """Comprehensive market analysis including sectors, correlations, and breadth"""
    
    def __init__(self):
        # Major market indices
        self.indices = {
            '^GSPC': 'S&P 500',
            '^DJI': 'Dow Jones',
            '^IXIC': 'NASDAQ',
            '^RUT': 'Russell 2000',
            '^VIX': 'VIX',
            '^TNX': '10-Year Treasury',
            'DX-Y.NYB': 'US Dollar Index',
            'GC=F': 'Gold',
            'CL=F': 'Crude Oil',
            'BTC-USD': 'Bitcoin'
        }
        
        # Sector ETFs
        self.sectors = {
            'XLK': 'Technology',
            'XLF': 'Financials',
            'XLV': 'Healthcare',
            'XLE': 'Energy',
            'XLI': 'Industrials',
            'XLY': 'Consumer Discretionary',
            'XLP': 'Consumer Staples',
            'XLB': 'Materials',
            'XLRE': 'Real Estate',
            'XLU': 'Utilities',
            'XLC': 'Communication Services'
        }
        
        # Market regimes
        self.market_regimes = {
            'bull': {'vix': (0, 20), 'trend': 'up', 'breadth': (0.6, 1.0)},
            'bear': {'vix': (30, 100), 'trend': 'down', 'breadth': (0, 0.4)},
            'neutral': {'vix': (20, 30), 'trend': 'sideways', 'breadth': (0.4, 0.6)},
            'volatile': {'vix': (25, 100), 'trend': 'any', 'breadth': 'any'}
        }
        
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    def get_market_overview(self, include_international=False):
        """Get comprehensive market overview"""
        try:
            overview = {
                'indices': {},
                'sectors': {},
                'market_breadth': {},
                'regime': None,
                'risk_indicators': {},
                'correlations': {},
                'timestamp': datetime.now().isoformat()
            }
            
            # Fetch index data
            logger.info("Fetching market indices...")
            index_data = self._fetch_multiple_quotes(list(self.indices.keys()))
            
            for symbol, name in self.indices.items():
                if symbol in index_data:
                    overview['indices'][name] = index_data[symbol]
            
            # Fetch sector data
            logger.info("Fetching sector performance...")
            sector_data = self._fetch_multiple_quotes(list(self.sectors.keys()))
            
            for symbol, name in self.sectors.items():
                if symbol in sector_data:
                    overview['sectors'][name] = sector_data[symbol]
            
            # Calculate market breadth
            overview['market_breadth'] = self.calculate_market_breadth()
            
            # Determine market regime
            overview['regime'] = self.determine_market_regime(overview)
            
            # Calculate risk indicators
            overview['risk_indicators'] = self.calculate_risk_indicators(overview)
            
            # Calculate key correlations
            overview['correlations'] = self.calculate_correlations()
            
            # Add international markets if requested
            if include_international:
                overview['international'] = self.get_international_markets()
            
            return overview
            
        except Exception as e:
            logger.error(f"Error getting market overview: {e}")
            return None
    
    def _fetch_multiple_quotes(self, symbols):
        """Fetch quotes for multiple symbols efficiently"""
        results = {}
        
        try:
            # Use ThreadPoolExecutor for parallel fetching
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_symbol = {
                    executor.submit(self._fetch_single_quote, symbol): symbol 
                    for symbol in symbols
                }
                
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        data = future.result()
                        if data:
                            results[symbol] = data
                    except Exception as e:
                        logger.error(f"Error fetching {symbol}: {e}")
        
        except Exception as e:
            logger.error(f"Error in parallel fetching: {e}")
        
        return results
    
    def _fetch_single_quote(self, symbol):
        """Fetch quote for a single symbol"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            
            if hist.empty:
                return None
            
            current = hist['Close'].iloc[-1]
            previous = hist['Close'].iloc[-2] if len(hist) > 1 else current
            
            change = current - previous
            change_pct = (change / previous) * 100 if previous != 0 else 0
            
            # Get additional data
            info = ticker.info
            
            return {
                'symbol': symbol,
                'price': float(current),
                'change': float(change),
                'change_percent': float(change_pct),
                'volume': int(hist['Volume'].iloc[-1]),
                'day_high': float(hist['High'].iloc[-1]),
                'day_low': float(hist['Low'].iloc[-1]),
                'avg_volume': info.get('averageVolume', 0),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0)
            }
            
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            return None
    
    def calculate_market_breadth(self):
        """Calculate market breadth indicators"""
        try:
            # In a real implementation, this would fetch advance/decline data
            # For now, we'll simulate based on sector performance
            
            breadth = {
                'advance_decline_ratio': 0,
                'new_highs_lows': {'highs': 0, 'lows': 0},
                'percent_above_ma': {'50_day': 0, '200_day': 0},
                'mcclellan_oscillator': 0,
                'arms_index': 0
            }
            
            # Fetch S&P 500 components (top 50 for performance)
            sp500_symbols = self._get_sp500_symbols()[:50]
            
            if sp500_symbols:
                quotes = self._fetch_multiple_quotes(sp500_symbols)
                
                advances = sum(1 for q in quotes.values() if q['change'] > 0)
                declines = sum(1 for q in quotes.values() if q['change'] < 0)
                
                breadth['advance_decline_ratio'] = advances / max(declines, 1)
                
                # Calculate percentage above moving averages
                above_50ma = 0
                above_200ma = 0
                
                for symbol in sp500_symbols[:20]:  # Sample for performance
                    try:
                        ticker = yf.Ticker(symbol)
                        hist = ticker.history(period="1y")
                        if len(hist) >= 200:
                            ma50 = hist['Close'].rolling(50).mean().iloc[-1]
                            ma200 = hist['Close'].rolling(200).mean().iloc[-1]
                            current = hist['Close'].iloc[-1]
                            
                            if current > ma50:
                                above_50ma += 1
                            if current > ma200:
                                above_200ma += 1
                    except:
                        pass
                
                breadth['percent_above_ma']['50_day'] = (above_50ma / 20) * 100
                breadth['percent_above_ma']['200_day'] = (above_200ma / 20) * 100
            
            return breadth
            
        except Exception as e:
            logger.error(f"Error calculating market breadth: {e}")
            return {}
    
    def determine_market_regime(self, overview_data):
        """Determine current market regime"""
        try:
            vix = overview_data['indices'].get('VIX', {}).get('price', 20)
            sp500_change = overview_data['indices'].get('S&P 500', {}).get('change_percent', 0)
            breadth = overview_data['market_breadth'].get('advance_decline_ratio', 1)
            
            # Trend determination
            sp500_ticker = yf.Ticker('^GSPC')
            hist = sp500_ticker.history(period="3mo")
            if not hist.empty:
                ma50 = hist['Close'].rolling(50).mean().iloc[-1]
                current = hist['Close'].iloc[-1]
                trend = 'up' if current > ma50 else 'down'
            else:
                trend = 'sideways'
            
            # Determine regime
            regime = {
                'type': 'neutral',
                'strength': 'moderate',
                'characteristics': [],
                'risk_level': 'medium'
            }
            
            # VIX-based regime
            if vix < 15:
                regime['characteristics'].append('Low volatility')
                regime['risk_level'] = 'low'
            elif vix > 30:
                regime['characteristics'].append('High volatility')
                regime['risk_level'] = 'high'
                regime['type'] = 'volatile'
            
            # Trend-based regime
            if trend == 'up' and breadth > 1.5:
                regime['type'] = 'bull'
                regime['characteristics'].append('Strong uptrend')
                regime['strength'] = 'strong'
            elif trend == 'down' and breadth < 0.7:
                regime['type'] = 'bear'
                regime['characteristics'].append('Downtrend')
                regime['strength'] = 'strong'
            
            # Breadth confirmation
            if breadth > 2:
                regime['characteristics'].append('Broad market participation')
            elif breadth < 0.5:
                regime['characteristics'].append('Narrow market weakness')
            
            return regime
            
        except Exception as e:
            logger.error(f"Error determining market regime: {e}")
            return {'type': 'unknown', 'strength': 'unknown', 'risk_level': 'unknown'}
    
    def calculate_risk_indicators(self, overview_data):
        """Calculate various risk indicators"""
        try:
            risk = {
                'vix_level': 'normal',
                'put_call_ratio': 0,
                'credit_spreads': 'normal',
                'dollar_strength': 'neutral',
                'yield_curve': 'normal',
                'risk_score': 50  # 0-100 scale
            }
            
            # VIX level
            vix = overview_data['indices'].get('VIX', {}).get('price', 20)
            if vix < 15:
                risk['vix_level'] = 'low'
                risk['risk_score'] -= 10
            elif vix > 25:
                risk['vix_level'] = 'high'
                risk['risk_score'] += 20
            elif vix > 35:
                risk['vix_level'] = 'extreme'
                risk['risk_score'] += 40
            
            # Dollar strength
            dxy = overview_data['indices'].get('US Dollar Index', {}).get('change_percent', 0)
            if dxy > 1:
                risk['dollar_strength'] = 'strong'
                risk['risk_score'] += 5
            elif dxy < -1:
                risk['dollar_strength'] = 'weak'
                risk['risk_score'] += 10
            
            # Yield curve (simplified)
            tsy_10y = overview_data['indices'].get('10-Year Treasury', {}).get('price', 4)
            if tsy_10y < 2:
                risk['yield_curve'] = 'low_rates'
                risk['risk_score'] += 5
            elif tsy_10y > 5:
                risk['yield_curve'] = 'high_rates'
                risk['risk_score'] += 15
            
            # Calculate composite risk score
            risk['risk_score'] = max(0, min(100, risk['risk_score']))
            
            # Risk level interpretation
            if risk['risk_score'] < 30:
                risk['overall_risk'] = 'low'
            elif risk['risk_score'] < 50:
                risk['overall_risk'] = 'moderate'
            elif risk['risk_score'] < 70:
                risk['overall_risk'] = 'elevated'
            else:
                risk['overall_risk'] = 'high'
            
            return risk
            
        except Exception as e:
            logger.error(f"Error calculating risk indicators: {e}")
            return {'risk_score': 50, 'overall_risk': 'unknown'}
    
    def calculate_correlations(self, period='1mo'):
        """Calculate key market correlations"""
        try:
            # Key pairs to analyze
            pairs = [
                ('^GSPC', 'Stocks', '^VIX', 'VIX'),
                ('^GSPC', 'Stocks', '^TNX', 'Bonds'),
                ('^GSPC', 'Stocks', 'GC=F', 'Gold'),
                ('DX-Y.NYB', 'Dollar', 'GC=F', 'Gold'),
                ('^GSPC', 'Stocks', 'BTC-USD', 'Bitcoin')
            ]
            
            correlations = {}
            
            for sym1, name1, sym2, name2 in pairs:
                try:
                    # Fetch data
                    ticker1 = yf.Ticker(sym1)
                    ticker2 = yf.Ticker(sym2)
                    
                    hist1 = ticker1.history(period=period)['Close']
                    hist2 = ticker2.history(period=period)['Close']
                    
                    if len(hist1) > 10 and len(hist2) > 10:
                        # Align data
                        common_dates = hist1.index.intersection(hist2.index)
                        if len(common_dates) > 10:
                            returns1 = hist1[common_dates].pct_change().dropna()
                            returns2 = hist2[common_dates].pct_change().dropna()
                            
                            corr = returns1.corr(returns2)
                            
                            correlations[f"{name1}/{name2}"] = {
                                'correlation': round(corr, 3),
                                'interpretation': self._interpret_correlation(corr),
                                'period': period
                            }
                except Exception as e:
                    logger.error(f"Error calculating correlation for {name1}/{name2}: {e}")
            
            return correlations
            
        except Exception as e:
            logger.error(f"Error calculating correlations: {e}")
            return {}
    
    def _interpret_correlation(self, corr):
        """Interpret correlation value"""
        if corr > 0.7:
            return "Strong positive"
        elif corr > 0.3:
            return "Moderate positive"
        elif corr > -0.3:
            return "Weak/None"
        elif corr > -0.7:
            return "Moderate negative"
        else:
            return "Strong negative"
    
    def get_sector_rotation(self):
        """Analyze sector rotation patterns"""
        try:
            rotation = {
                'current_leaders': [],
                'current_laggards': [],
                'momentum_shifts': [],
                'recommended_sectors': []
            }
            
            # Get sector performance
            sector_data = self._fetch_multiple_quotes(list(self.sectors.keys()))
            
            # Sort by performance
            sector_performance = []
            for symbol, name in self.sectors.items():
                if symbol in sector_data:
                    perf = {
                        'sector': name,
                        'symbol': symbol,
                        'change_1d': sector_data[symbol]['change_percent'],
                        'momentum': 'neutral'
                    }
                    
                    # Get longer-term performance
                    try:
                        ticker = yf.Ticker(symbol)
                        hist = ticker.history(period="3mo")
                        if len(hist) > 20:
                            perf['change_1m'] = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-20]) / 
                                               hist['Close'].iloc[-20]) * 100
                            perf['change_3m'] = ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / 
                                               hist['Close'].iloc[0]) * 100
                            
                            # Determine momentum
                            if perf['change_1m'] > 5 and perf['change_3m'] > 10:
                                perf['momentum'] = 'strong'
                            elif perf['change_1m'] < -5 and perf['change_3m'] < -10:
                                perf['momentum'] = 'weak'
                    except:
                        pass
                    
                    sector_performance.append(perf)
            
            # Sort and identify leaders/laggards
            sector_performance.sort(key=lambda x: x.get('change_1m', 0), reverse=True)
            
            rotation['current_leaders'] = sector_performance[:3]
            rotation['current_laggards'] = sector_performance[-3:]
            
            # Identify momentum shifts
            for sector in sector_performance:
                if sector.get('change_1m', 0) > 0 and sector.get('change_1d', 0) < -2:
                    rotation['momentum_shifts'].append({
                        'sector': sector['sector'],
                        'signal': 'potential_weakness',
                        'reason': 'Negative daily move despite positive monthly trend'
                    })
                elif sector.get('change_1m', 0) < 0 and sector.get('change_1d', 0) > 2:
                    rotation['momentum_shifts'].append({
                        'sector': sector['sector'],
                        'signal': 'potential_strength',
                        'reason': 'Positive daily move despite negative monthly trend'
                    })
            
            # Recommendations based on market regime
            # This would be more sophisticated in production
            rotation['recommended_sectors'] = self._get_sector_recommendations(
                sector_performance, 
                rotation
            )
            
            return rotation
            
        except Exception as e:
            logger.error(f"Error analyzing sector rotation: {e}")
            return {}
    
    def _get_sector_recommendations(self, sector_performance, rotation_data):
        """Get sector recommendations based on current conditions"""
        recommendations = []
        
        # Simple logic - in production this would be much more sophisticated
        for sector in sector_performance:
            if sector['momentum'] == 'strong' and sector['change_1d'] > -1:
                recommendations.append({
                    'sector': sector['sector'],
                    'action': 'overweight',
                    'reason': 'Strong momentum continuing'
                })
            elif sector['momentum'] == 'weak' and sector['change_1d'] < -2:
                recommendations.append({
                    'sector': sector['sector'],
                    'action': 'underweight',
                    'reason': 'Weak momentum accelerating'
                })
        
        return recommendations[:5]  # Top 5 recommendations
    
    def get_international_markets(self):
        """Get international market data"""
        international = {
            '^FTSE': 'FTSE 100 (UK)',
            '^GDAXI': 'DAX (Germany)',
            '^FCHI': 'CAC 40 (France)',
            '^N225': 'Nikkei 225 (Japan)',
            '^HSI': 'Hang Seng (Hong Kong)',
            '000001.SS': 'Shanghai Composite',
            '^BSESN': 'BSE SENSEX (India)'
        }
        
        return self._fetch_multiple_quotes(list(international.keys()))
    
    def _get_sp500_symbols(self):
        """Get S&P 500 symbols"""
        # In production, this would fetch from a reliable source
        # For now, return a sample of major stocks
        return [
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM',
            'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'BAC', 'ADBE', 'NFLX',
            'CMCSA', 'VZ', 'INTC', 'T', 'KO', 'PFE', 'MRK', 'PEP', 'CSCO',
            'XOM', 'CVX', 'WMT', 'ABT', 'TMO', 'CRM', 'ABBV', 'NKE', 'ACN',
            'COST', 'LLY', 'MCD', 'NEE', 'TXN', 'DHR', 'AVGO', 'MDT', 'HON',
            'UNP', 'QCOM', 'LOW', 'LIN', 'AMT'
        ]
    
    def get_market_sentiment_score(self):
        """Calculate overall market sentiment score"""
        try:
            sentiment_factors = {
                'vix': {'weight': 0.25, 'score': 0},
                'breadth': {'weight': 0.20, 'score': 0},
                'momentum': {'weight': 0.20, 'score': 0},
                'volume': {'weight': 0.15, 'score': 0},
                'sectors': {'weight': 0.20, 'score': 0}
            }
            
            # Get market data
            overview = self.get_market_overview()
            
            # VIX sentiment (inverted)
            vix = overview['indices'].get('VIX', {}).get('price', 20)
            if vix < 15:
                sentiment_factors['vix']['score'] = 80
            elif vix < 20:
                sentiment_factors['vix']['score'] = 60
            elif vix < 30:
                sentiment_factors['vix']['score'] = 40
            else:
                sentiment_factors['vix']['score'] = 20
            
            # Breadth sentiment
            breadth_ratio = overview['market_breadth'].get('advance_decline_ratio', 1)
            sentiment_factors['breadth']['score'] = min(100, breadth_ratio * 40)
            
            # Momentum sentiment
            sp500_change = overview['indices'].get('S&P 500', {}).get('change_percent', 0)
            if sp500_change > 1:
                sentiment_factors['momentum']['score'] = 80
            elif sp500_change > 0:
                sentiment_factors['momentum']['score'] = 60
            elif sp500_change > -1:
                sentiment_factors['momentum']['score'] = 40
            else:
                sentiment_factors['momentum']['score'] = 20
            
            # Sector sentiment (based on positive sectors)
            positive_sectors = sum(1 for s in overview['sectors'].values() 
                                 if s.get('change_percent', 0) > 0)
            sentiment_factors['sectors']['score'] = (positive_sectors / 11) * 100
            
            # Calculate weighted sentiment score
            total_score = sum(f['weight'] * f['score'] 
                            for f in sentiment_factors.values())
            
            return {
                'overall_score': round(total_score, 1),
                'factors': sentiment_factors,
                'interpretation': self._interpret_sentiment_score(total_score)
            }
            
        except Exception as e:
            logger.error(f"Error calculating market sentiment: {e}")
            return {'overall_score': 50, 'interpretation': 'Neutral'}
    
    def _interpret_sentiment_score(self, score):
        """Interpret sentiment score"""
        if score >= 80:
            return "Extreme Greed"
        elif score >= 65:
            return "Greed"
        elif score >= 50:
            return "Neutral"
        elif score >= 35:
            return "Fear"
        else:
            return "Extreme Fear"