# src/web/services/analysis_service.py
"""
Technical analysis service layer
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class AnalysisService:
    """Service for technical analysis operations"""
    
    def __init__(self):
        self._db = None
        self._fetcher = None
        self._predictor = None
    
    def init_app(self, db, fetcher, predictor):
        """Initialize with dependencies"""
        self._db = db
        self._fetcher = fetcher
        self._predictor = predictor
    
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
    
    @property
    def predictor(self):
        """Lazy load predictor"""
        if self._predictor is None:
            from models.predictor import DayTradePredictor
            self._predictor = DayTradePredictor()
        return self._predictor
    
    def analyze_stock(self, symbol):
        """Perform complete technical analysis on a stock"""
        try:
            # Get historical data
            df = self.fetcher.get_stock_data(symbol)
            
            if df.empty:
                logger.warning(f"No data available for {symbol}")
                return None
            
            # Sort data chronologically
            df = df.sort_index(ascending=True)
            
            # Ensure lowercase column names for predictor
            if 'Close' in df.columns:
                df = df.rename(columns={
                    'Open': 'open',
                    'High': 'high', 
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
            
            # Analyze with indicators
            indicators_df = self.predictor.analyze_stock(symbol, df)
            
            if indicators_df is None:
                return None
            
            # Get latest indicators - safely handle the iloc operation
            latest_row = indicators_df.iloc[-1]
            latest = {}
            
            # Convert each value individually to handle array issues
            for col in latest_row.index:
                val = latest_row[col]
                # Handle numpy arrays and pandas series
                if hasattr(val, '__len__') and not isinstance(val, str):
                    # If it's an array-like object, take the first element
                    latest[col] = float(val[0]) if len(val) > 0 else None
                elif pd.isna(val):
                    latest[col] = None
                else:
                    try:
                        latest[col] = float(val)
                    except:
                        latest[col] = val
            
            # Get entry/exit points
            suggestions = self.predictor.get_entry_exit_points(symbol, indicators_df)
            
            # Score the opportunity
            score, reasons, signals, patterns = self.predictor.score_opportunity(
                latest, 
                indicators_df
            )
            
            # Get company info
            name = self.fetcher.get_company_name(symbol)
            realtime = self.fetcher.get_realtime_metrics(symbol)
            
            # Prepare chart data
            chart_data = self._prepare_chart_data(indicators_df.tail(60))
            
            # Build response with safe value extraction
            response = {
                'symbol': symbol,
                'name': name,
                'current_indicators': {},
                'score': score,
                'reasons': reasons,
                'signals': signals,
                'patterns': patterns,
                'suggestions': suggestions,
                'realtime_metrics': realtime,
                'chart_data': chart_data,
                'analyzed_at': datetime.now().isoformat()
            }
            
            # Safely extract indicator values
            indicator_mappings = {
                'price': 'Close',
                'rsi': 'RSI',
                'volatility': 'Volatility',
                'volume_trend': 'Volume_Trend',
                'ma_signal': 'Signal',
                'macd': 'MACD',
                'bb_upper': 'BB_Upper',
                'bb_lower': 'BB_Lower',
                'atr': 'ATR',
                'adx': 'ADX',
                'mfi': 'MFI'
            }
            
            for key, col in indicator_mappings.items():
                if col in latest:
                    val = latest[col]
                    if val is not None and not pd.isna(val):
                        response['current_indicators'][key] = float(val) if key != 'ma_signal' else int(val)
                    else:
                        response['current_indicators'][key] = None
                else:
                    response['current_indicators'][key] = None
            
            return response
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}", exc_info=True)
            return None
    
    def get_indicators(self, symbol, indicators, period='1m'):
        """Get specific indicators for a stock"""
        try:
            # Get historical data
            period_map = {
                '1d': '5d',
                '1w': '1mo',
                '1m': '3mo',
                '3m': '6mo',
                '6m': '1y',
                '1y': '2y'
            }
            yf_period = period_map.get(period, '3mo')
            
            df = self.fetcher.get_stock_data(symbol, period=yf_period)
            
            if df.empty:
                return None
            
            # Calculate requested indicators
            indicators_df = self.predictor.analyze_stock(symbol, df, period=period)
            
            if indicators_df is None:
                return None
            
            # Filter to requested indicators
            if 'all' not in indicators:
                available_indicators = {
                    'rsi': 'RSI',
                    'macd': ['MACD', 'MACD_Signal', 'MACD_Histogram'],
                    'bb': ['BB_Upper', 'BB_Middle', 'BB_Lower'],
                    'ma': ['MA_Short', 'MA_Long'],
                    'volume': 'Volume_Trend',
                    'atr': 'ATR',
                    'adx': 'ADX',
                    'mfi': 'MFI',
                    'cci': 'CCI',
                    'stoch': ['Stoch_K', 'Stoch_D']
                }
                
                columns_to_keep = ['Close', 'Volume']
                for ind in indicators:
                    if ind in available_indicators:
                        if isinstance(available_indicators[ind], list):
                            columns_to_keep.extend(available_indicators[ind])
                        else:
                            columns_to_keep.append(available_indicators[ind])
                
                # Keep only requested columns that exist
                columns_to_keep = [col for col in columns_to_keep if col in indicators_df.columns]
                indicators_df = indicators_df[columns_to_keep]
            
            # Convert to dict format
            result = {
                'symbol': symbol,
                'period': period,
                'data': []
            }
            
            for date, row in indicators_df.tail(100).iterrows():
                data_point = {'date': date.strftime('%Y-%m-%d')}
                for col in indicators_df.columns:
                    value = row[col]
                    if pd.notna(value):
                        # Handle array-like values
                        if hasattr(value, '__len__') and not isinstance(value, str):
                            data_point[col] = float(value[0]) if len(value) > 0 else None
                        else:
                            data_point[col] = float(value) if col != 'Volume' else int(value)
                    else:
                        data_point[col] = None
                result['data'].append(data_point)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting indicators for {symbol}: {str(e)}")
            return None
    
    def detect_patterns(self, symbol, min_strength=0.5, lookback_days=60):
        """Detect chart patterns for a stock"""
        try:
            # Get historical data
            df = self.fetcher.get_stock_data(symbol)
            
            if df.empty or len(df) < lookback_days:
                return []
            
            # Use pattern recognition
            from analysis.pattern_recognition import PatternRecognition
            pattern_recognition = PatternRecognition()
            
            # Detect patterns
            patterns = pattern_recognition.detect_chart_patterns(df.tail(lookback_days))
            
            # Filter by strength
            filtered_patterns = [
                p for p in patterns 
                if p.get('strength', 0) >= min_strength
            ]
            
            # Add pattern score
            current_price = df['close'].iloc[-1] if 'close' in df.columns else df['Close'].iloc[-1]
            pattern_score = pattern_recognition.calculate_pattern_score(
                filtered_patterns, 
                current_price
            )
            
            # Enhance pattern data
            for pattern in filtered_patterns:
                pattern['impact_score'] = pattern_score
                pattern['detected_at'] = datetime.now().isoformat()
            
            return filtered_patterns
            
        except Exception as e:
            logger.error(f"Error detecting patterns for {symbol}: {str(e)}")
            return []
    
    def get_support_resistance(self, symbol, window=20, num_touches=2):
        """Get support and resistance levels"""
        try:
            # Get historical data
            df = self.fetcher.get_stock_data(symbol)
            
            if df.empty:
                return {'support': [], 'resistance': []}
            
            # Use pattern recognition
            from analysis.pattern_recognition import PatternRecognition
            pattern_recognition = PatternRecognition()
            
            # Find levels
            levels = pattern_recognition.find_support_resistance(
                df, 
                window=window, 
                num_touches=num_touches
            )
            
            # Add current price for context
            current_price = df['close'].iloc[-1] if 'close' in df.columns else df['Close'].iloc[-1]
            levels['current_price'] = float(current_price)
            
            # Calculate distance from current price
            for support in levels['support']:
                support['distance'] = (current_price - support['level']) / current_price
                support['distance_percent'] = support['distance'] * 100
            
            for resistance in levels['resistance']:
                resistance['distance'] = (resistance['level'] - current_price) / current_price
                resistance['distance_percent'] = resistance['distance'] * 100
            
            return levels
            
        except Exception as e:
            logger.error(f"Error getting S/R levels for {symbol}: {str(e)}")
            return {'support': [], 'resistance': []}
    
    def get_sentiment_context(self, symbol):
        """Get additional context for sentiment analysis"""
        try:
            # Get recent price action
            df = self.fetcher.get_stock_data(symbol)
            
            if df.empty:
                return {}
            
            # Calculate various metrics
            close_col = 'close' if 'close' in df.columns else 'Close'
            volume_col = 'volume' if 'volume' in df.columns else 'Volume'
            
            # Price changes
            price_change_1w = (df[close_col].iloc[-1] - df[close_col].iloc[-6]) / df[close_col].iloc[-6] * 100 if len(df) >= 6 else 0
            price_change_1m = (df[close_col].iloc[-1] - df[close_col].iloc[-22]) / df[close_col].iloc[-22] * 100 if len(df) >= 22 else 0
            
            # Volume analysis
            avg_volume = df[volume_col].tail(20).mean()
            current_volume = df[volume_col].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Volatility
            returns = df[close_col].pct_change()
            volatility = returns.tail(20).std() * np.sqrt(252)  # Annualized
            
            # Get fundamentals
            fundamentals = self.fetcher.get_stock_fundamentals(symbol)
            
            return {
                'price_momentum': {
                    'change_1w': price_change_1w,
                    'change_1m': price_change_1m,
                    'trend': 'bullish' if price_change_1m > 5 else 'bearish' if price_change_1m < -5 else 'neutral'
                },
                'volume_analysis': {
                    'current_vs_average': volume_ratio,
                    'interpretation': 'high' if volume_ratio > 1.5 else 'low' if volume_ratio < 0.5 else 'normal'
                },
                'volatility': {
                    'annual': volatility,
                    'level': 'high' if volatility > 0.5 else 'low' if volatility < 0.2 else 'moderate'
                },
                'valuation': {
                    'pe_ratio': fundamentals.get('pe_ratio') if fundamentals else None,
                    'price_target': fundamentals.get('price_target') if fundamentals else None,
                    'analyst_rating': fundamentals.get('analyst_rating') if fundamentals else None
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting sentiment context for {symbol}: {str(e)}")
            return {}
    
    def backtest_strategy(self, symbol, strategy='default', period='1y', initial_capital=10000):
        """Backtest a trading strategy"""
        try:
            # Get historical data
            period_map = {
                '3m': '6mo',
                '6m': '1y',
                '1y': '2y',
                '2y': '5y'
            }
            yf_period = period_map.get(period, '2y')
            
            df = self.fetcher.get_stock_data(symbol, period=yf_period)
            
            if df.empty or len(df) < 100:
                return None
            
            # Define strategy parameters
            strategy_params = {
                'default': {
                    'min_score': 5,
                    'stop_loss_pct': 0.02,
                    'take_profit_pct': 0.05,
                    'exit_score': -3
                },
                'aggressive': {
                    'min_score': 3,
                    'stop_loss_pct': 0.03,
                    'take_profit_pct': 0.10,
                    'exit_score': -2
                },
                'conservative': {
                    'min_score': 7,
                    'stop_loss_pct': 0.01,
                    'take_profit_pct': 0.03,
                    'exit_score': -5
                }
            }
            
            params = strategy_params.get(strategy, strategy_params['default'])
            
            # Run backtest
            results = self.predictor.backtest_strategy(symbol, df, params)
            
            if not results:
                return None
            
            # Add additional metrics
            results['strategy'] = strategy
            results['period'] = period
            results['initial_capital'] = initial_capital
            results['final_capital'] = initial_capital * (1 + results['total_return'])
            results['annualized_return'] = results['total_return'] / float(period[:-1]) if period.endswith('y') else results['total_return'] * 4
            
            return results
            
        except Exception as e:
            logger.error(f"Error backtesting {symbol}: {str(e)}")
            return None
    
    def _prepare_chart_data(self, df):
        """Prepare data for charting"""
        chart_data = []
        
        for date, row in df.iterrows():
            chart_point = {
                'date': date.strftime('%Y-%m-%d'),
                'close': self._safe_float(row.get('Close', row.get('close'))),
                'volume': self._safe_int(row.get('Volume', row.get('volume'))),
                'rsi': self._safe_float(row.get('RSI')),
                'ma_short': self._safe_float(row.get('MA_Short')),
                'ma_long': self._safe_float(row.get('MA_Long')),
                'bb_upper': self._safe_float(row.get('BB_Upper')),
                'bb_lower': self._safe_float(row.get('BB_Lower')),
                'macd': self._safe_float(row.get('MACD')),
                'macd_signal': self._safe_float(row.get('MACD_Signal'))
            }
            chart_data.append(chart_point)
        
        return chart_data
    
    def _safe_float(self, value):
        """Safely convert value to float"""
        if value is None or pd.isna(value):
            return None
        if hasattr(value, '__len__') and not isinstance(value, str):
            return float(value[0]) if len(value) > 0 else None
        try:
            return float(value)
        except:
            return None
    
    def _safe_int(self, value):
        """Safely convert value to int"""
        if value is None or pd.isna(value):
            return None
        if hasattr(value, '__len__') and not isinstance(value, str):
            return int(value[0]) if len(value) > 0 else None
        try:
            return int(value)
        except:
            return None