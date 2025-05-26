# src/web/services/prediction_service.py
"""
AI prediction service layer
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class PredictionService:
    """Service for AI prediction operations"""
    
    def __init__(self):
        self._db = None
        self._fetcher = None
        self._predictor = None
        self.predictions_cache = None
        self.predictions_last_update = None
        self.predictions_interval = 300  # 5 minutes
    
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
    
    def get_predictions(self, force_refresh=False, top_n=20, min_score=2, period='1d'):
        """Get AI-powered trading predictions"""
        now = datetime.now()
        
        # Check cache
        if (not force_refresh and 
            self.predictions_cache is not None and 
            self.predictions_last_update is not None and
            (now - self.predictions_last_update).total_seconds() < self.predictions_interval):
            
            logger.info("Returning cached predictions")
            return self.predictions_cache
        
        logger.info("Generating new AI-powered predictions...")
        
        try:
            # Get stock data for analysis
            stocks_data = self.fetcher.get_stocks_for_analysis()
            
            if not stocks_data:
                logger.warning("No stock data available for analysis")
                self.predictions_cache = {
                    'opportunities': [],
                    'generated_at': now.isoformat(),
                    'status': 'no_data',
                    'message': 'Unable to retrieve sufficient stock data for analysis'
                }
                return self.predictions_cache
            
            logger.info(f"Analyzing {len(stocks_data)} stocks for opportunities")
            
            # Find opportunities
            opportunities_df = self.predictor.find_opportunities(
                stocks_data, 
                top_n=top_n * 2,  # Get more to filter
                period=period
            )
            
            if not opportunities_df.empty:
                # Convert to list
                opportunities = opportunities_df.to_dict('records')
                
                # Filter by minimum score
                opportunities = [
                    opp for opp in opportunities 
                    if opp.get('score', 0) >= min_score
                ]
                
                # Sort and limit
                opportunities.sort(key=lambda x: x['score'], reverse=True)
                opportunities = opportunities[:top_n]
                
                # Enhance opportunities
                for opp in opportunities:
                    # Add company name
                    opp['name'] = self.fetcher.get_company_name(opp['ticker'])
                    
                    # Clean up numpy types and handle arrays
                    for key, value in opp.items():
                        if isinstance(value, (np.integer, np.int64)):
                            opp[key] = int(value)
                        elif isinstance(value, (np.floating, np.float64)):
                            opp[key] = float(value)
                        elif isinstance(value, np.ndarray):
                            opp[key] = value.tolist()
                        elif isinstance(value, (list, dict)):
                            # Skip complex types
                            continue
                        else:
                            # Handle scalar NaN values
                            try:
                                if pd.isna(value):
                                    opp[key] = None
                            except:
                                # If it fails, just keep the value
                                pass
                
                # Get market context
                market_overview = self.fetcher.get_market_overview()
                
                self.predictions_cache = {
                    'opportunities': opportunities,
                    'generated_at': now.isoformat(),
                    'status': 'success',
                    'stocks_analyzed': len(stocks_data),
                    'market_context': {
                        'fear_greed_index': self._calculate_market_fear_greed(market_overview),
                        'market_trend': 'bullish' if market_overview.get('S&P 500', {}).get('change', 0) > 0 else 'bearish',
                        'vix_level': market_overview.get('VIX (Fear Index)', {}).get('value', 0)
                    },
                    'parameters': {
                        'period': period,
                        'min_score': min_score,
                        'top_n': top_n
                    }
                }
                
                logger.info(f"Found {len(opportunities)} trading opportunities")
            else:
                self.predictions_cache = {
                    'opportunities': [],
                    'generated_at': now.isoformat(),
                    'status': 'no_opportunities',
                    'stocks_analyzed': len(stocks_data),
                    'message': f'Analyzed {len(stocks_data)} stocks but found no strong trading signals'
                }
                logger.info("No trading opportunities found")
            
            self.predictions_last_update = now
            return self.predictions_cache
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}", exc_info=True)
            return {
                'opportunities': [],
                'generated_at': now.isoformat(),
                'status': 'error',
                'error': str(e)
            }
    
    def get_available_strategies(self):
        """Get list of available prediction strategies"""
        # Define available strategies
        strategies = [
            {
                'name': 'momentum',
                'display_name': 'Momentum Trading',
                'description': 'Identifies stocks with strong price momentum',
                'parameters': {
                    'lookback_period': 20,
                    'min_momentum': 0.05
                }
            },
            {
                'name': 'mean_reversion',
                'display_name': 'Mean Reversion',
                'description': 'Finds oversold stocks likely to bounce back',
                'parameters': {
                    'rsi_threshold': 30,
                    'bb_deviation': 2
                }
            },
            {
                'name': 'breakout',
                'display_name': 'Breakout Trading',
                'description': 'Detects stocks breaking out of consolidation',
                'parameters': {
                    'consolidation_period': 20,
                    'volume_surge': 1.5
                }
            },
            {
                'name': 'pattern',
                'display_name': 'Pattern Recognition',
                'description': 'Uses chart patterns for trading signals',
                'parameters': {
                    'min_pattern_strength': 0.7,
                    'pattern_types': ['double_bottom', 'ascending_triangle', 'bull_flag']
                }
            },
            {
                'name': 'ai_composite',
                'display_name': 'AI Composite',
                'description': 'Combines multiple strategies with machine learning',
                'parameters': {
                    'min_confidence': 0.7,
                    'use_sentiment': True
                }
            }
        ]
        
        return strategies
    
    def strategy_exists(self, strategy_name):
        """Check if a strategy exists"""
        strategies = self.get_available_strategies()
        return any(s['name'] == strategy_name for s in strategies)
    
    def get_strategy_predictions(self, strategy_name, symbols=None, top_n=10):
        """Get predictions for a specific strategy"""
        if not self.strategy_exists(strategy_name):
            return None
        
        # Get stock data
        if symbols:
            stocks_data = {}
            for symbol in symbols:
                df = self.fetcher.get_stock_data(symbol)
                if not df.empty:
                    stocks_data[symbol] = df
        else:
            stocks_data = self.fetcher.get_stocks_for_analysis()
        
        if not stocks_data:
            return {
                'strategy': strategy_name,
                'opportunities': [],
                'message': 'No stock data available'
            }
        
        # Apply strategy-specific logic
        opportunities = []
        
        if strategy_name == 'momentum':
            opportunities = self._momentum_strategy(stocks_data)
        elif strategy_name == 'mean_reversion':
            opportunities = self._mean_reversion_strategy(stocks_data)
        elif strategy_name == 'breakout':
            opportunities = self._breakout_strategy(stocks_data)
        elif strategy_name == 'pattern':
            opportunities = self._pattern_strategy(stocks_data)
        elif strategy_name == 'ai_composite':
            # Use the main predictor
            opportunities_df = self.predictor.find_opportunities(stocks_data, top_n=top_n*2)
            if not opportunities_df.empty:
                opportunities = opportunities_df.to_dict('records')
        
        # Sort and limit
        opportunities.sort(key=lambda x: x.get('score', 0), reverse=True)
        opportunities = opportunities[:top_n]
        
        # Enhance opportunities
        for opp in opportunities:
            opp['name'] = self.fetcher.get_company_name(opp['ticker'])
            opp['strategy'] = strategy_name
        
        return {
            'strategy': strategy_name,
            'opportunities': opportunities,
            'generated_at': datetime.now().isoformat()
        }
    
    def get_detailed_score(self, symbol):
        """Get detailed prediction score breakdown for a stock"""
        try:
            # Get stock data
            df = self.fetcher.get_stock_data(symbol)
            
            if df.empty:
                return None
            
            # Analyze the stock
            indicators_df = self.predictor.analyze_stock(symbol, df)
            
            if indicators_df is None:
                return None
            
            # Get latest data
            latest = indicators_df.iloc[-1]
            
            # Score the opportunity
            score, reasons, signals, patterns = self.predictor.score_opportunity(
                latest, 
                indicators_df
            )
            
            # Break down the score
            score_breakdown = {
                'technical_indicators': 0,
                'patterns': 0,
                'momentum': 0,
                'volume': 0,
                'ai_adjustment': 0
            }
            
            # Analyze score components
            if 'rsi_signal' in signals:
                score_breakdown['technical_indicators'] += 2 if signals['rsi_signal'] == 'oversold' else -1
            
            if patterns:
                score_breakdown['patterns'] = len([p for p in patterns if 'bullish' in str(p.get('pattern', ''))]) * 2
            
            if 'momentum' in signals:
                score_breakdown['momentum'] = 2 if signals['momentum'] == 'strong_bullish' else -2
            
            if 'volume_trend' in signals and signals.get('volume_trend', 1) > 1.5:
                score_breakdown['volume'] = 1
            
            # AI adjustment (difference between raw and adjusted score)
            raw_score = sum(score_breakdown.values())
            score_breakdown['ai_adjustment'] = score - raw_score
            
            return {
                'symbol': symbol,
                'total_score': score,
                'score_breakdown': score_breakdown,
                'reasons': reasons,
                'signals': signals,
                'patterns': patterns,
                'confidence': self._calculate_confidence(signals, patterns),
                'recommendation': self._get_recommendation(score),
                'analyzed_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting detailed score for {symbol}: {str(e)}")
            return None
    
    def submit_feedback(self, symbol, predicted_score, actual_return, patterns=None, signals=None):
        """Submit feedback on prediction accuracy"""
        try:
            # Record feedback in the learning engine
            self.predictor.learning_engine.record_feedback(
                symbol=symbol,
                predicted_score=predicted_score,
                actual_return=actual_return,
                patterns=patterns or [],
                signals=signals or {}
            )
            
            logger.info(f"Recorded feedback for {symbol}: score={predicted_score}, return={actual_return}")
            return True
            
        except Exception as e:
            logger.error(f"Error submitting feedback: {str(e)}")
            return False
    
    def get_performance_metrics(self, period='30d', strategy='all'):
        """Get performance metrics for predictions"""
        try:
            # Parse period
            days = 30
            if period.endswith('d'):
                days = int(period[:-1])
            elif period.endswith('w'):
                days = int(period[:-1]) * 7
            elif period.endswith('m'):
                days = int(period[:-1]) * 30
            
            # Get feedback data from learning engine
            conn = self.predictor.learning_engine.db_path
            
            # This is a placeholder - implement actual performance tracking
            metrics = {
                'period': period,
                'strategy': strategy,
                'total_predictions': 0,
                'successful_predictions': 0,
                'win_rate': 0,
                'average_return': 0,
                'best_prediction': None,
                'worst_prediction': None,
                'score_distribution': {},
                'pattern_performance': {}
            }
            
            # TODO: Query actual feedback data and calculate metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            return None
    
    def _momentum_strategy(self, stocks_data):
        """Momentum trading strategy"""
        opportunities = []
        
        for symbol, df in stocks_data.items():
            try:
                if len(df) < 20:
                    continue
                
                # Calculate momentum
                close_col = 'Close' if 'Close' in df.columns else 'close'
                momentum_5d = (df[close_col].iloc[-1] - df[close_col].iloc[-6]) / df[close_col].iloc[-6]
                momentum_20d = (df[close_col].iloc[-1] - df[close_col].iloc[-20]) / df[close_col].iloc[-20]
                
                if momentum_5d > 0.03 and momentum_20d > 0.05:
                    opportunities.append({
                        'ticker': symbol,
                        'score': int(momentum_5d * 100),
                        'price': float(df[close_col].iloc[-1]),
                        'momentum_5d': momentum_5d,
                        'momentum_20d': momentum_20d,
                        'reasons': [f"Strong momentum: 5d={momentum_5d:.1%}, 20d={momentum_20d:.1%}"]
                    })
                    
            except Exception as e:
                logger.error(f"Error in momentum strategy for {symbol}: {e}")
        
        return opportunities
    
    def _mean_reversion_strategy(self, stocks_data):
        """Mean reversion strategy"""
        opportunities = []
        
        for symbol, df in stocks_data.items():
            try:
                # Calculate indicators
                indicators_df = self.predictor.analyze_stock(symbol, df)
                if indicators_df is None:
                    continue
                
                latest = indicators_df.iloc[-1]
                
                # Check for oversold conditions
                if (pd.notna(latest.get('RSI')) and latest['RSI'] < 30 and
                    pd.notna(latest.get('BB_Lower')) and latest['Close'] < latest['BB_Lower']):
                    
                    opportunities.append({
                        'ticker': symbol,
                        'score': int((30 - latest['RSI']) / 3),
                        'price': float(latest['Close']),
                        'rsi': float(latest['RSI']),
                        'reasons': ["Oversold conditions for mean reversion"]
                    })
                    
            except Exception as e:
                logger.error(f"Error in mean reversion strategy for {symbol}: {e}")
        
        return opportunities
    
    def _breakout_strategy(self, stocks_data):
        """Breakout trading strategy"""
        opportunities = []
        
        for symbol, df in stocks_data.items():
            try:
                if len(df) < 20:
                    continue
                
                # Detect breakouts using pattern recognition
                from analysis.pattern_recognition import PatternRecognition
                pr = PatternRecognition()
                
                breakouts = pr.detect_breakouts(df)
                
                if breakouts:
                    latest_breakout = breakouts[-1]
                    if latest_breakout['type'] == 'bullish_breakout':
                        opportunities.append({
                            'ticker': symbol,
                            'score': int(latest_breakout['strength'] * 100),
                            'price': float(df['close'].iloc[-1] if 'close' in df.columns else df['Close'].iloc[-1]),
                            'breakout_level': latest_breakout['breakout_level'],
                            'volume_surge': latest_breakout['volume_surge'],
                            'reasons': ["Bullish breakout detected with volume confirmation"]
                        })
                        
            except Exception as e:
                logger.error(f"Error in breakout strategy for {symbol}: {e}")
        
        return opportunities
    
    def _pattern_strategy(self, stocks_data):
        """Pattern recognition strategy"""
        opportunities = []
        
        for symbol, df in stocks_data.items():
            try:
                # Detect patterns
                from analysis.pattern_recognition import PatternRecognition
                pr = PatternRecognition()
                
                patterns = pr.detect_chart_patterns(df)
                
                # Filter bullish patterns
                bullish_patterns = [
                    p for p in patterns 
                    if p.get('pattern') in ['double_bottom', 'ascending_triangle', 'bull_flag', 'falling_wedge']
                    and p.get('strength', 0) >= 0.7
                ]
                
                if bullish_patterns:
                    strongest_pattern = max(bullish_patterns, key=lambda x: x.get('strength', 0))
                    
                    opportunities.append({
                        'ticker': symbol,
                        'score': int(strongest_pattern['strength'] * 10),
                        'price': float(df['close'].iloc[-1] if 'close' in df.columns else df['Close'].iloc[-1]),
                        'pattern': strongest_pattern['pattern'],
                        'pattern_strength': strongest_pattern['strength'],
                        'reasons': [f"{strongest_pattern['pattern'].replace('_', ' ').title()} pattern detected"]
                    })
                    
            except Exception as e:
                logger.error(f"Error in pattern strategy for {symbol}: {e}")
        
        return opportunities
    
    def _calculate_market_fear_greed(self, market_data):
        """Calculate fear and greed index"""
        if 'fear_greed_index' in market_data:
            return market_data['fear_greed_index']
        
        # Simple calculation based on VIX
        vix = market_data.get('VIX (Fear Index)', {}).get('value', 20)
        if vix < 12:
            return 80  # Extreme greed
        elif vix < 16:
            return 65  # Greed
        elif vix < 20:
            return 50  # Neutral
        elif vix < 28:
            return 35  # Fear
        else:
            return 20  # Extreme fear
    
    def _calculate_confidence(self, signals, patterns):
        """Calculate confidence level for prediction"""
        confidence = 0.5
        
        # Adjust based on signal alignment
        bullish_signals = sum(1 for s in signals.values() if 'bullish' in str(s).lower())
        bearish_signals = sum(1 for s in signals.values() if 'bearish' in str(s).lower())
        
        if bullish_signals > bearish_signals * 2:
            confidence += 0.2
        elif bearish_signals > bullish_signals * 2:
            confidence -= 0.2
        
        # Adjust based on patterns
        if patterns:
            confidence += min(0.3, len(patterns) * 0.1)
        
        return max(0, min(1, confidence))
    
    def _get_recommendation(self, score):
        """Get recommendation based on score"""
        if score >= 7:
            return 'STRONG BUY'
        elif score >= 5:
            return 'BUY'
        elif score >= 3:
            return 'HOLD'
        elif score >= 0:
            return 'WATCH'
        elif score >= -3:
            return 'SELL'
        else:
            return 'STRONG SELL'