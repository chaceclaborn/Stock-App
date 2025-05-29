# src/models/predictor.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import sqlite3
import os
import sys

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import from analysis modules
from analysis.indicators import StockIndicators
from analysis.pattern_recognition import PatternRecognition

logger = logging.getLogger(__name__)

class AIAnalyst:
    """AI-powered analysis and explanations"""
    
    def __init__(self):
        self.explanations = {
            'rsi_oversold': "The RSI (Relative Strength Index) is below 30, indicating the stock may be oversold. This suggests potential buying opportunity as the price might bounce back.",
            'rsi_overbought': "The RSI is above 70, indicating the stock may be overbought. This suggests caution as the price might pull back.",
            'macd_bullish': "The MACD line crossed above the signal line, which is a bullish signal. This suggests upward momentum is building.",
            'macd_bearish': "The MACD line crossed below the signal line, which is a bearish signal. This suggests downward momentum.",
            'ma_bullish': "The short-term moving average is above the long-term moving average, indicating an uptrend.",
            'ma_bearish': "The short-term moving average is below the long-term moving average, indicating a downtrend.",
            'bb_squeeze': "Bollinger Bands are narrowing, indicating low volatility. This often precedes a significant price movement.",
            'volume_surge': "Trading volume is significantly above average, indicating strong interest in the stock.",
            'pattern_detected': "A chart pattern has been detected that may indicate future price movement.",
            'support_bounce': "Price is bouncing off a strong support level, suggesting potential upward movement.",
            'resistance_test': "Price is testing a resistance level. A breakout could lead to significant gains.",
            'divergence_bullish': "Bullish divergence detected - price making lower lows while indicator makes higher lows.",
            'divergence_bearish': "Bearish divergence detected - price making higher highs while indicator makes lower highs."
        }
    
    def generate_explanation(self, indicators, signals, score, patterns=None):
        """Generate natural language explanation of the analysis"""
        explanation = []
        
        # Overall assessment
        if score >= 7:
            explanation.append("🚀 **Strong Trading Opportunity** - Multiple bullish signals are aligning.")
        elif score >= 5:
            explanation.append("📈 **Moderate Trading Opportunity** - Several positive indicators present.")
        elif score >= 3:
            explanation.append("⚠️ **Mixed Signals** - Some opportunities but proceed with caution.")
        else:
            explanation.append("📊 **Weak Signal** - Limited trading opportunities at this time.")
        
        # Specific indicator explanations
        if 'rsi' in indicators:
            rsi_value = indicators['rsi']
            if rsi_value < 30:
                explanation.append(f"\n• {self.explanations['rsi_oversold']} (RSI: {rsi_value:.1f})")
            elif rsi_value > 70:
                explanation.append(f"\n• {self.explanations['rsi_overbought']} (RSI: {rsi_value:.1f})")
        
        if 'macd_crossover' in signals:
            if signals['macd_crossover'] == 'bullish':
                explanation.append(f"\n• {self.explanations['macd_bullish']}")
            else:
                explanation.append(f"\n• {self.explanations['macd_bearish']}")
        
        # Pattern explanations
        if patterns:
            for pattern in patterns:
                pattern_type = pattern.get('pattern', pattern.get('type', ''))
                if 'divergence' in pattern_type:
                    if 'bullish' in pattern_type:
                        explanation.append(f"\n• {self.explanations['divergence_bullish']}")
                    else:
                        explanation.append(f"\n• {self.explanations['divergence_bearish']}")
                elif pattern_type in ['head_and_shoulders', 'double_top', 'double_bottom']:
                    explanation.append(f"\n• Strong {pattern_type.replace('_', ' ')} pattern detected with {pattern.get('strength', 0.5)*100:.0f}% confidence")
        
        return "\n".join(explanation)
    
    def generate_trading_strategy(self, symbol, analysis_data):
        """Generate AI-powered trading strategy suggestions"""
        strategy = {
            'symbol': symbol,
            'recommendation': '',
            'entry_strategy': '',
            'exit_strategy': '',
            'risk_management': '',
            'time_horizon': '',
            'confidence': 0
        }
        
        score = analysis_data.get('score', 0)
        signals = analysis_data.get('signals', {})
        current_price = analysis_data.get('current_price', 0)
        patterns = analysis_data.get('patterns', [])
        
        # Calculate confidence based on multiple factors
        confidence_factors = []
        
        # Pattern confidence
        if patterns:
            pattern_strengths = [p.get('strength', 0.5) for p in patterns]
            confidence_factors.append(np.mean(pattern_strengths))
        
        # Signal alignment
        bullish_signals = sum(1 for s in signals.values() if 'bullish' in str(s).lower())
        bearish_signals = sum(1 for s in signals.values() if 'bearish' in str(s).lower())
        signal_alignment = abs(bullish_signals - bearish_signals) / max(len(signals), 1)
        confidence_factors.append(signal_alignment)
        
        # Overall confidence
        strategy['confidence'] = np.mean(confidence_factors) if confidence_factors else 0.5
        
        # Determine recommendation
        if score >= 7:
            strategy['recommendation'] = 'STRONG BUY'
            strategy['time_horizon'] = 'Short-term (1-5 days)'
        elif score >= 5:
            strategy['recommendation'] = 'BUY'
            strategy['time_horizon'] = 'Short to Medium-term (1-2 weeks)'
        elif score >= 3:
            strategy['recommendation'] = 'HOLD/WATCH'
            strategy['time_horizon'] = 'Wait for better entry'
        elif score <= -3:
            strategy['recommendation'] = 'SELL/SHORT'
            strategy['time_horizon'] = 'Exit positions'
        else:
            strategy['recommendation'] = 'AVOID'
            strategy['time_horizon'] = 'No clear opportunity'
        
        # Entry strategy
        if 'support_entry' in analysis_data:
            strategy['entry_strategy'] = f"Consider entering near support at ${analysis_data['support_entry']:.2f}"
        else:
            strategy['entry_strategy'] = f"Consider scaling into position around ${current_price:.2f}"
        
        # Exit strategy
        if 'resistance_exit' in analysis_data:
            strategy['exit_strategy'] = f"Target exit near resistance at ${analysis_data['resistance_exit']:.2f}"
        else:
            strategy['exit_strategy'] = f"Set profit target at +5-10% from entry"
        
        # Risk management
        strategy['risk_management'] = f"Set stop loss at -2% from entry. Never risk more than 1-2% of portfolio on single trade."
        
        return strategy

class AILearningEngine:
    """Machine learning engine that improves predictions based on feedback"""
    
    def __init__(self, db_path):
        self.db_path = db_path
        self.init_database()
        self.model_weights = self.load_model_weights()
        self.pattern_success_rates = self.load_pattern_success_rates()
    
    def init_database(self):
        """Initialize the AI feedback database"""
        try:
            # Create directory if it doesn't exist
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create feedback table if it doesn't exist
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                predicted_score REAL,
                actual_return REAL,
                feedback_date TEXT,
                patterns_detected TEXT,
                signals_detected TEXT
            )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("AI feedback database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing AI feedback database: {e}")
    
    def load_model_weights(self):
        """Load model weights from feedback data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get feedback data
            cursor.execute('''
            SELECT predicted_score, AVG(actual_return), COUNT(*), 
                   SUM(CASE WHEN actual_return > 0 THEN 1 ELSE 0 END) as positive_returns
            FROM ai_feedback
            WHERE feedback_date > datetime('now', '-90 days')
            GROUP BY predicted_score
            ''')
            
            weights = {}
            for row in cursor.fetchall():
                score = row[0]
                avg_return = row[1] or 0
                count = row[2]
                positive_rate = (row[3] / count) if count > 0 else 0.5
                
                # Calculate weight based on performance
                weight = 1.0
                if count >= 10:  # Enough data for this score
                    if positive_rate > 0.6:  # Good performance
                        weight = 1.2
                    elif positive_rate < 0.4:  # Poor performance
                        weight = 0.8
                
                weights[score] = weight
            
            conn.close()
            return weights
            
        except Exception as e:
            logger.error(f"Error loading model weights: {e}")
            return {}
    
    def load_pattern_success_rates(self):
        """Load success rates for different patterns"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT patterns_detected, AVG(actual_return), COUNT(*)
            FROM ai_feedback
            WHERE patterns_detected IS NOT NULL
            AND feedback_date > datetime('now', '-90 days')
            GROUP BY patterns_detected
            ''')
            
            success_rates = {}
            for row in cursor.fetchall():
                patterns = json.loads(row[0]) if row[0] else []
                avg_return = row[1] or 0
                count = row[2]
                
                for pattern in patterns:
                    if pattern not in success_rates:
                        success_rates[pattern] = {'returns': [], 'count': 0}
                    success_rates[pattern]['returns'].append(avg_return)
                    success_rates[pattern]['count'] += count
            
            conn.close()
            return success_rates
            
        except Exception as e:
            logger.error(f"Error loading pattern success rates: {e}")
            return {}
    
    def adjust_score(self, base_score, signals, patterns=None):
        """Adjust score based on learned patterns"""
        # Get weight for this score range
        score_key = int(base_score)  # Round to nearest integer
        weight = self.model_weights.get(score_key, 1.0)
        
        # Apply weight
        adjusted_score = base_score * weight
        
        # Adjust based on pattern success rates
        if patterns and self.pattern_success_rates:
            pattern_adjustment = 0
            for pattern in patterns:
                pattern_type = pattern.get('pattern', pattern.get('type', ''))
                if pattern_type in self.pattern_success_rates:
                    success_data = self.pattern_success_rates[pattern_type]
                    if success_data['count'] >= 5:  # Enough data
                        avg_return = np.mean(success_data['returns'])
                        if avg_return > 0.02:  # 2% average return
                            pattern_adjustment += 0.5
                        elif avg_return < -0.02:
                            pattern_adjustment -= 0.5
            
            adjusted_score += pattern_adjustment
        
        # Additional adjustments based on signal combinations
        if 'rsi_divergence' in signals and signals.get('momentum') == 'strong_bullish':
            adjusted_score *= 1.1  # Boost for strong signal combination
        
        if signals.get('volume_price_confirmation') == 'bullish':
            adjusted_score *= 1.05  # Small boost for volume confirmation
        
        return min(10, max(-10, adjusted_score))  # Keep score in -10 to +10 range
    
    def record_feedback(self, symbol, predicted_score, actual_return, patterns, signals):
        """Record actual results for learning"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO ai_feedback (symbol, predicted_score, actual_return, 
                                   feedback_date, patterns_detected, signals_detected)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                predicted_score,
                actual_return,
                datetime.now().isoformat(),
                json.dumps(patterns) if patterns else None,
                json.dumps(signals) if signals else None
            ))
            
            conn.commit()
            conn.close()
            
            # Reload weights periodically
            if np.random.random() < 0.1:  # 10% chance
                self.model_weights = self.load_model_weights()
                self.pattern_success_rates = self.load_pattern_success_rates()
                
        except Exception as e:
            logger.error(f"Error recording feedback: {e}")

class DayTradePredictor:
    """Enhanced day trading predictor with AI capabilities and dual prediction models"""
    
    def __init__(self):
        """Initialize with default parameters"""
        self.indicators = StockIndicators()
        self.pattern_recognition = PatternRecognition()
        self.ai_analyst = AIAnalyst()
        
        # Get database path - create the data directory structure
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        db_dir = os.path.join(project_root, 'data')
        
        # Create directory if it doesn't exist
        os.makedirs(db_dir, exist_ok=True)
        
        db_path = os.path.join(db_dir, 'ai_feedback.db')
        
        # Initialize learning engine
        self.learning_engine = AILearningEngine(db_path)
        
        # Trading parameters (will be adjusted based on period)
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.min_volatility = 0.01
        self.volume_surge_threshold = 1.5
        
        # Risk parameters
        self.max_risk_per_trade = 0.02  # 2% max risk
        self.risk_reward_ratio = 2.0    # Minimum 2:1 reward/risk
        
        # Period-specific parameters
        self.period_params = {
            '1d': {  # Short-term (day trading)
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'min_volatility': 0.01,
                'volume_surge': 1.5,
                'ma_short': 5,
                'ma_long': 10,
                'pattern_weight': 1.2,
                'momentum_weight': 1.5,
                'volume_weight': 1.3
            },
            '1w': {  # Medium-term (swing trading)
                'rsi_oversold': 35,
                'rsi_overbought': 65,
                'min_volatility': 0.02,
                'volume_surge': 1.3,
                'ma_short': 10,
                'ma_long': 20,
                'pattern_weight': 1.3,
                'momentum_weight': 1.2,
                'volume_weight': 1.1
            },
            '1m': {  # Long-term (position trading)
                'rsi_oversold': 40,
                'rsi_overbought': 60,
                'min_volatility': 0.05,
                'volume_surge': 1.2,
                'ma_short': 20,
                'ma_long': 50,
                'pattern_weight': 1.5,
                'momentum_weight': 1.0,
                'volume_weight': 0.9
            },
            '1y': {  # Investment timeframe
                'rsi_oversold': 45,
                'rsi_overbought': 55,
                'min_volatility': 0.10,
                'volume_surge': 1.1,
                'ma_short': 50,
                'ma_long': 200,
                'pattern_weight': 1.0,
                'momentum_weight': 0.8,
                'volume_weight': 0.7
            }
        }
        
        # Learned weights for different predictors
        self.short_term_weights = {
            'patterns': {},
            'signals': {}
        }
        
        self.long_term_weights = {
            'patterns': {},
            'signals': {}
        }
    def set_period_parameters(self, period='1d'):
        """Set parameters based on trading period"""
        params = self.period_params.get(period, self.period_params['1d'])
        
        self.rsi_oversold = params['rsi_oversold']
        self.rsi_overbought = params['rsi_overbought']
        self.min_volatility = params['min_volatility']
        self.volume_surge_threshold = params['volume_surge']
        
        return params
    
    def analyze_stock(self, symbol, df, period='1d'):
        """Analyze a single stock and calculate all indicators"""
        if df.empty or len(df) < 30:
            return None
        
        # Set period parameters
        params = self.set_period_parameters(period)
        
        try:
            # Handle both uppercase and lowercase column names
            if 'Close' in df.columns:
                # Uppercase columns (from yfinance direct)
                close_prices = df['Close']
                volume = df['Volume']
                high_prices = df['High']
                low_prices = df['Low']
                open_prices = df['Open'] if 'Open' in df.columns else close_prices
            elif 'close' in df.columns:
                # Lowercase columns (from our database)
                close_prices = df['close']
                volume = df['volume']
                high_prices = df['high']
                low_prices = df['low']
                open_prices = df['open'] if 'open' in df.columns else close_prices
            else:
                logger.error(f"Unknown column format in DataFrame for {symbol}")
                return None
            
            # Calculate all indicators with period-specific parameters
            rsi = self.indicators.calculate_rsi(close_prices)
            ma_data = self.indicators.calculate_moving_averages(
                close_prices, 
                short_period=params['ma_short'], 
                long_period=params['ma_long']
            )
            volatility = self.indicators.calculate_volatility(close_prices)
            volume_trend = self.indicators.calculate_volume_trend(volume)
            bb_data = self.indicators.calculate_bollinger_bands(close_prices)
            macd_data = self.indicators.calculate_macd(close_prices)
            stoch_data = self.indicators.calculate_stochastic(high_prices, low_prices, close_prices)
            atr = self.indicators.calculate_atr(high_prices, low_prices, close_prices)
            
            # Additional indicators
            adx = self.indicators.calculate_adx(high_prices, low_prices, close_prices)
            williams_r = self.indicators.calculate_williams_r(high_prices, low_prices, close_prices)
            cci = self.indicators.calculate_cci(high_prices, low_prices, close_prices)
            mfi = self.indicators.calculate_mfi(high_prices, low_prices, close_prices, volume)
            obv = self.indicators.calculate_obv(close_prices, volume)
            vwap = self.indicators.calculate_vwap(high_prices, low_prices, close_prices, volume)
            
            # Combine all indicators
            indicators_df = pd.DataFrame(index=df.index)
            indicators_df['Open'] = open_prices
            indicators_df['High'] = high_prices
            indicators_df['Low'] = low_prices
            indicators_df['Close'] = close_prices
            indicators_df['Volume'] = volume
            indicators_df['RSI'] = rsi
            indicators_df = pd.concat([indicators_df, ma_data, bb_data, macd_data, stoch_data], axis=1)
            indicators_df['Volatility'] = volatility
            indicators_df['Volume_Trend'] = volume_trend
            indicators_df['ATR'] = atr
            indicators_df['ADX'] = adx
            indicators_df['Williams_R'] = williams_r
            indicators_df['CCI'] = cci
            indicators_df['MFI'] = mfi
            indicators_df['OBV'] = obv
            indicators_df['VWAP'] = vwap
            
            return indicators_df
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}")
            return None
    
    def calculate_risk_metrics(self, symbol, df, entry_price):
        """Calculate risk metrics for a trade"""
        if df.empty or len(df) < 20:
            return None
        
        try:
            # Handle both uppercase and lowercase column names
            if 'High' in df.columns:
                high = df['High']
                low = df['Low']
                close = df['Close']
            else:
                high = df['high']
                low = df['low']
                close = df['close']
            
            # Calculate ATR for stop loss
            atr = self.indicators.calculate_atr(high, low, close)
            current_atr = atr.iloc[-1] if not atr.empty else 0
            
            # Calculate support/resistance levels
            support_resistance = self.pattern_recognition.find_support_resistance(df)
            
            # Find nearest support and resistance
            nearest_support = None
            nearest_resistance = None
            
            for support in support_resistance['support']:
                if support['level'] < entry_price:
                    nearest_support = support['level']
                    break
            
            for resistance in support_resistance['resistance']:
                if resistance['level'] > entry_price:
                    nearest_resistance = resistance['level']
                    break
            
            # Risk calculations
            stop_loss = nearest_support if nearest_support else entry_price - (2 * current_atr)
            take_profit_1 = entry_price + (1.5 * current_atr)  # 1.5:1 R/R
            take_profit_2 = nearest_resistance if nearest_resistance else entry_price + (3 * current_atr)
            
            risk_amount = entry_price - stop_loss
            reward_amount = take_profit_2 - entry_price
            risk_reward = reward_amount / risk_amount if risk_amount > 0 else 0
            
            # Calculate win probability based on recent price action
            recent_wins = sum(close.pct_change().tail(20) > 0)
            win_rate = recent_wins / 20
            
            # Kelly Criterion for position sizing
            kelly_fraction = (win_rate * risk_reward - (1 - win_rate)) / risk_reward if risk_reward > 0 else 0
            suggested_position_size = min(max(kelly_fraction * 0.25, 0), 0.1)  # Cap at 10% of portfolio
            
            return {
                'stop_loss': stop_loss,
                'take_profit_1': take_profit_1,
                'take_profit_2': take_profit_2,
                'risk_reward_ratio': risk_reward,
                'win_probability': win_rate,
                'suggested_position_size': suggested_position_size,
                'atr': current_atr,
                'nearest_support': nearest_support,
                'nearest_resistance': nearest_resistance
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return None
    
    def score_opportunity(self, latest_data, historical_data, period='1d'):
        """Enhanced scoring with period-specific analysis, patterns, and AI learning"""
        score = 0
        reasons = []
        signals = {}
        
        # Period-specific multipliers
        period_multipliers = {
            '1d': 1.0,
            '1w': 0.9,
            '1m': 0.8,
            '1y': 0.7
        }
        multiplier = period_multipliers.get(period, 1.0)
        
        # Pattern recognition
        patterns = []
        if len(historical_data) >= 60:
            # Create a DataFrame with both uppercase and lowercase columns for pattern recognition
            # The pattern recognition expects lowercase, so we'll create a copy with lowercase
            pattern_df = pd.DataFrame()
            
            # Map columns appropriately
            if 'Close' in historical_data.columns:
                pattern_df['close'] = historical_data['Close']
                pattern_df['high'] = historical_data['High']
                pattern_df['low'] = historical_data['Low']
                pattern_df['open'] = historical_data['Open']
                pattern_df['volume'] = historical_data['Volume']
            else:
                pattern_df = historical_data[['close', 'high', 'low', 'open', 'volume']].copy()
            
            # Detect chart patterns
            detected_patterns = self.pattern_recognition.detect_chart_patterns(pattern_df)
            if detected_patterns:
                patterns.extend(detected_patterns)
                
                # Add pattern score
                pattern_score = self.pattern_recognition.calculate_pattern_score(
                    detected_patterns, 
                    latest_data['Close']
                )
                score += pattern_score * multiplier
                
                for pattern in detected_patterns:
                    reasons.append(f"{pattern.get('pattern', 'Pattern')} detected with {pattern.get('strength', 0.5)*100:.0f}% confidence")
                    signals[f"pattern_{pattern.get('pattern', 'unknown')}"] = pattern
            
            # Detect divergences
            if 'RSI' in latest_data and not pd.isna(latest_data['RSI']):
                divergences = self.pattern_recognition.detect_divergences(
                    historical_data['Close'] if 'Close' in historical_data.columns else historical_data['close'], 
                    historical_data['RSI']
                )
                if divergences:
                    patterns.extend(divergences)
                    for div in divergences:
                        score += 2 * multiplier if 'bullish' in div['type'] else -2 * multiplier
                        reasons.append(f"{div['type'].replace('_', ' ').title()} detected")
                        signals[div['type']] = True
        
        # RSI Analysis with divergence detection
        if not pd.isna(latest_data.get('RSI')):
            rsi_value = latest_data['RSI']
            signals['rsi'] = rsi_value
            
            # Check for RSI divergence
            if len(historical_data) >= 10:
                recent_prices = historical_data['Close'] if 'Close' in historical_data.columns else historical_data['close']
                recent_rsi = historical_data['RSI'].tail(10)
                
                # Bullish divergence: price makes lower low but RSI makes higher low
                if (recent_prices.iloc[-1] < recent_prices.iloc[-5] and 
                    recent_rsi.iloc[-1] > recent_rsi.iloc[-5] and 
                    rsi_value < 40):
                    score += 3 * multiplier
                    reasons.append("Bullish RSI divergence detected")
                    signals['rsi_divergence'] = 'bullish'
            
            if rsi_value < self.rsi_oversold:
                score += 2 * multiplier
                reasons.append(f"RSI indicates oversold condition ({rsi_value:.1f})")
                signals['rsi_signal'] = 'oversold'
            elif rsi_value > self.rsi_overbought:
                score += 1 * multiplier  # Changed from negative to positive
                reasons.append(f"RSI indicates overbought condition ({rsi_value:.1f})")
                signals['rsi_signal'] = 'overbought'
        
        # Enhanced Moving Average Analysis
        if not pd.isna(latest_data.get('Signal')):
            ma_signal = latest_data['Signal']
            if len(historical_data) > 1:
                prev_signal = historical_data.iloc[-2]['Signal']
                
                # Golden Cross / Death Cross detection
                if 'MA_Short' in latest_data and 'MA_Long' in latest_data:
                    ma_short = latest_data['MA_Short']
                    ma_long = latest_data['MA_Long']
                    
                    if ma_signal == 1 and prev_signal == -1:
                        if ma_long > 0:  # Ensure valid MA values
                            cross_strength = abs(ma_short - ma_long) / ma_long
                            if cross_strength > 0.02:  # Strong crossover
                                score += 4 * multiplier
                                reasons.append("Strong bullish moving average crossover (Golden Cross)")
                                signals['ma_crossover'] = 'golden_cross'
                            else:
                                score += 2 * multiplier
                                reasons.append("Bullish moving average crossover")
                                signals['ma_crossover'] = 'bullish'
                    elif ma_signal == -1 and prev_signal == 1:
                        score -= 2 * multiplier
                        reasons.append("Bearish moving average crossover (Death Cross)")
                        signals['ma_crossover'] = 'death_cross'
        
        # Bollinger Bands Analysis
        if all(key in latest_data for key in ['Close', 'BB_Upper', 'BB_Lower', 'BB_Middle']):
            close = latest_data['Close']
            bb_upper = latest_data['BB_Upper']
            bb_lower = latest_data['BB_Lower']
            bb_middle = latest_data['BB_Middle']
            
            if not pd.isna(bb_upper) and not pd.isna(bb_lower):
                bb_width = (bb_upper - bb_lower) / bb_middle if bb_middle > 0 else 0
                
                # Bollinger Band Squeeze
                if bb_width < 0.1:  # Narrow bands
                    score += 2 * multiplier
                    reasons.append("Bollinger Band squeeze detected - potential breakout incoming")
                    signals['bb_squeeze'] = True
                
                # Price touching bands
                if close <= bb_lower * 1.02:
                    score += 2 * multiplier
                    reasons.append("Price near lower Bollinger Band (potential bounce)")
                    signals['bb_signal'] = 'near_lower'
                elif close >= bb_upper * 0.98:
                    score -= 1 * multiplier
                    reasons.append("Price near upper Bollinger Band (potential reversal)")
                    signals['bb_signal'] = 'near_upper'
        
        # Additional Indicator Analysis
        
        # ADX - Trend Strength
        if 'ADX' in latest_data and not pd.isna(latest_data['ADX']):
            adx_value = latest_data['ADX']
            if adx_value > 25:
                score += 1 * multiplier
                reasons.append(f"Strong trend detected (ADX: {adx_value:.1f})")
                signals['trend_strength'] = 'strong'
            elif adx_value < 20:
                reasons.append(f"Weak trend - potential range-bound market (ADX: {adx_value:.1f})")
                signals['trend_strength'] = 'weak'
        
        # MFI - Money Flow
        if 'MFI' in latest_data and not pd.isna(latest_data['MFI']):
            mfi_value = latest_data['MFI']
            if mfi_value < 20:
                score += 2 * multiplier
                reasons.append(f"Strong buying opportunity - MFI oversold ({mfi_value:.1f})")
                signals['mfi'] = 'oversold'
            elif mfi_value > 80:
                score -= 1 * multiplier
                reasons.append(f"Potential selling pressure - MFI overbought ({mfi_value:.1f})")
                signals['mfi'] = 'overbought'
        
        # CCI - Commodity Channel Index
        if 'CCI' in latest_data and not pd.isna(latest_data['CCI']):
            cci_value = latest_data['CCI']
            if cci_value < -100:
                score += 1.5 * multiplier
                reasons.append(f"CCI indicates oversold condition ({cci_value:.1f})")
                signals['cci'] = 'oversold'
            elif cci_value > 100:
                score -= 1 * multiplier
                reasons.append(f"CCI indicates overbought condition ({cci_value:.1f})")
                signals['cci'] = 'overbought'
        
        # Stochastic Oscillator
        if 'Stoch_K' in latest_data and 'Stoch_D' in latest_data:
            stoch_k = latest_data['Stoch_K']
            stoch_d = latest_data['Stoch_D']
            
            if not pd.isna(stoch_k) and not pd.isna(stoch_d):
                if stoch_k < 20 and stoch_d < 20:
                    score += 2 * multiplier
                    reasons.append(f"Stochastic indicates oversold ({stoch_k:.1f}/{stoch_d:.1f})")
                    signals['stochastic'] = 'oversold'
                elif stoch_k > 80 and stoch_d > 80:
                    score -= 1 * multiplier
                    reasons.append(f"Stochastic indicates overbought ({stoch_k:.1f}/{stoch_d:.1f})")
                    signals['stochastic'] = 'overbought'
        
        # Volume Analysis with Price Action
        if not pd.isna(latest_data.get('Volume_Trend')):
            volume_trend = latest_data['Volume_Trend']
            signals['volume_trend'] = volume_trend
            
            if volume_trend > self.volume_surge_threshold:
                # Check if volume surge aligns with price movement
                if len(historical_data) >= 2:
                    price_change = (latest_data['Close'] - historical_data.iloc[-2]['Close']) / historical_data.iloc[-2]['Close']
                    if price_change > 0.01 and volume_trend > 2.0:
                        score += 3 * multiplier
                        reasons.append(f"Strong volume surge with upward price movement ({volume_trend:.1f}x average)")
                        signals['volume_price_confirmation'] = 'bullish'
                    else:
                        score += 1 * multiplier
                        reasons.append(f"Volume surge detected ({volume_trend:.1f}x average)")
                        signals['volume_signal'] = 'surge'
        
        # MACD Histogram Momentum
        if 'MACD_Histogram' in latest_data:
            macd_hist = latest_data['MACD_Histogram']
            if len(historical_data) >= 5 and not pd.isna(macd_hist):
                hist_trend = historical_data['MACD_Histogram'].tail(5)
                
                # Check if histogram is increasing (momentum building)
                if all(hist_trend.iloc[i] < hist_trend.iloc[i+1] for i in range(len(hist_trend)-1) if not pd.isna(hist_trend.iloc[i])):
                    score += 2 * multiplier
                    reasons.append("MACD histogram shows increasing bullish momentum")
                    signals['macd_momentum'] = 'increasing'
        
        # VWAP Analysis
        if 'VWAP' in latest_data and not pd.isna(latest_data['VWAP']):
            vwap = latest_data['VWAP']
            close = latest_data['Close']
            
            if close > vwap * 1.01:
                score += 1 * multiplier
                reasons.append(f"Price above VWAP - bullish intraday sentiment")
                signals['vwap_position'] = 'above'
            elif close < vwap * 0.99:
                score -= 0.5 * multiplier
                reasons.append(f"Price below VWAP - bearish intraday sentiment")
                signals['vwap_position'] = 'below'
        
        # Multi-timeframe momentum (adjusted for period)
        lookback_days = {
            '1d': (5, 20),
            '1w': (10, 40),
            '1m': (20, 60),
            '1y': (60, 200)
        }
        short_lb, long_lb = lookback_days.get(period, (5, 20))
        
        if len(historical_data) >= long_lb:
            # Short-term momentum
            short_momentum = (latest_data['Close'] - historical_data.iloc[-short_lb]['Close']) / historical_data.iloc[-short_lb]['Close']
            # Long-term momentum
            long_momentum = (latest_data['Close'] - historical_data.iloc[-long_lb]['Close']) / historical_data.iloc[-long_lb]['Close']
            
            momentum_thresholds = {
                '1d': (0.03, 0.05),
                '1w': (0.05, 0.10),
                '1m': (0.10, 0.20),
                '1y': (0.20, 0.50)
            }
            short_thresh, long_thresh = momentum_thresholds.get(period, (0.03, 0.05))
            
            if short_momentum > short_thresh and long_momentum > long_thresh:
                score += 2 * multiplier
                reasons.append(f"Strong multi-timeframe momentum ({short_lb}d: {short_momentum:.1%}, {long_lb}d: {long_momentum:.1%})")
                signals['momentum'] = 'strong_bullish'
            elif short_momentum < -short_thresh and long_momentum < -long_thresh:
                score -= 2 * multiplier
                reasons.append(f"Strong bearish momentum - consider short opportunity")
                signals['momentum'] = 'strong_bearish'
        
        # Apply AI learning adjustments
        adjusted_score = self.learning_engine.adjust_score(score, signals, patterns)
        
        return int(adjusted_score), reasons, signals, patterns
    
    def find_opportunities(self, stocks_data, top_n=10, period='1d'):
        """Find day trading opportunities with enhanced analysis"""
        opportunities = []
        
        for symbol, df in stocks_data.items():
            try:
                # Analyze the stock
                indicators_df = self.analyze_stock(symbol, df, period)
                if indicators_df is None or indicators_df.empty:
                    continue
                
                # Get the latest data
                latest = indicators_df.iloc[-1]
                
                # Score the opportunity
                score, reasons, signals, patterns = self.score_opportunity(latest, indicators_df, period)
                
                # Calculate risk metrics
                risk_metrics = self.calculate_risk_metrics(symbol, df, latest['Close'])
                
                # Generate AI explanation
                ai_explanation = self.ai_analyst.generate_explanation(
                    {'rsi': latest.get('RSI')},
                    signals,
                    score,
                    patterns
                )
                
                # Generate trading strategy
                strategy = self.ai_analyst.generate_trading_strategy(symbol, {
                    'score': score,
                    'signals': signals,
                    'patterns': patterns,
                    'current_price': latest['Close'],
                    'support_entry': risk_metrics['stop_loss'] if risk_metrics else None,
                    'resistance_exit': risk_metrics['take_profit_2'] if risk_metrics else None
                })
                
                # Lower the threshold to show more opportunities
                if score >= 2:  # Changed from > 0 to >= 2 to show moderate opportunities
                    opportunities.append({
                        'ticker': symbol,
                        'score': score,
                        'price': latest['Close'],
                        'rsi': latest.get('RSI'),
                        'volatility': latest.get('Volatility'),
                        'volume_trend': latest.get('Volume_Trend'),
                        'ma_signal': latest.get('Signal'),
                        'reasons': reasons,
                        'signals': signals,
                        'patterns': patterns,
                        'risk_metrics': risk_metrics,
                        'ai_explanation': ai_explanation,
                        'strategy': strategy,
                        'period': period,
                        'last_updated': datetime.now().isoformat()
                    })
                    
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                continue
        
        # Sort by score and return top N
        opportunities.sort(key=lambda x: x['score'], reverse=True)
        
        return pd.DataFrame(opportunities[:top_n]) if opportunities else pd.DataFrame()
    
    def get_entry_exit_points(self, symbol, indicators_df):
        """Calculate suggested entry and exit points with enhanced logic"""
        if indicators_df.empty:
            return None
        
        latest = indicators_df.iloc[-1]
        
        # Calculate Fibonacci levels
        recent_high = indicators_df['High'].tail(20).max()
        recent_low = indicators_df['Low'].tail(20).min()
        fib_levels = self.indicators.calculate_fibonacci_levels(recent_high, recent_low)
        
        # Calculate pivot points
        pivot_data = self.indicators.calculate_pivot_points(
            latest['High'], 
            latest['Low'], 
            latest['Close']
        )
        
        # Create a DataFrame with proper column names for pattern recognition
        pattern_df = indicators_df[['High', 'Low', 'Close', 'Open', 'Volume']].rename(columns={
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Open': 'open',
            'Volume': 'volume'
        })
        
        # Get support and resistance from pattern recognition
        support_resistance = self.pattern_recognition.find_support_resistance(pattern_df)
        
        suggestions = {
            'symbol': symbol,
            'current_price': latest['Close'],
            'fibonacci_levels': fib_levels,
            'pivot_points': pivot_data,
            'support_resistance': support_resistance
        }
        
        # Entry points based on multiple factors
        if support_resistance['support']:
            suggestions['support_entry'] = support_resistance['support'][0]['level']
        elif 'BB_Lower' in latest and not pd.isna(latest['BB_Lower']):
            suggestions['support_entry'] = latest['BB_Lower']
        else:
            suggestions['support_entry'] = pivot_data['s1']
        
        # Exit points
        if support_resistance['resistance']:
            suggestions['resistance_exit'] = support_resistance['resistance'][0]['level']
        elif 'BB_Upper' in latest and not pd.isna(latest['BB_Upper']):
            suggestions['resistance_exit'] = latest['BB_Upper']
        else:
            suggestions['resistance_exit'] = pivot_data['r1']
        
        # Dynamic stop loss based on ATR
        if 'ATR' in latest and not pd.isna(latest['ATR']):
            suggestions['stop_loss'] = latest['Close'] - (2 * latest['ATR'])
            suggestions['trailing_stop'] = latest['ATR'] * 1.5
        else:
            suggestions['stop_loss'] = latest['Close'] * 0.98
            suggestions['trailing_stop'] = latest['Close'] * 0.02
        
        # Multiple take profit targets
        suggestions['target_1'] = latest['Close'] * 1.02  # 2% gain
        suggestions['target_2'] = latest['Close'] * 1.05  # 5% gain
        suggestions['target_3'] = pivot_data['r2']  # Second resistance
        
        return suggestions
    
    def backtest_strategy(self, symbol, df, strategy_params):
        """Simple backtesting function for strategy validation"""
        if df.empty or len(df) < 100:
            return None
        
        try:
            # Initialize tracking variables
            trades = []
            position = None
            capital = 10000
            shares = 0
            
            # Analyze historical data
            indicators_df = self.analyze_stock(symbol, df)
            if indicators_df is None:
                return None
            
            for i in range(50, len(indicators_df)):
                current = indicators_df.iloc[i]
                historical = indicators_df.iloc[:i]
                
                # Get signals
                score, reasons, signals, patterns = self.score_opportunity(current, historical)
                
                # Entry logic
                if position is None and score >= strategy_params.get('min_score', 5):
                    position = {
                        'entry_price': current['Close'],
                        'entry_date': indicators_df.index[i],
                        'shares': int(capital * 0.95 / current['Close']),
                        'stop_loss': current['Close'] * (1 - strategy_params.get('stop_loss_pct', 0.02)),
                        'take_profit': current['Close'] * (1 + strategy_params.get('take_profit_pct', 0.05))
                    }
                    capital -= position['shares'] * position['entry_price']
                
                # Exit logic
                elif position is not None:
                    exit_price = None
                    exit_reason = None
                    
                    # Check stop loss
                    if current['Low'] <= position['stop_loss']:
                        exit_price = position['stop_loss']
                        exit_reason = 'stop_loss'
                    # Check take profit
                    elif current['High'] >= position['take_profit']:
                        exit_price = position['take_profit']
                        exit_reason = 'take_profit'
                    # Check exit signals
                    elif score <= -strategy_params.get('exit_score', -3):
                        exit_price = current['Close']
                        exit_reason = 'signal'
                    
                    if exit_price:
                        capital += position['shares'] * exit_price
                        trade_return = (exit_price - position['entry_price']) / position['entry_price']
                        
                        trades.append({
                            'entry_date': position['entry_date'],
                            'exit_date': indicators_df.index[i],
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'shares': position['shares'],
                            'return': trade_return,
                            'exit_reason': exit_reason
                        })
                        
                        position = None
            
            # Calculate performance metrics
            if trades:
                returns = [t['return'] for t in trades]
                winning_trades = [r for r in returns if r > 0]
                losing_trades = [r for r in returns if r < 0]
                
                total_return = (capital - 10000) / 10000
                win_rate = len(winning_trades) / len(trades)
                avg_win = np.mean(winning_trades) if winning_trades else 0
                avg_loss = np.mean(losing_trades) if losing_trades else 0
                
                # Calculate Sharpe ratio
                if len(returns) > 1:
                    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
                else:
                    sharpe_ratio = 0
                
                # Calculate max drawdown
                cumulative_returns = np.cumprod(1 + np.array(returns))
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdowns = (cumulative_returns - running_max) / running_max
                max_drawdown = np.min(drawdowns)
                
                return {
                    'symbol': symbol,
                    'total_trades': len(trades),
                    'winning_trades': len(winning_trades),
                    'losing_trades': len(losing_trades),
                    'win_rate': win_rate,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'total_return': total_return,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'trades': trades
                }
            else:
                return {
                    'symbol': symbol,
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0,
                    'total_return': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'trades': []
                }
                
        except Exception as e:
            logger.error(f"Error backtesting {symbol}: {e}")
            return None