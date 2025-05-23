# src/models/predictor.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json

logger = logging.getLogger(__name__)

class StockIndicators:
    """Calculate technical indicators for stock analysis"""
    
    @staticmethod
    def calculate_rsi(data, period=14):
        """Calculate Relative Strength Index"""
        if len(data) < period + 1:
            return pd.Series(index=data.index, dtype=float)
        
        delta = data.diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        avg_gains = gains.rolling(window=period, min_periods=period).mean()
        avg_losses = losses.rolling(window=period, min_periods=period).mean()
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_moving_averages(data, short_period=10, long_period=20):
        """Calculate simple moving averages and generate signals"""
        if len(data) < long_period:
            return pd.DataFrame(index=data.index)
        
        ma_short = data.rolling(window=short_period, min_periods=short_period).mean()
        ma_long = data.rolling(window=long_period, min_periods=long_period).mean()
        
        signals = pd.Series(index=data.index, dtype=float)
        signals[ma_short > ma_long] = 1
        signals[ma_short <= ma_long] = -1
        
        return pd.DataFrame({
            'MA_Short': ma_short,
            'MA_Long': ma_long,
            'Signal': signals
        })
    
    @staticmethod
    def calculate_volatility(data, period=20):
        """Calculate price volatility (standard deviation of returns)"""
        if len(data) < period + 1:
            return pd.Series(index=data.index, dtype=float)
        
        returns = data.pct_change()
        volatility = returns.rolling(window=period, min_periods=period).std()
        
        return volatility
    
    @staticmethod
    def calculate_volume_trend(volume_data, period=10):
        """Calculate volume trend (current vs average)"""
        if len(volume_data) < period:
            return pd.Series(index=volume_data.index, dtype=float)
        
        avg_volume = volume_data.rolling(window=period, min_periods=period).mean()
        volume_ratio = volume_data / avg_volume
        
        return volume_ratio
    
    @staticmethod
    def calculate_bollinger_bands(data, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        if len(data) < period:
            return pd.DataFrame(index=data.index)
        
        sma = data.rolling(window=period, min_periods=period).mean()
        std = data.rolling(window=period, min_periods=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return pd.DataFrame({
            'BB_Middle': sma,
            'BB_Upper': upper_band,
            'BB_Lower': lower_band
        })
    
    @staticmethod
    def calculate_macd(data, fast=12, slow=26, signal=9):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        if len(data) < slow + signal:
            return pd.DataFrame(index=data.index)
        
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'MACD': macd_line,
            'MACD_Signal': signal_line,
            'MACD_Histogram': histogram
        })
    
    @staticmethod
    def calculate_stochastic(high, low, close, k_period=14, d_period=3):
        """Calculate Stochastic Oscillator"""
        if len(close) < k_period:
            return pd.DataFrame(index=close.index)
        
        lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
        highest_high = high.rolling(window=k_period, min_periods=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period, min_periods=d_period).mean()
        
        return pd.DataFrame({
            'Stoch_K': k_percent,
            'Stoch_D': d_percent
        })
    
    @staticmethod
    def calculate_atr(high, low, close, period=14):
        """Calculate Average True Range"""
        if len(close) < period + 1:
            return pd.Series(index=close.index, dtype=float)
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=period).mean()
        
        return atr
    
    @staticmethod
    def calculate_fibonacci_levels(high, low):
        """Calculate Fibonacci retracement levels"""
        diff = high - low
        
        levels = {
            'high': high,
            'fib_786': high - (diff * 0.786),
            'fib_618': high - (diff * 0.618),
            'fib_500': high - (diff * 0.500),
            'fib_382': high - (diff * 0.382),
            'fib_236': high - (diff * 0.236),
            'low': low
        }
        
        return levels

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
            'pattern_detected': "A chart pattern has been detected that may indicate future price movement."
        }
    
    def generate_explanation(self, indicators, signals, score):
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
        
        return "\n".join(explanation)
    
    def generate_trading_strategy(self, symbol, analysis_data):
        """Generate AI-powered trading strategy suggestions"""
        strategy = {
            'symbol': symbol,
            'recommendation': '',
            'entry_strategy': '',
            'exit_strategy': '',
            'risk_management': '',
            'time_horizon': ''
        }
        
        score = analysis_data.get('score', 0)
        signals = analysis_data.get('signals', {})
        current_price = analysis_data.get('current_price', 0)
        
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

class DayTradePredictor:
    """Enhanced day trading predictor with AI capabilities"""
    
    def __init__(self):
        """Initialize with default parameters"""
        self.indicators = StockIndicators()
        self.ai_analyst = AIAnalyst()
        
        # Trading parameters
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.min_volatility = 0.01
        self.volume_surge_threshold = 1.5
        
        # Risk parameters
        self.max_risk_per_trade = 0.02  # 2% max risk
        self.risk_reward_ratio = 2.0    # Minimum 2:1 reward/risk
    
    def analyze_stock(self, symbol, df):
        """Analyze a single stock and calculate all indicators"""
        if df.empty or len(df) < 30:
            return None
        
        try:
            if not all(col in df.columns for col in ['close', 'volume', 'high', 'low']):
                return None
            
            close_prices = df['close']
            volume = df['volume']
            high_prices = df['high']
            low_prices = df['low']
            
            # Calculate all indicators
            rsi = self.indicators.calculate_rsi(close_prices)
            ma_data = self.indicators.calculate_moving_averages(close_prices)
            volatility = self.indicators.calculate_volatility(close_prices)
            volume_trend = self.indicators.calculate_volume_trend(volume)
            bb_data = self.indicators.calculate_bollinger_bands(close_prices)
            macd_data = self.indicators.calculate_macd(close_prices)
            stoch_data = self.indicators.calculate_stochastic(high_prices, low_prices, close_prices)
            atr = self.indicators.calculate_atr(high_prices, low_prices, close_prices)
            
            # Combine all indicators
            indicators_df = pd.DataFrame(index=df.index)
            indicators_df['Close'] = close_prices
            indicators_df['Volume'] = volume
            indicators_df['High'] = high_prices
            indicators_df['Low'] = low_prices
            indicators_df['RSI'] = rsi
            indicators_df = pd.concat([indicators_df, ma_data, bb_data, macd_data, stoch_data], axis=1)
            indicators_df['Volatility'] = volatility
            indicators_df['Volume_Trend'] = volume_trend
            indicators_df['ATR'] = atr
            
            return indicators_df
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}")
            return None
    
    def calculate_risk_metrics(self, symbol, df, entry_price):
        """Calculate risk metrics for a trade"""
        if df.empty or len(df) < 20:
            return None
        
        try:
            # Calculate ATR for stop loss
            atr = self.indicators.calculate_atr(df['high'], df['low'], df['close'])
            current_atr = atr.iloc[-1] if not atr.empty else 0
            
            # Calculate support/resistance levels
            recent_high = df['high'].tail(20).max()
            recent_low = df['low'].tail(20).min()
            
            # Risk calculations
            stop_loss = entry_price - (2 * current_atr)  # 2 ATR stop
            take_profit_1 = entry_price + (2 * current_atr)  # 1:1 R/R
            take_profit_2 = entry_price + (4 * current_atr)  # 2:1 R/R
            
            risk_amount = entry_price - stop_loss
            reward_amount = take_profit_2 - entry_price
            risk_reward = reward_amount / risk_amount if risk_amount > 0 else 0
            
            # Calculate win probability based on recent price action
            recent_wins = sum(df['close'].pct_change().tail(20) > 0)
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
                'atr': current_atr
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return None
    
    def score_opportunity(self, latest_data, historical_data):
        """Enhanced scoring with more sophisticated analysis"""
        score = 0
        reasons = []
        signals = {}
        
        # RSI Analysis with divergence detection
        if not pd.isna(latest_data.get('RSI')):
            rsi_value = latest_data['RSI']
            signals['rsi'] = rsi_value
            
            # Check for RSI divergence
            if len(historical_data) >= 10:
                recent_prices = historical_data['Close'].tail(10)
                recent_rsi = historical_data['RSI'].tail(10)
                
                # Bullish divergence: price makes lower low but RSI makes higher low
                if (recent_prices.iloc[-1] < recent_prices.iloc[-5] and 
                    recent_rsi.iloc[-1] > recent_rsi.iloc[-5] and 
                    rsi_value < 40):
                    score += 3
                    reasons.append("Bullish RSI divergence detected")
                    signals['rsi_divergence'] = 'bullish'
            
            if rsi_value < self.rsi_oversold:
                score += 2
                reasons.append(f"RSI indicates oversold condition ({rsi_value:.1f})")
                signals['rsi_signal'] = 'oversold'
            elif rsi_value > self.rsi_overbought:
                score += 1
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
                                score += 4
                                reasons.append("Strong bullish moving average crossover (Golden Cross)")
                                signals['ma_crossover'] = 'golden_cross'
                            else:
                                score += 2
                                reasons.append("Bullish moving average crossover")
                                signals['ma_crossover'] = 'bullish'
                    elif ma_signal == -1 and prev_signal == 1:
                        score += 2
                        reasons.append("Bearish moving average crossover (Death Cross)")
                        signals['ma_crossover'] = 'death_cross'
        
        # Bollinger Bands Squeeze Detection
        if all(key in latest_data for key in ['Close', 'BB_Upper', 'BB_Lower', 'BB_Middle']):
            close = latest_data['Close']
            bb_upper = latest_data['BB_Upper']
            bb_lower = latest_data['BB_Lower']
            bb_middle = latest_data['BB_Middle']
            
            if not pd.isna(bb_upper) and not pd.isna(bb_lower):
                bb_width = (bb_upper - bb_lower) / bb_middle if bb_middle > 0 else 0
                
                # Bollinger Band Squeeze
                if bb_width < 0.1:  # Narrow bands
                    score += 2
                    reasons.append("Bollinger Band squeeze detected - potential breakout incoming")
                    signals['bb_squeeze'] = True
                
                # Price touching bands
                if close <= bb_lower * 1.02:
                    score += 2
                    reasons.append("Price near lower Bollinger Band (potential bounce)")
                    signals['bb_signal'] = 'near_lower'
                elif close >= bb_upper * 0.98:
                    score += 1
                    reasons.append("Price near upper Bollinger Band (potential reversal)")
                    signals['bb_signal'] = 'near_upper'
        
        # Stochastic Oscillator
        if 'Stoch_K' in latest_data and 'Stoch_D' in latest_data:
            stoch_k = latest_data['Stoch_K']
            stoch_d = latest_data['Stoch_D']
            
            if not pd.isna(stoch_k) and not pd.isna(stoch_d):
                if stoch_k < 20 and stoch_d < 20:
                    score += 2
                    reasons.append(f"Stochastic indicates oversold ({stoch_k:.1f}/{stoch_d:.1f})")
                    signals['stochastic'] = 'oversold'
                elif stoch_k > 80 and stoch_d > 80:
                    score += 1
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
                        score += 3
                        reasons.append(f"Strong volume surge with upward price movement ({volume_trend:.1f}x average)")
                        signals['volume_price_confirmation'] = 'bullish'
                    else:
                        score += 1
                        reasons.append(f"Volume surge detected ({volume_trend:.1f}x average)")
                        signals['volume_signal'] = 'surge'
        
        # MACD Histogram Momentum
        if 'MACD_Histogram' in latest_data:
            macd_hist = latest_data['MACD_Histogram']
            if len(historical_data) >= 5 and not pd.isna(macd_hist):
                hist_trend = historical_data['MACD_Histogram'].tail(5)
                
                # Check if histogram is increasing (momentum building)
                if all(hist_trend.iloc[i] < hist_trend.iloc[i+1] for i in range(len(hist_trend)-1) if not pd.isna(hist_trend.iloc[i])):
                    score += 2
                    reasons.append("MACD histogram shows increasing bullish momentum")
                    signals['macd_momentum'] = 'increasing'
        
        # Multi-timeframe momentum
        if len(historical_data) >= 20:
            # Short-term momentum (5 days)
            short_momentum = (latest_data['Close'] - historical_data.iloc[-5]['Close']) / historical_data.iloc[-5]['Close']
            # Medium-term momentum (20 days)
            medium_momentum = (latest_data['Close'] - historical_data.iloc[-20]['Close']) / historical_data.iloc[-20]['Close']
            
            if short_momentum > 0.03 and medium_momentum > 0.05:
                score += 2
                reasons.append(f"Strong multi-timeframe momentum (5d: {short_momentum:.1%}, 20d: {medium_momentum:.1%})")
                signals['momentum'] = 'strong_bullish'
            elif short_momentum < -0.03 and medium_momentum < -0.05:
                score += 1
                reasons.append(f"Strong bearish momentum - consider short opportunity")
                signals['momentum'] = 'strong_bearish'
        
        return score, reasons, signals
    
    def find_opportunities(self, stocks_data, top_n=10):
        """Find day trading opportunities with enhanced analysis"""
        opportunities = []
        
        for symbol, df in stocks_data.items():
            try:
                # Analyze the stock
                indicators_df = self.analyze_stock(symbol, df)
                if indicators_df is None or indicators_df.empty:
                    continue
                
                # Get the latest data
                latest = indicators_df.iloc[-1]
                
                # Score the opportunity
                score, reasons, signals = self.score_opportunity(latest, indicators_df)
                
                # Calculate risk metrics
                risk_metrics = self.calculate_risk_metrics(symbol, df, latest['Close'])
                
                # Generate AI explanation
                ai_explanation = self.ai_analyst.generate_explanation(
                    {'rsi': latest.get('RSI')},
                    signals,
                    score
                )
                
                # Generate trading strategy
                strategy = self.ai_analyst.generate_trading_strategy(symbol, {
                    'score': score,
                    'signals': signals,
                    'current_price': latest['Close'],
                    'support_entry': risk_metrics['stop_loss'] if risk_metrics else None,
                    'resistance_exit': risk_metrics['take_profit_2'] if risk_metrics else None
                })
                
                if score > 0:
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
                        'risk_metrics': risk_metrics,
                        'ai_explanation': ai_explanation,
                        'strategy': strategy,
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
        pivot = (latest['High'] + latest['Low'] + latest['Close']) / 3
        r1 = 2 * pivot - latest['Low']
        r2 = pivot + (latest['High'] - latest['Low'])
        s1 = 2 * pivot - latest['High']
        s2 = pivot - (latest['High'] - latest['Low'])
        
        suggestions = {
            'symbol': symbol,
            'current_price': latest['Close'],
            'fibonacci_levels': fib_levels,
            'pivot_points': {
                'pivot': pivot,
                'resistance_1': r1,
                'resistance_2': r2,
                'support_1': s1,
                'support_2': s2
            }
        }
        
        # Entry points based on multiple factors
        if 'BB_Lower' in latest and not pd.isna(latest['BB_Lower']):
            suggestions['support_entry'] = latest['BB_Lower']
        else:
            suggestions['support_entry'] = s1
        
        # Exit points
        if 'BB_Upper' in latest and not pd.isna(latest['BB_Upper']):
            suggestions['resistance_exit'] = latest['BB_Upper']
        else:
            suggestions['resistance_exit'] = r1
        
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
        suggestions['target_3'] = r2  # Second resistance
        
        return suggestions
    
    def backtest_strategy(self, symbol, df, strategy_params):
        """Simple backtesting function for strategy validation"""
        # This is a placeholder for backtesting functionality
        # In production, implement comprehensive backtesting
        return {
            'symbol': symbol,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0
        }