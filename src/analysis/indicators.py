# src/analysis/indicators.py
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class StockIndicators:
    """Calculate various technical indicators for stock analysis"""
    
    def __init__(self):
        self.default_periods = {
            'rsi': 14,
            'ma_short': 10,
            'ma_long': 20,
            'bb_period': 20,
            'bb_std': 2,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'stoch_period': 14,
            'atr_period': 14,
            'adx_period': 14
        }
    
    def calculate_rsi(self, prices, period=None):
        """Calculate Relative Strength Index"""
        if period is None:
            period = self.default_periods['rsi']
        
        if len(prices) < period + 1:
            return pd.Series(index=prices.index, dtype=float)
        
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=period, min_periods=period).mean()
        avg_losses = losses.rolling(window=period, min_periods=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        # Handle division by zero
        rsi = rsi.fillna(50)
        
        return rsi
    
    def calculate_moving_averages(self, prices, short_period=None, long_period=None):
        """Calculate simple moving averages and generate signals"""
        if short_period is None:
            short_period = self.default_periods['ma_short']
        if long_period is None:
            long_period = self.default_periods['ma_long']
        
        ma_short = prices.rolling(window=short_period, min_periods=1).mean()
        ma_long = prices.rolling(window=long_period, min_periods=1).mean()
        
        # Generate trading signals
        signal = pd.Series(index=prices.index, dtype=float)
        signal[ma_short > ma_long] = 1
        signal[ma_short < ma_long] = -1
        signal[ma_short == ma_long] = 0
        
        return pd.DataFrame({
            'MA_Short': ma_short,
            'MA_Long': ma_long,
            'Signal': signal
        })
    
    def calculate_bollinger_bands(self, prices, period=None, std_dev=None):
        """Calculate Bollinger Bands"""
        if period is None:
            period = self.default_periods['bb_period']
        if std_dev is None:
            std_dev = self.default_periods['bb_std']
        
        sma = prices.rolling(window=period, min_periods=1).mean()
        std = prices.rolling(window=period, min_periods=1).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return pd.DataFrame({
            'BB_Middle': sma,
            'BB_Upper': upper_band,
            'BB_Lower': lower_band,
            'BB_Width': upper_band - lower_band,
            'BB_Percent': (prices - lower_band) / (upper_band - lower_band)
        })
    
    def calculate_macd(self, prices, fast_period=None, slow_period=None, signal_period=None):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        if fast_period is None:
            fast_period = self.default_periods['macd_fast']
        if slow_period is None:
            slow_period = self.default_periods['macd_slow']
        if signal_period is None:
            signal_period = self.default_periods['macd_signal']
        
        # Calculate exponential moving averages
        ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
        ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'MACD': macd_line,
            'MACD_Signal': signal_line,
            'MACD_Histogram': histogram
        })
    
    def calculate_stochastic(self, high, low, close, period=None, smooth_k=3, smooth_d=3):
        """Calculate Stochastic Oscillator"""
        if period is None:
            period = self.default_periods['stoch_period']
        
        # Calculate %K
        lowest_low = low.rolling(window=period, min_periods=1).min()
        highest_high = high.rolling(window=period, min_periods=1).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        k_percent = k_percent.fillna(50)
        
        # Smooth %K
        k_smooth = k_percent.rolling(window=smooth_k, min_periods=1).mean()
        
        # Calculate %D (smoothed %K)
        d_percent = k_smooth.rolling(window=smooth_d, min_periods=1).mean()
        
        return pd.DataFrame({
            'Stoch_K': k_smooth,
            'Stoch_D': d_percent
        })
    
    def calculate_atr(self, high, low, close, period=None):
        """Calculate Average True Range"""
        if period is None:
            period = self.default_periods['atr_period']
        
        # Calculate True Range
        high_low = high - low
        high_close = abs(high - close.shift(1))
        low_close = abs(low - close.shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = true_range.rolling(window=period, min_periods=1).mean()
        
        return atr
    
    def calculate_volatility(self, prices, period=20):
        """Calculate price volatility (standard deviation of returns)"""
        returns = prices.pct_change()
        volatility = returns.rolling(window=period, min_periods=1).std()
        return volatility
    
    def calculate_volume_trend(self, volume, period=20):
        """Calculate volume trend (current volume vs average)"""
        avg_volume = volume.rolling(window=period, min_periods=1).mean()
        volume_trend = volume / avg_volume
        return volume_trend.fillna(1)
    
    def calculate_adx(self, high, low, close, period=None):
        """Calculate Average Directional Index"""
        if period is None:
            period = self.default_periods['adx_period']
        
        # Calculate directional movement
        high_diff = high.diff()
        low_diff = -low.diff()
        
        # Positive directional movement
        pos_dm = pd.Series(index=high.index, dtype=float)
        pos_dm[(high_diff > low_diff) & (high_diff > 0)] = high_diff
        pos_dm = pos_dm.fillna(0)
        
        # Negative directional movement
        neg_dm = pd.Series(index=low.index, dtype=float)
        neg_dm[(low_diff > high_diff) & (low_diff > 0)] = low_diff
        neg_dm = neg_dm.fillna(0)
        
        # Calculate ATR
        atr = self.calculate_atr(high, low, close, period)
        
        # Calculate directional indicators
        pos_di = 100 * (pos_dm.rolling(window=period).mean() / atr)
        neg_di = 100 * (neg_dm.rolling(window=period).mean() / atr)
        
        # Calculate DX
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
        
        # Calculate ADX
        adx = dx.rolling(window=period).mean()
        
        return adx.fillna(0)
    
    def calculate_williams_r(self, high, low, close, period=14):
        """Calculate Williams %R"""
        highest_high = high.rolling(window=period, min_periods=1).max()
        lowest_low = low.rolling(window=period, min_periods=1).min()
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        
        return williams_r.fillna(-50)
    
    def calculate_cci(self, high, low, close, period=20):
        """Calculate Commodity Channel Index"""
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=period, min_periods=1).mean()
        mean_deviation = typical_price.rolling(window=period, min_periods=1).apply(
            lambda x: np.mean(np.abs(x - np.mean(x)))
        )
        
        cci = (typical_price - sma) / (0.015 * mean_deviation)
        
        return cci.fillna(0)
    
    def calculate_mfi(self, high, low, close, volume, period=14):
        """Calculate Money Flow Index"""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        # Determine positive and negative money flow
        price_diff = typical_price.diff()
        
        positive_flow = pd.Series(index=high.index, dtype=float)
        negative_flow = pd.Series(index=high.index, dtype=float)
        
        positive_flow[price_diff > 0] = money_flow[price_diff > 0]
        negative_flow[price_diff < 0] = money_flow[price_diff < 0]
        
        positive_flow = positive_flow.fillna(0)
        negative_flow = negative_flow.fillna(0)
        
        # Calculate money flow ratio
        positive_mf = positive_flow.rolling(window=period, min_periods=1).sum()
        negative_mf = negative_flow.rolling(window=period, min_periods=1).sum()
        
        mf_ratio = positive_mf / negative_mf
        mfi = 100 - (100 / (1 + mf_ratio))
        
        return mfi.fillna(50)
    
    def calculate_obv(self, close, volume):
        """Calculate On-Balance Volume"""
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def calculate_vwap(self, high, low, close, volume):
        """Calculate Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        cumulative_tp_volume = (typical_price * volume).cumsum()
        cumulative_volume = volume.cumsum()
        
        vwap = cumulative_tp_volume / cumulative_volume
        
        return vwap.fillna(typical_price)
    
    def calculate_fibonacci_levels(self, high_price, low_price):
        """Calculate Fibonacci retracement levels"""
        diff = high_price - low_price
        
        levels = {
            '0.0%': high_price,
            '23.6%': high_price - (diff * 0.236),
            '38.2%': high_price - (diff * 0.382),
            '50.0%': high_price - (diff * 0.5),
            '61.8%': high_price - (diff * 0.618),
            '78.6%': high_price - (diff * 0.786),
            '100.0%': low_price
        }
        
        return levels
    
    def calculate_pivot_points(self, high, low, close):
        """Calculate pivot points for support and resistance"""
        pivot = (high + low + close) / 3
        
        r1 = (2 * pivot) - low
        s1 = (2 * pivot) - high
        
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        
        r3 = high + 2 * (pivot - low)
        s3 = low - 2 * (high - pivot)
        
        return {
            'pivot': pivot,
            'r1': r1,
            'r2': r2,
            'r3': r3,
            's1': s1,
            's2': s2,
            's3': s3
        }