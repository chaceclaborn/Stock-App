# src/analysis/pattern_recognition.py
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import logging

logger = logging.getLogger(__name__)

class PatternRecognition:
    """Advanced pattern recognition for technical analysis"""
    
    def __init__(self):
        self.min_pattern_length = 20
        self.pattern_threshold = 0.02  # 2% price movement threshold
    
    def normalize_dataframe(self, df):
        """Normalize DataFrame to have lowercase column names"""
        if df.empty:
            return df
        
        # Create a copy to avoid modifying the original
        normalized_df = df.copy()
        
        # Map columns to lowercase if they're uppercase
        column_mapping = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        
        # Rename columns if they exist
        for old_name, new_name in column_mapping.items():
            if old_name in normalized_df.columns and new_name not in normalized_df.columns:
                normalized_df.rename(columns={old_name: new_name}, inplace=True)
        
        return normalized_df
    
    def find_support_resistance(self, df, window=20, num_touches=2):
        """Find support and resistance levels using multiple methods"""
        # Normalize the dataframe
        df = self.normalize_dataframe(df)
        
        if df.empty or len(df) < window:
            return {'support': [], 'resistance': []}
        
        levels = {'support': [], 'resistance': []}
        
        try:
            # Method 1: Local minima and maxima
            highs = df['high'].values
            lows = df['low'].values
            closes = df['close'].values
            
            # Find local maxima and minima
            max_indices = argrelextrema(highs, np.greater, order=window//2)[0]
            min_indices = argrelextrema(lows, np.less, order=window//2)[0]
            
            # Method 2: High volume areas
            if 'volume' in df.columns:
                volume = df['volume'].values
                high_volume_indices = np.where(volume > np.percentile(volume, 75))[0]
                
                # Combine with price levels
                for idx in high_volume_indices:
                    if idx in max_indices:
                        max_indices = np.append(max_indices, idx)
                    if idx in min_indices:
                        min_indices = np.append(min_indices, idx)
            
            # Process resistance levels
            for idx in max_indices:
                if idx < len(highs):
                    level = highs[idx]
                    strength = self._calculate_level_strength(df, level, 'resistance')
                    if strength >= num_touches:
                        levels['resistance'].append({
                            'level': level,
                            'strength': strength,
                            'index': idx
                        })
            
            # Process support levels
            for idx in min_indices:
                if idx < len(lows):
                    level = lows[idx]
                    strength = self._calculate_level_strength(df, level, 'support')
                    if strength >= num_touches:
                        levels['support'].append({
                            'level': level,
                            'strength': strength,
                            'index': idx
                        })
            
            # Method 3: Psychological levels (round numbers)
            current_price = closes[-1]
            round_levels = self._find_round_levels(current_price)
            
            for level in round_levels:
                if level > current_price:
                    levels['resistance'].append({
                        'level': level,
                        'strength': 1,
                        'type': 'psychological'
                    })
                else:
                    levels['support'].append({
                        'level': level,
                        'strength': 1,
                        'type': 'psychological'
                    })
            
            # Sort and remove duplicates
            levels['support'] = self._remove_duplicate_levels(levels['support'])
            levels['resistance'] = self._remove_duplicate_levels(levels['resistance'])
            
            # Sort by proximity to current price
            levels['support'].sort(key=lambda x: x['level'], reverse=True)
            levels['resistance'].sort(key=lambda x: x['level'])
            
        except Exception as e:
            logger.error(f"Error finding support/resistance: {e}")
        
        return levels
    
    def _calculate_level_strength(self, df, level, level_type, tolerance=0.02):
        """Calculate how many times price has touched a level"""
        df = self.normalize_dataframe(df)
        
        touches = 0
        
        if level_type == 'support':
            # Count times low was near this level
            near_level = np.abs(df['low'] - level) / level < tolerance
            touches = near_level.sum()
        else:
            # Count times high was near this level
            near_level = np.abs(df['high'] - level) / level < tolerance
            touches = near_level.sum()
        
        return touches
    
    def _find_round_levels(self, price, num_levels=5):
        """Find psychological round number levels"""
        # Determine the scale
        if price > 1000:
            round_to = 50
        elif price > 100:
            round_to = 10
        elif price > 10:
            round_to = 1
        else:
            round_to = 0.5
        
        base_level = round(price / round_to) * round_to
        levels = []
        
        for i in range(-num_levels//2, num_levels//2 + 1):
            level = base_level + (i * round_to)
            if level > 0:
                levels.append(level)
        
        return levels
    
    def _remove_duplicate_levels(self, levels, tolerance=0.01):
        """Remove duplicate levels within tolerance"""
        if not levels:
            return levels
        
        unique_levels = []
        for level in levels:
            is_duplicate = False
            for unique in unique_levels:
                if abs(level['level'] - unique['level']) / unique['level'] < tolerance:
                    # Keep the one with higher strength
                    if level.get('strength', 0) > unique.get('strength', 0):
                        unique_levels.remove(unique)
                        unique_levels.append(level)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_levels.append(level)
        
        return unique_levels
    
    def detect_chart_patterns(self, df, min_pattern_bars=20):
        """Detect various chart patterns"""
        df = self.normalize_dataframe(df)
        
        patterns = []
        
        if df.empty or len(df) < min_pattern_bars:
            return patterns
        
        try:
            # Detect different patterns
            patterns.extend(self._detect_head_and_shoulders(df))
            patterns.extend(self._detect_double_pattern(df))
            patterns.extend(self._detect_triangle_patterns(df))
            patterns.extend(self._detect_flag_patterns(df))
            patterns.extend(self._detect_wedge_patterns(df))
            
        except Exception as e:
            logger.error(f"Error detecting chart patterns: {e}")
        
        return patterns
    
    def _detect_head_and_shoulders(self, df):
        """Detect head and shoulders patterns"""
        df = self.normalize_dataframe(df)
        patterns = []
        
        if len(df) < 35:
            return patterns
        
        try:
            highs = df['high'].values
            lows = df['low'].values
            
            # Find peaks and troughs
            peaks = argrelextrema(highs, np.greater, order=5)[0]
            troughs = argrelextrema(lows, np.less, order=5)[0]
            
            # Look for head and shoulders pattern
            for i in range(2, len(peaks) - 2):
                if i < len(peaks) - 2:
                    left_shoulder = highs[peaks[i-1]]
                    head = highs[peaks[i]]
                    right_shoulder = highs[peaks[i+1]]
                    
                    # Check if it forms a head and shoulders
                    if (head > left_shoulder * 1.02 and 
                        head > right_shoulder * 1.02 and
                        abs(left_shoulder - right_shoulder) / left_shoulder < 0.03):
                        
                        # Find neckline
                        neckline_start = min(troughs[troughs > peaks[i-1]][0], troughs[troughs < peaks[i]][0]) if len(troughs[troughs > peaks[i-1]]) > 0 and len(troughs[troughs < peaks[i]]) > 0 else peaks[i]
                        neckline_end = min(troughs[troughs > peaks[i]][0], troughs[troughs < peaks[i+1]][0]) if len(troughs[troughs > peaks[i]]) > 0 and len(troughs[troughs < peaks[i+1]]) > 0 else peaks[i]
                        
                        neckline = (lows[neckline_start] + lows[neckline_end]) / 2
                        
                        patterns.append({
                            'pattern': 'head_and_shoulders',
                            'start_idx': peaks[i-1],
                            'end_idx': peaks[i+1],
                            'neckline': neckline,
                            'target': neckline - (head - neckline),
                            'strength': 0.8
                        })
            
        except Exception as e:
            logger.error(f"Error detecting head and shoulders: {e}")
        
        return patterns
    
    def _detect_double_pattern(self, df):
        """Detect double top and double bottom patterns"""
        df = self.normalize_dataframe(df)
        patterns = []
        
        if len(df) < 40:
            return patterns
        
        try:
            # Look at recent data
            recent = df.tail(40)
            highs = recent['high'].values
            lows = recent['low'].values
            
            # Find peaks and troughs
            peaks = argrelextrema(highs, np.greater, order=5)[0]
            troughs = argrelextrema(lows, np.less, order=5)[0]
            
            # Double top
            if len(peaks) >= 2:
                for i in range(len(peaks)-1):
                    peak1 = highs[peaks[i]]
                    peak2 = highs[peaks[i+1]]
                    
                    if abs(peak1 - peak2) / peak1 < 0.03:  # Within 3%
                        # Find valley between peaks
                        valley_indices = troughs[(troughs > peaks[i]) & (troughs < peaks[i+1])]
                        if len(valley_indices) > 0:
                            valley = lows[valley_indices[0]]
                            patterns.append({
                                'pattern': 'double_top',
                                'resistance': (peak1 + peak2) / 2,
                                'support': valley,
                                'target': valley - (peak1 - valley),
                                'strength': 0.7
                            })
            
            # Double bottom
            if len(troughs) >= 2:
                for i in range(len(troughs)-1):
                    trough1 = lows[troughs[i]]
                    trough2 = lows[troughs[i+1]]
                    
                    if abs(trough1 - trough2) / trough1 < 0.03:  # Within 3%
                        # Find peak between troughs
                        peak_indices = peaks[(peaks > troughs[i]) & (peaks < troughs[i+1])]
                        if len(peak_indices) > 0:
                            peak = highs[peak_indices[0]]
                            patterns.append({
                                'pattern': 'double_bottom',
                                'support': (trough1 + trough2) / 2,
                                'resistance': peak,
                                'target': peak + (peak - trough1),
                                'strength': 0.7
                            })
            
        except Exception as e:
            logger.error(f"Error detecting double patterns: {e}")
        
        return patterns
    
    def _detect_triangle_patterns(self, df):
        """Detect triangle patterns (ascending, descending, symmetrical)"""
        df = self.normalize_dataframe(df)
        patterns = []
        
        if len(df) < 30:
            return patterns
        
        try:
            recent = df.tail(30)
            highs = recent['high'].values
            lows = recent['low'].values
            
            # Find peaks and troughs
            peaks = argrelextrema(highs, np.greater, order=3)[0]
            troughs = argrelextrema(lows, np.less, order=3)[0]
            
            if len(peaks) >= 2 and len(troughs) >= 2:
                # Get trend lines
                peak_trend = np.polyfit(peaks, highs[peaks], 1)
                trough_trend = np.polyfit(troughs, lows[troughs], 1)
                
                # Ascending triangle: flat top, rising bottom
                if abs(peak_trend[0]) < 0.001 and trough_trend[0] > 0.001:
                    patterns.append({
                        'pattern': 'ascending_triangle',
                        'resistance': np.mean(highs[peaks]),
                        'trend_slope': trough_trend[0],
                        'strength': 0.6,
                        'breakout_direction': 'bullish'
                    })
                
                # Descending triangle: falling top, flat bottom
                elif peak_trend[0] < -0.001 and abs(trough_trend[0]) < 0.001:
                    patterns.append({
                        'pattern': 'descending_triangle',
                        'support': np.mean(lows[troughs]),
                        'trend_slope': peak_trend[0],
                        'strength': 0.6,
                        'breakout_direction': 'bearish'
                    })
                
                # Symmetrical triangle: converging trends
                elif peak_trend[0] < -0.001 and trough_trend[0] > 0.001:
                    patterns.append({
                        'pattern': 'symmetrical_triangle',
                        'upper_slope': peak_trend[0],
                        'lower_slope': trough_trend[0],
                        'strength': 0.5,
                        'breakout_direction': 'neutral'
                    })
        
        except Exception as e:
            logger.error(f"Error detecting triangle patterns: {e}")
        
        return patterns
    
    def _detect_flag_patterns(self, df):
        """Detect flag and pennant patterns"""
        df = self.normalize_dataframe(df)
        patterns = []
        
        if len(df) < 20:
            return patterns
        
        try:
            closes = df['close'].values
            volumes = df['volume'].values if 'volume' in df.columns else None
            
            # Look for sharp move followed by consolidation
            for i in range(10, len(df) - 10):
                # Check for pole (sharp move)
                pole_start = i - 10
                pole_end = i
                pole_move = (closes[pole_end] - closes[pole_start]) / closes[pole_start]
                
                # Significant move (5%+)
                if abs(pole_move) > 0.05:
                    # Check for flag (consolidation)
                    flag_data = closes[pole_end:pole_end+10]
                    flag_high = np.max(flag_data)
                    flag_low = np.min(flag_data)
                    flag_range = (flag_high - flag_low) / flag_low
                    
                    # Small consolidation range (less than 3%)
                    if flag_range < 0.03:
                        pattern_type = 'bull_flag' if pole_move > 0 else 'bear_flag'
                        patterns.append({
                            'pattern': pattern_type,
                            'pole_height': abs(pole_move),
                            'flag_range': flag_range,
                            'target': closes[pole_end] + (closes[pole_end] - closes[pole_start]),
                            'strength': 0.6
                        })
        
        except Exception as e:
            logger.error(f"Error detecting flag patterns: {e}")
        
        return patterns
    
    def _detect_wedge_patterns(self, df):
        """Detect wedge patterns (rising and falling)"""
        df = self.normalize_dataframe(df)
        patterns = []
        
        if len(df) < 30:
            return patterns
        
        try:
            recent = df.tail(30)
            highs = recent['high'].values
            lows = recent['low'].values
            
            # Find peaks and troughs
            peaks = argrelextrema(highs, np.greater, order=3)[0]
            troughs = argrelextrema(lows, np.less, order=3)[0]
            
            if len(peaks) >= 3 and len(troughs) >= 3:
                # Get trend lines
                peak_trend = np.polyfit(peaks, highs[peaks], 1)
                trough_trend = np.polyfit(troughs, lows[troughs], 1)
                
                # Both trends in same direction but converging
                if peak_trend[0] > 0 and trough_trend[0] > 0 and peak_trend[0] < trough_trend[0]:
                    patterns.append({
                        'pattern': 'rising_wedge',
                        'upper_slope': peak_trend[0],
                        'lower_slope': trough_trend[0],
                        'strength': 0.7,
                        'breakout_direction': 'bearish'  # Rising wedges typically break down
                    })
                
                elif peak_trend[0] < 0 and trough_trend[0] < 0 and peak_trend[0] > trough_trend[0]:
                    patterns.append({
                        'pattern': 'falling_wedge',
                        'upper_slope': peak_trend[0],
                        'lower_slope': trough_trend[0],
                        'strength': 0.7,
                        'breakout_direction': 'bullish'  # Falling wedges typically break up
                    })
        
        except Exception as e:
            logger.error(f"Error detecting wedge patterns: {e}")
        
        return patterns
    
    def detect_divergences(self, price_data, indicator_data, lookback=14):
        """Detect divergences between price and indicators"""
        divergences = []
        
        if len(price_data) < lookback or len(indicator_data) < lookback:
            return divergences
        
        try:
            # Find price peaks and troughs
            price_peaks = argrelextrema(price_data.values, np.greater, order=5)[0]
            price_troughs = argrelextrema(price_data.values, np.less, order=5)[0]
            
            # Find indicator peaks and troughs
            ind_peaks = argrelextrema(indicator_data.values, np.greater, order=5)[0]
            ind_troughs = argrelextrema(indicator_data.values, np.less, order=5)[0]
            
            # Check for bullish divergence (price lower low, indicator higher low)
            if len(price_troughs) >= 2 and len(ind_troughs) >= 2:
                recent_price_trough = price_data.iloc[price_troughs[-1]]
                prev_price_trough = price_data.iloc[price_troughs[-2]]
                
                recent_ind_trough = indicator_data.iloc[ind_troughs[-1]]
                prev_ind_trough = indicator_data.iloc[ind_troughs[-2]]
                
                if (recent_price_trough < prev_price_trough and 
                    recent_ind_trough > prev_ind_trough):
                    divergences.append({
                        'type': 'bullish_divergence',
                        'strength': abs(recent_ind_trough - prev_ind_trough) / prev_ind_trough,
                        'price_points': [price_troughs[-2], price_troughs[-1]],
                        'indicator_points': [ind_troughs[-2], ind_troughs[-1]]
                    })
            
            # Check for bearish divergence (price higher high, indicator lower high)
            if len(price_peaks) >= 2 and len(ind_peaks) >= 2:
                recent_price_peak = price_data.iloc[price_peaks[-1]]
                prev_price_peak = price_data.iloc[price_peaks[-2]]
                
                recent_ind_peak = indicator_data.iloc[ind_peaks[-1]]
                prev_ind_peak = indicator_data.iloc[ind_peaks[-2]]
                
                if (recent_price_peak > prev_price_peak and 
                    recent_ind_peak < prev_ind_peak):
                    divergences.append({
                        'type': 'bearish_divergence',
                        'strength': abs(prev_ind_peak - recent_ind_peak) / prev_ind_peak,
                        'price_points': [price_peaks[-2], price_peaks[-1]],
                        'indicator_points': [ind_peaks[-2], ind_peaks[-1]]
                    })
        
        except Exception as e:
            logger.error(f"Error detecting divergences: {e}")
        
        return divergences
    
    def calculate_pattern_score(self, patterns, current_price):
        """Calculate a score based on detected patterns"""
        if not patterns:
            return 0
        
        score = 0
        
        for pattern in patterns:
            pattern_type = pattern.get('pattern', pattern.get('type', ''))
            strength = pattern.get('strength', 0.5)
            
            # Bullish patterns
            if pattern_type in ['double_bottom', 'ascending_triangle', 'bull_flag', 
                               'falling_wedge', 'bullish_divergence']:
                score += 2 * strength
            
            # Bearish patterns
            elif pattern_type in ['double_top', 'descending_triangle', 'bear_flag', 
                                 'rising_wedge', 'bearish_divergence', 'head_and_shoulders']:
                score -= 2 * strength
            
            # Neutral patterns
            elif pattern_type in ['symmetrical_triangle']:
                # Check breakout direction based on recent price action
                if 'breakout_direction' in pattern:
                    if pattern['breakout_direction'] == 'bullish':
                        score += 1 * strength
                    elif pattern['breakout_direction'] == 'bearish':
                        score -= 1 * strength
        
        return score
    
    def detect_candlestick_patterns(self, df, lookback=5):
        """Detect candlestick patterns"""
        df = self.normalize_dataframe(df)
        patterns = []
        
        if len(df) < lookback:
            return patterns
        
        try:
            recent = df.tail(lookback)
            
            # Calculate candlestick properties
            for i in range(len(recent)):
                open_price = recent['open'].iloc[i]
                high = recent['high'].iloc[i]
                low = recent['low'].iloc[i]
                close = recent['close'].iloc[i]
                
                body = close - open_price
                body_size = abs(body)
                upper_shadow = high - max(open_price, close)
                lower_shadow = min(open_price, close) - low
                total_range = high - low
                
                # Doji
                if body_size < total_range * 0.1:
                    patterns.append({
                        'pattern': 'doji',
                        'position': i,
                        'significance': 'neutral/reversal'
                    })
                
                # Hammer (bullish)
                if (lower_shadow > body_size * 2 and 
                    upper_shadow < body_size * 0.5 and
                    i > 0 and recent['close'].iloc[i-1] < recent['close'].iloc[i]):
                    patterns.append({
                        'pattern': 'hammer',
                        'position': i,
                        'significance': 'bullish_reversal'
                    })
                
                # Shooting star (bearish)
                if (upper_shadow > body_size * 2 and 
                    lower_shadow < body_size * 0.5 and
                    i > 0 and recent['close'].iloc[i-1] > recent['close'].iloc[i]):
                    patterns.append({
                        'pattern': 'shooting_star',
                        'position': i,
                        'significance': 'bearish_reversal'
                    })
                
                # Engulfing patterns
                if i > 0:
                    prev_body = recent['close'].iloc[i-1] - recent['open'].iloc[i-1]
                    
                    # Bullish engulfing
                    if (prev_body < 0 and body > 0 and 
                        open_price < recent['close'].iloc[i-1] and 
                        close > recent['open'].iloc[i-1]):
                        patterns.append({
                            'pattern': 'bullish_engulfing',
                            'position': i,
                            'significance': 'bullish_reversal'
                        })
                    
                    # Bearish engulfing
                    if (prev_body > 0 and body < 0 and 
                        open_price > recent['close'].iloc[i-1] and 
                        close < recent['open'].iloc[i-1]):
                        patterns.append({
                            'pattern': 'bearish_engulfing',
                            'position': i,
                            'significance': 'bearish_reversal'
                        })
        
        except Exception as e:
            logger.error(f"Error detecting candlestick patterns: {e}")
        
        return patterns
    
    def identify_trend(self, df, period=20):
        """Identify the current trend (uptrend, downtrend, sideways)"""
        df = self.normalize_dataframe(df)
        
        if len(df) < period:
            return {'trend': 'unknown', 'strength': 0}
        
        try:
            closes = df['close'].tail(period)
            highs = df['high'].tail(period)
            lows = df['low'].tail(period)
            
            # Calculate trend using linear regression
            x = np.arange(len(closes))
            trend_slope = np.polyfit(x, closes.values, 1)[0]
            
            # Calculate trend strength using R-squared
            y_pred = np.polyval(np.polyfit(x, closes.values, 1), x)
            ss_res = np.sum((closes.values - y_pred) ** 2)
            ss_tot = np.sum((closes.values - np.mean(closes.values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Determine trend direction
            avg_price = closes.mean()
            slope_percent = (trend_slope / avg_price) * 100
            
            if slope_percent > 0.5:
                trend = 'uptrend'
            elif slope_percent < -0.5:
                trend = 'downtrend'
            else:
                trend = 'sideways'
            
            # Calculate trend channels
            upper_channel = highs.max()
            lower_channel = lows.min()
            channel_width = (upper_channel - lower_channel) / avg_price
            
            return {
                'trend': trend,
                'strength': r_squared,
                'slope': slope_percent,
                'upper_channel': upper_channel,
                'lower_channel': lower_channel,
                'channel_width': channel_width
            }
            
        except Exception as e:
            logger.error(f"Error identifying trend: {e}")
            return {'trend': 'unknown', 'strength': 0}
    
    def detect_breakouts(self, df, lookback=20, volume_threshold=1.5):
        """Detect price breakouts from consolidation"""
        df = self.normalize_dataframe(df)
        breakouts = []
        
        if len(df) < lookback + 5:
            return breakouts
        
        try:
            for i in range(lookback, len(df)):
                # Get the consolidation period
                consolidation = df.iloc[i-lookback:i]
                current = df.iloc[i]
                
                # Calculate consolidation range
                high_range = consolidation['high'].max()
                low_range = consolidation['low'].min()
                avg_volume = consolidation['volume'].mean()
                
                # Check for breakout
                if current['close'] > high_range * 1.02:  # Upward breakout
                    if current['volume'] > avg_volume * volume_threshold:
                        breakouts.append({
                            'type': 'bullish_breakout',
                            'index': i,
                            'breakout_level': high_range,
                            'volume_surge': current['volume'] / avg_volume,
                            'strength': (current['close'] - high_range) / high_range
                        })
                
                elif current['close'] < low_range * 0.98:  # Downward breakout
                    if current['volume'] > avg_volume * volume_threshold:
                        breakouts.append({
                            'type': 'bearish_breakout',
                            'index': i,
                            'breakout_level': low_range,
                            'volume_surge': current['volume'] / avg_volume,
                            'strength': (low_range - current['close']) / low_range
                        })
        
        except Exception as e:
            logger.error(f"Error detecting breakouts: {e}")
        
        return breakouts