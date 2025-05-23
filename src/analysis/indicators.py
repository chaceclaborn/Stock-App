# G:\Coding\Stocks\src\analysis\indicators.py
import pandas as pd
import numpy as np


class TechnicalIndicators:
    """Calculate technical indicators for stock analysis."""
    
    @staticmethod
    def add_moving_averages(df, short_window=5, long_window=20):
        """Add short and long moving averages to dataframe."""
        # Make a copy so we don't modify the original
        df = df.copy()
        
        # Add short and long moving averages
        df['SMA_short'] = df['Close'].rolling(window=short_window).mean()
        df['SMA_long'] = df['Close'].rolling(window=long_window).mean()
        
        # Add moving average crossover signal
        df['Signal'] = 0
        df.loc[df['SMA_short'] > df['SMA_long'], 'Signal'] = 1
        df.loc[df['SMA_short'] < df['SMA_long'], 'Signal'] = -1
        
        return df
    
    @staticmethod
    def add_rsi(df, window=14):
        """Add Relative Strength Index indicator."""
        df = df.copy()
        
        # Calculate price changes
        delta = df['Close'].diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        # Handle division by zero
        avg_loss = avg_loss.replace(0, 0.001)  # Avoid division by zero
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df
    
    @staticmethod
    def add_volatility(df, window=20):
        """Add volatility indicator."""
        df = df.copy()
        df['Volatility'] = df['Close'].pct_change().rolling(window=window).std() * np.sqrt(252)
        return df
    
    @staticmethod
    def add_all_indicators(df):
        """Add all technical indicators to dataframe."""
        try:
            # Ensure dataframe has the required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Column {col} missing from dataframe")
            
            # Add each indicator
            df = TechnicalIndicators.add_moving_averages(df)
            df = TechnicalIndicators.add_rsi(df)
            df = TechnicalIndicators.add_volatility(df)
            
            # Drop NaN values that result from calculations
            df = df.fillna(method='bfill')
            
            return df
        except Exception as e:
            print(f"Error adding indicators: {str(e)}")
            # Return the original DataFrame if there's an error
            return df