# Stock Market Analyzer & Predictor

A real-time stock market analysis tool with technical indicators and trading predictions using Yahoo Finance data.

## Features

- **Real-time Stock Quotes** - Live prices during market hours
- **Technical Analysis** - RSI, MACD, Bollinger Bands, Moving Averages
- **Trading Predictions** - AI-powered opportunity detection
- **Entry/Exit Points** - Suggested support/resistance levels
- **No API Key Required** - Uses Yahoo Finance (yfinance)
- **No Rate Limits** - Query as often as needed

## Quick Start

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Populate Historical Data
```bash
python fix_data.py
```
This fetches 3 months of historical data for all stocks (takes ~30 seconds).

### 3. Run the Web App
```bash
python run_webapp.py
```

The app will open automatically at http://localhost:5000

## Usage Guide

### Stock Market Overview
- View real-time prices for 15 major stocks
- Prices update automatically every 30 seconds
- Click any stock card for detailed analysis

### Trading Predictions
- Click "Run Analysis" to scan for opportunities
- Analyzes technical indicators across all stocks
- Shows stocks with strongest trading signals
- Score 4+ = Strong opportunity
- Score 2-3 = Moderate opportunity

### Technical Analysis
- Select any stock from the dropdown
- View price chart with moving averages and Bollinger Bands
- See RSI and MACD indicators
- Get entry/exit point suggestions

## How It Works

### Data Source
- **Yahoo Finance (yfinance)** - Real-time quotes and historical data
- **No delays** - Live data during market hours
- **Reliable** - Works globally without restrictions

### Technical Indicators

1. **RSI (Relative Strength Index)**
   - Oversold < 30 = Potential buy signal
   - Overbought > 70 = Potential sell signal

2. **Moving Averages**
   - Short MA crosses above Long MA = Bullish signal
   - Short MA crosses below Long MA = Bearish signal

3. **Bollinger Bands**
   - Price near lower band = Potential bounce
   - Price near upper band = Potential reversal

4. **MACD**
   - MACD crosses above signal = Bullish
   - MACD crosses below signal = Bearish

5. **Volume Analysis**
   - Volume surge (>1.5x average) = Strong interest

### Scoring System
The predictor combines multiple signals:
- Moving average crossover: +3 points
- RSI oversold/overbought: +2 points
- Bollinger Band touch: +2 points
- MACD crossover: +2 points
- Volume surge: +2 points
- Strong momentum: +1 point

## Project Structure
```
├── run_webapp.py          # Main entry point
├── fix_data.py           # Data population script
├── requirements.txt      # Python dependencies
├── src/
│   ├── web/
│   │   ├── app.py       # Flask application
│   │   └── templates/   # HTML templates
│   ├── data/
│   │   ├── fetcher.py   # Yahoo Finance data fetcher
│   │   └── database.py  # SQLite database handler
│   └── models/
│       └── predictor.py # Technical analysis & predictions
```

## Troubleshooting

### No Trading Opportunities Found
- Run `python fix_data.py` to ensure all stocks have data
- Check that at least 5 stocks have 30+ days of history
- Market conditions might not favor any opportunities

### Empty Stock Display
- Yahoo Finance might be temporarily down
- Check your internet connection
- Data will be cached after first successful fetch

### Slow Initial Load
- First fetch downloads 3 months of data for all stocks
- Subsequent loads use cached data (much faster)

## Advanced Usage

### Change Stock List
Edit `get_top_stocks()` in `src/data/fetcher.py` to track different stocks.

### Adjust Prediction Sensitivity
Edit thresholds in `src/models/predictor.py`:
- `rsi_oversold` / `rsi_overbought`
- `min_volatility`
- `volume_surge_threshold`

### Change Update Interval
Edit `update_interval` in `src/web/app.py` (default: 30 seconds).

## License
MIT License - Feel free to use and modify!