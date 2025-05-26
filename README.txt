# Chace's Stock Analyzer - Setup Guide

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   pip install flask yfinance pandas numpy requests textblob
   ```

2. **Run Diagnostic Test**
   ```bash
   python test_setup.py
   ```

3. **Start the Application**
   ```bash
   python run_webapp.py
   ```

4. **Open in Browser**
   Navigate to: http://localhost:5000

## 📁 Required File Structure

```
your-project/
├── run_webapp.py
├── test_setup.py
├── data/                    # Will be created automatically
├── src/
│   ├── __init__.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── indicators.py
│   │   └── pattern_recognition.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── database.py
│   │   └── fetcher.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── predictor.py
│   └── web/
│       ├── __init__.py
│       ├── app.py
│       ├── routes/
│       │   ├── __init__.py
│       │   ├── stocks.py
│       │   ├── analysis.py
│       │   ├── market.py
│       │   ├── portfolio.py
│       │   └── predictions.py
│       ├── services/
│       │   ├── __init__.py
│       │   ├── stock_service.py
│       │   ├── analysis_service.py
│       │   ├── market_service.py
│       │   ├── prediction_service.py
│       │   └── sentiment_service.py
│       ├── utils/
│       │   ├── __init__.py
│       │   └── api_response.py
│       └── templates/
│           ├── index.html
│           └── partials/
│               ├── header.html
│               ├── styles.html
│               ├── scripts.html
│               ├── search.html
│               ├── tabs.html
│               ├── overview.html
│               ├── stocks_grid.html
│               └── stock_modal.html
```

## 🔧 Troubleshooting

### Stocks Not Loading?

1. **Check API Status**
   - Visit: http://localhost:5000/api/debug
   - Look for any errors or missing services

2. **Test Single Stock**
   - Visit: http://localhost:5000/api/test-stock/AAPL
   - This tests if Yahoo Finance is working

3. **Check Console**
   - Open browser DevTools (F12)
   - Look for errors in Console tab
   - Check Network tab for failed requests

4. **Common Issues**
   - **Import Error**: Make sure all __init__.py files exist
   - **No Data**: Yahoo Finance might be temporarily down
   - **Service Not Available**: Services didn't initialize properly

### Manual Fixes

1. **Clear Browser Cache**
   ```
   Ctrl+Shift+R (Windows/Linux)
   Cmd+Shift+R (Mac)
   ```

2. **Restart Application**
   ```bash
   # Stop with Ctrl+C
   # Start again
   python run_webapp.py --debug
   ```

3. **Check Yahoo Finance**
   ```python
   import yfinance as yf
   ticker = yf.Ticker("AAPL")
   print(ticker.info)
   ```

## 📊 Features

- **Real-time Stock Data**: Live prices from Yahoo Finance
- **Technical Analysis**: RSI, MACD, Bollinger Bands, and more
- **AI Predictions**: Machine learning-based trading signals
- **Portfolio Tracking**: Track your investments
- **Market Overview**: Major indices and sentiment
- **Pattern Recognition**: Detect chart patterns
- **Risk Management**: Stop loss and take profit suggestions

## 🛠️ Development

### Debug Mode
```bash
python run_webapp.py --debug
```

### Change Port
```bash
python run_webapp.py --port 8080
```

### No Browser Launch
```bash
python run_webapp.py --no-browser
```

## 📝 API Endpoints

- `/api/stocks` - Get all stocks
- `/api/stock/{symbol}` - Get stock details
- `/api/stock/{symbol}/analysis` - Technical analysis
- `/api/predictions` - AI predictions
- `/api/market-overview` - Market indices
- `/api/debug` - System status

## 💡 Tips

1. **First Run**: The app needs to initialize the database and fetch initial data. This might take a minute.

2. **Caching**: The app caches data to reduce API calls. Use the refresh button or add `?force=true` to API calls to get fresh data.

3. **Rate Limits**: Yahoo Finance has rate limits. If you see errors, wait a few minutes.

4. **Portfolio**: Portfolio data is stored in browser localStorage. Export it regularly.

## 🐛 Reporting Issues

If stocks still aren't loading after following this guide:

1. Run the diagnostic test and save output
2. Check browser console for errors
3. Check `/api/debug` response
4. Look at server console output

Include all this information when reporting issues.