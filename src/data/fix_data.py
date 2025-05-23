# fix_data.py
"""
Quick script to fetch all historical data using Yahoo Finance.
Much faster and more reliable than Alpha Vantage!
"""
import os
import sys
import logging

# Add src to path
sys.path.append('src')

from data.database import StockDatabase
from data.fetcher import StockDataFetcher

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Create database and fetcher
    db_path = os.path.join('src', 'data', 'stock_data.db')
    db = StockDatabase(db_path=db_path)
    fetcher = StockDataFetcher(db=db)
    
    logger.info("=== Stock Data Fetcher (Yahoo Finance) ===")
    logger.info("This will fetch historical data for all stocks.")
    logger.info("Yahoo Finance has no rate limits, so this should be fast!\n")
    
    # Get all stocks
    all_stocks = fetcher.get_top_stocks()
    
    logger.info(f"Fetching data for {len(all_stocks)} stocks: {all_stocks}\n")
    
    # Use batch download for maximum efficiency
    stocks_data = fetcher.get_stocks_for_analysis(all_stocks)
    
    logger.info(f"\n✅ Successfully fetched data for {len(stocks_data)} stocks!")
    
    # Show summary
    logger.info("\n=== Database Summary ===")
    total_records = 0
    for symbol in all_stocks:
        data = db.get_historical_data(symbol, days=100)
        if not data.empty:
            logger.info(f"{symbol}: {len(data)} days of data ✓")
            total_records += len(data)
        else:
            logger.warning(f"{symbol}: NO DATA ✗")
    
    logger.info(f"\nTotal records in database: {total_records}")
    logger.info("\n✨ Done! You can now run the web app and all features should work perfectly.")

if __name__ == "__main__":
    main()