# src/data/database.py
import os
import sqlite3
import pandas as pd
import json
import logging
import threading
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)

class StockDatabase:
    """Thread-safe database for storing stock data"""
    
    def __init__(self, db_path="stock_data.db"):
        """Initialize database connection"""
        self.db_path = db_path
        self.lock = threading.Lock()  # Add lock for thread safety
        
        # Create database directory if it doesn't exist
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
        
        self.initialize_db()
    
    def get_connection(self):
        """Get a thread-safe database connection"""
        return sqlite3.connect(self.db_path, check_same_thread=False)
    
    def initialize_db(self):
        """Initialize database and create tables if they don't exist"""
        try:
            with self.lock:  # Use lock when accessing database
                conn = self.get_connection()
                cursor = conn.cursor()
                
                # Create tables
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS quotes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    change REAL,
                    change_percent TEXT,
                    volume INTEGER,
                    timestamp TEXT NOT NULL,
                    source TEXT
                )
                ''')
                
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS historical_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    timestamp TEXT NOT NULL,
                    UNIQUE(symbol, date)
                )
                ''')
                
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS api_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    endpoint TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    status TEXT,
                    response_time REAL
                )
                ''')
                
                conn.commit()
                conn.close()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
    
    def store_quote(self, quote_data, source="alpha_vantage"):
        """Store quote data in database"""
        try:
            with self.lock:  # Use lock when accessing database
                conn = self.get_connection()
                cursor = conn.cursor()
                
                timestamp = datetime.now().isoformat()
                cursor.execute('''
                INSERT INTO quotes (symbol, price, change, change_percent, volume, timestamp, source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    quote_data['symbol'],
                    quote_data['price'],
                    quote_data['change'],
                    quote_data['change_percent'],
                    quote_data['volume'],
                    timestamp,
                    source
                ))
                
                conn.commit()
                conn.close()
                logger.info(f"Stored quote for {quote_data['symbol']} in database")
                return True
                
        except Exception as e:
            logger.error(f"Error storing quote: {str(e)}")
            return False
    
    def store_historical_data(self, symbol, data_df):
        """Store historical data in database"""
        if data_df.empty:
            logger.error("Empty data, nothing to store")
            return False
            
        try:
            with self.lock:  # Use lock when accessing database
                conn = self.get_connection()
                cursor = conn.cursor()
                
                timestamp = datetime.now().isoformat()
                
                # Prepare data for bulk insert
                records = []
                for date, row in data_df.iterrows():
                    records.append((
                        symbol,
                        date.strftime('%Y-%m-%d'),
                        float(row['open']),
                        float(row['high']),
                        float(row['low']),
                        float(row['close']),
                        int(row['volume']),
                        timestamp
                    ))
                
                # Use executemany for efficient bulk insert
                cursor.executemany('''
                INSERT OR REPLACE INTO historical_data 
                (symbol, date, open, high, low, close, volume, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', records)
                
                conn.commit()
                conn.close()
                logger.info(f"Stored {len(records)} historical records for {symbol} in database")
                return True
                
        except Exception as e:
            logger.error(f"Error storing historical data: {str(e)}")
            return False
    
    def log_api_call(self, endpoint, status, response_time):
        """Log API call for tracking usage"""
        try:
            with self.lock:  # Use lock when accessing database
                conn = self.get_connection()
                cursor = conn.cursor()
                
                timestamp = datetime.now().isoformat()
                cursor.execute('''
                INSERT INTO api_usage (endpoint, timestamp, status, response_time)
                VALUES (?, ?, ?, ?)
                ''', (endpoint, timestamp, status, response_time))
                
                conn.commit()
                conn.close()
        except Exception as e:
            logger.error(f"Error logging API call: {str(e)}")
    
    def get_latest_quotes(self, symbols=None, limit=20):
        """Get the latest quotes for specified symbols"""
        try:
            with self.lock:  # Use lock when accessing database
                conn = self.get_connection()
                cursor = conn.cursor()
                
                if symbols:
                    # Get latest quotes for specific symbols
                    placeholders = ', '.join('?' for _ in symbols)
                    query = f'''
                    SELECT q1.* FROM quotes q1
                    JOIN (
                        SELECT symbol, MAX(timestamp) as max_time
                        FROM quotes
                        WHERE symbol IN ({placeholders})
                        GROUP BY symbol
                    ) q2
                    ON q1.symbol = q2.symbol AND q1.timestamp = q2.max_time
                    ORDER BY q1.symbol
                    '''
                    cursor.execute(query, symbols)
                else:
                    # Get latest quotes for all symbols
                    query = '''
                    SELECT q1.* FROM quotes q1
                    JOIN (
                        SELECT symbol, MAX(timestamp) as max_time
                        FROM quotes
                        GROUP BY symbol
                        LIMIT ?
                    ) q2
                    ON q1.symbol = q2.symbol AND q1.timestamp = q2.max_time
                    ORDER BY q1.symbol
                    '''
                    cursor.execute(query, (limit,))
                
                columns = [desc[0] for desc in cursor.description]
                results = []
                
                for row in cursor.fetchall():
                    quote = dict(zip(columns, row))
                    results.append(quote)
                
                conn.close()
                return results
            
        except Exception as e:
            logger.error(f"Error retrieving latest quotes: {str(e)}")
            return []
    
    def get_historical_data(self, symbol, days=30):
        """Get historical data for a symbol"""
        try:
            with self.lock:  # Use lock when accessing database
                conn = self.get_connection()
                
                query = '''
                SELECT date, open, high, low, close, volume
                FROM historical_data
                WHERE symbol = ?
                ORDER BY date DESC
                LIMIT ?
                '''
                
                df = pd.read_sql_query(query, conn, params=(symbol, days))
                conn.close()
                
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                
                return df
            
        except Exception as e:
            logger.error(f"Error retrieving historical data: {str(e)}")
            return pd.DataFrame()
    
    def get_api_usage_stats(self, hours=24):
        """Get API usage statistics for the past hours"""
        try:
            with self.lock:  # Use lock when accessing database
                conn = self.get_connection()
                cursor = conn.cursor()
                
                time_threshold = (datetime.now() - pd.Timedelta(hours=hours)).isoformat()
                
                query = '''
                SELECT endpoint, COUNT(*) as count
                FROM api_usage
                WHERE timestamp > ?
                GROUP BY endpoint
                '''
                
                cursor.execute(query, (time_threshold,))
                results = dict(cursor.fetchall())
                
                conn.close()
                return results
                
        except Exception as e:
            logger.error(f"Error retrieving API usage stats: {str(e)}")
            return {}