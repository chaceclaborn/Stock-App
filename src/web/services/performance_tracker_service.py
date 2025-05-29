# src/web/services/performance_tracker_service.py
"""
Automated performance tracking service for AI predictions
"""
import logging
import json
import sqlite3
from datetime import datetime, timedelta
import threading
import time
import pandas as pd
import numpy as np
import os

logger = logging.getLogger(__name__)

class PerformanceTrackerService:
    """Automatically tracks and evaluates AI prediction performance"""
    
    def __init__(self):
        self._db = None
        self._fetcher = None
        self._predictor = None
        self.tracking_thread = None
        self.is_tracking = False
        self.tracked_predictions = {}
        self.check_interval = 300  # 5 minutes
        
    def init_app(self, db, fetcher, predictor):
        """Initialize with dependencies"""
        self._db = db
        self._fetcher = fetcher
        self._predictor = predictor
        self._init_tracking_database()
        
    def _init_tracking_database(self):
        """Initialize tracking database tables"""
        try:
            # Create a separate tracking database
            db_dir = os.path.dirname(self._db.db_path)
            tracking_db_path = os.path.join(db_dir, 'prediction_tracking.db')
            
            conn = sqlite3.connect(tracking_db_path)
            cursor = conn.cursor()
            
            # Create tracking table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS prediction_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                prediction_date TEXT NOT NULL,
                prediction_type TEXT NOT NULL,
                predicted_score REAL,
                entry_price REAL,
                target_price REAL,
                stop_loss REAL,
                timeframe TEXT,
                patterns TEXT,
                signals TEXT,
                status TEXT DEFAULT 'active',
                exit_date TEXT,
                exit_price REAL,
                actual_return REAL,
                success BOOLEAN,
                UNIQUE(symbol, prediction_date, prediction_type)
            )
            ''')
            
            # Create performance metrics table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                predictor_type TEXT NOT NULL,
                total_predictions INTEGER,
                successful_predictions INTEGER,
                win_rate REAL,
                avg_return REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                best_prediction TEXT,
                worst_prediction TEXT,
                UNIQUE(date, predictor_type)
            )
            ''')
            
            # Create learning history table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                predictor_type TEXT NOT NULL,
                pattern_weights TEXT,
                signal_weights TEXT,
                model_version INTEGER DEFAULT 1
            )
            ''')
            
            conn.commit()
            conn.close()
            self.tracking_db_path = tracking_db_path
            logger.info("Performance tracking database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing tracking database: {e}")
    
    def start_automated_tracking(self):
        """Start automated tracking of top predictions"""
        if not self.is_tracking:
            self.is_tracking = True
            self.tracking_thread = threading.Thread(target=self._tracking_loop)
            self.tracking_thread.daemon = True
            self.tracking_thread.start()
            logger.info("Started automated performance tracking")
    
    def stop_automated_tracking(self):
        """Stop automated tracking"""
        self.is_tracking = False
        if self.tracking_thread:
            self.tracking_thread.join()
        logger.info("Stopped automated performance tracking")
    
    def _tracking_loop(self):
        """Main tracking loop"""
        while self.is_tracking:
            try:
                # Track new predictions
                self._track_top_predictions()
                
                # Update existing tracked predictions
                self._update_tracked_predictions()
                
                # Calculate and store performance metrics
                self._calculate_performance_metrics()
                
                # Train predictors based on feedback
                self._train_predictors()
                
                # Sleep until next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in tracking loop: {e}")
                time.sleep(60)  # Wait a minute on error
    
    def _track_top_predictions(self):
        """Track the top 3 predictions from both short and long term predictors"""
        try:
            # Get current predictions for both timeframes
            short_term_predictions = self._get_predictions('1d', top_n=3)
            long_term_predictions = self._get_predictions('1m', top_n=3)
            
            # Track short-term predictions
            for pred in short_term_predictions:
                self._add_prediction_to_tracking(pred, 'short_term')
            
            # Track long-term predictions
            for pred in long_term_predictions:
                self._add_prediction_to_tracking(pred, 'long_term')
                
            logger.info(f"Tracked {len(short_term_predictions)} short-term and "
                       f"{len(long_term_predictions)} long-term predictions")
                
        except Exception as e:
            logger.error(f"Error tracking top predictions: {e}")
    
    def _get_predictions(self, period, top_n=3):
        """Get top predictions for a specific period"""
        try:
            # Get stock data
            stocks_data = self._fetcher.get_stocks_for_analysis()
            
            if not stocks_data:
                return []
            
            # Find opportunities with period-specific parameters
            opportunities_df = self._predictor.find_opportunities(
                stocks_data, 
                top_n=top_n,
                period=period
            )
            
            if opportunities_df.empty:
                return []
            
            # Filter by minimum score thresholds
            min_score = 5 if period == '1d' else 6  # Higher threshold for long-term
            opportunities_df = opportunities_df[opportunities_df['score'] >= min_score]
            
            return opportunities_df.head(top_n).to_dict('records')
            
        except Exception as e:
            logger.error(f"Error getting predictions for period {period}: {e}")
            return []
    
    def _add_prediction_to_tracking(self, prediction, prediction_type):
        """Add a prediction to tracking"""
        try:
            conn = sqlite3.connect(self.tracking_db_path)
            cursor = conn.cursor()
            
            # Check if already tracking this prediction today
            cursor.execute('''
            SELECT id FROM prediction_tracking 
            WHERE symbol = ? AND DATE(prediction_date) = DATE('now') AND prediction_type = ?
            ''', (prediction['ticker'], prediction_type))
            
            if cursor.fetchone():
                conn.close()
                return  # Already tracking
            
            # Get risk metrics and strategy
            risk_metrics = prediction.get('risk_metrics', {})
            strategy = prediction.get('strategy', {})
            
            # Calculate targets based on prediction type
            if prediction_type == 'short_term':
                # Short-term: 2-5% targets
                target_price = prediction['price'] * 1.03
                stop_loss = prediction['price'] * 0.98
            else:
                # Long-term: 5-15% targets
                target_price = prediction['price'] * 1.10
                stop_loss = prediction['price'] * 0.95
            
            # Use risk metrics if available
            if risk_metrics:
                target_price = risk_metrics.get('take_profit_2', target_price)
                stop_loss = risk_metrics.get('stop_loss', stop_loss)
            
            # Insert new tracking record
            cursor.execute('''
            INSERT INTO prediction_tracking (
                symbol, prediction_date, prediction_type, predicted_score,
                entry_price, target_price, stop_loss, timeframe,
                patterns, signals
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                prediction['ticker'],
                datetime.now().isoformat(),
                prediction_type,
                prediction.get('score', 0),
                prediction.get('price', 0),
                target_price,
                stop_loss,
                prediction.get('period', '1d'),
                json.dumps(prediction.get('patterns', [])),
                json.dumps(prediction.get('signals', {}))
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Started tracking {prediction_type} prediction for {prediction['ticker']} "
                       f"(score: {prediction.get('score', 0)}, target: ${target_price:.2f})")
            
        except Exception as e:
            logger.error(f"Error adding prediction to tracking: {e}")
    
    def _update_tracked_predictions(self):
        """Update all active tracked predictions"""
        try:
            conn = sqlite3.connect(self.tracking_db_path)
            cursor = conn.cursor()
            
            # Get active predictions
            cursor.execute('''
            SELECT id, symbol, entry_price, target_price, stop_loss, 
                   prediction_date, prediction_type, predicted_score,
                   patterns, signals
            FROM prediction_tracking 
            WHERE status = 'active'
            ''')
            
            active_predictions = cursor.fetchall()
            
            for pred in active_predictions:
                pred_id, symbol, entry_price, target_price, stop_loss = pred[:5]
                prediction_date, prediction_type, predicted_score = pred[5:8]
                patterns, signals = pred[8:10]
                
                # Get current price
                quote = self._fetcher.get_quote(symbol)
                if not quote:
                    continue
                
                current_price = quote['price']
                
                # Check if we should exit
                should_exit = False
                exit_reason = None
                
                # Check stop loss
                if current_price <= stop_loss:
                    should_exit = True
                    exit_reason = 'stop_loss'
                
                # Check target reached
                elif current_price >= target_price:
                    should_exit = True
                    exit_reason = 'target_reached'
                
                # Check time-based exit
                else:
                    pred_date = datetime.fromisoformat(prediction_date)
                    days_held = (datetime.now() - pred_date).days
                    
                    if prediction_type == 'short_term':
                        # Exit after 5 days for short term
                        if days_held >= 5:
                            should_exit = True
                            exit_reason = 'time_exit'
                    else:
                        # Exit after 30 days for long term
                        if days_held >= 30:
                            should_exit = True
                            exit_reason = 'time_exit'
                
                # Also check for trailing stop (protect profits)
                if not should_exit and current_price > entry_price * 1.05:
                    # If we're up 5%, use trailing stop at 2% below high
                    trailing_stop = current_price * 0.98
                    if current_price < trailing_stop:
                        should_exit = True
                        exit_reason = 'trailing_stop'
                
                if should_exit:
                    # Calculate return
                    actual_return = (current_price - entry_price) / entry_price
                    success = actual_return > 0
                    
                    # Update record
                    cursor.execute('''
                    UPDATE prediction_tracking 
                    SET status = ?, exit_date = ?, exit_price = ?, 
                        actual_return = ?, success = ?
                    WHERE id = ?
                    ''', (exit_reason, datetime.now().isoformat(), 
                          current_price, actual_return, success, pred_id))
                    
                    # Submit feedback to learning engine
                    self._predictor.learning_engine.record_feedback(
                        symbol=symbol,
                        predicted_score=predicted_score,
                        actual_return=actual_return,
                        patterns=json.loads(patterns) if patterns else [],
                        signals=json.loads(signals) if signals else {}
                    )
                    
                    logger.info(f"Closed {prediction_type} position for {symbol}: "
                              f"return={actual_return:.2%}, reason={exit_reason}")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating tracked predictions: {e}")
    
    def _calculate_performance_metrics(self):
        """Calculate and store performance metrics"""
        try:
            conn = sqlite3.connect(self.tracking_db_path)
            cursor = conn.cursor()
            
            # Calculate metrics for each predictor type
            for predictor_type in ['short_term', 'long_term']:
                # Get completed predictions from last 30 days
                cursor.execute('''
                SELECT actual_return, success, symbol, predicted_score
                FROM prediction_tracking
                WHERE prediction_type = ? 
                AND status != 'active'
                AND exit_date > datetime('now', '-30 days')
                ''', (predictor_type,))
                
                results = cursor.fetchall()
                
                if not results:
                    continue
                
                # Calculate metrics
                returns = [r[0] for r in results]
                successes = sum(1 for r in results if r[1])
                total = len(results)
                
                win_rate = successes / total if total > 0 else 0
                avg_return = np.mean(returns) if returns else 0
                
                # Calculate Sharpe ratio (simplified)
                if len(returns) > 1:
                    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
                else:
                    sharpe = 0
                
                # Calculate max drawdown
                cumulative = np.cumprod(1 + np.array(returns))
                running_max = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
                
                # Find best and worst predictions
                best_idx = np.argmax(returns) if returns else 0
                worst_idx = np.argmin(returns) if returns else 0
                
                best_pred = f"{results[best_idx][2]}: {returns[best_idx]:.2%}" if results else "N/A"
                worst_pred = f"{results[worst_idx][2]}: {returns[worst_idx]:.2%}" if results else "N/A"
                
                # Store metrics (update if exists for today)
                cursor.execute('''
                INSERT OR REPLACE INTO performance_metrics (
                    date, predictor_type, total_predictions,
                    successful_predictions, win_rate, avg_return,
                    sharpe_ratio, max_drawdown, best_prediction,
                    worst_prediction
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().date().isoformat(),
                    predictor_type,
                    total,
                    successes,
                    win_rate,
                    avg_return,
                    sharpe,
                    max_drawdown,
                    best_pred,
                    worst_pred
                ))
                
                logger.info(f"{predictor_type} performance: win_rate={win_rate:.1%}, "
                           f"avg_return={avg_return:.2%}, sharpe={sharpe:.2f}")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
    
    def _train_predictors(self):
        """Train predictors based on accumulated feedback"""
        try:
            # Only train once per day
            conn = sqlite3.connect(self.tracking_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT COUNT(*) FROM learning_history
            WHERE DATE(date) = DATE('now')
            ''')
            
            if cursor.fetchone()[0] > 0:
                conn.close()
                return  # Already trained today
            
            # Get feedback data for training
            cursor.execute('''
            SELECT symbol, predicted_score, actual_return, patterns, signals, prediction_type
            FROM prediction_tracking
            WHERE status != 'active'
            AND exit_date > datetime('now', '-90 days')
            ''')
            
            feedback_data = cursor.fetchall()
            
            if len(feedback_data) < 20:
                conn.close()
                return  # Not enough data to train
            
            # Separate by predictor type
            short_term_data = [f for f in feedback_data if f[5] == 'short_term']
            long_term_data = [f for f in feedback_data if f[5] == 'long_term']
            
            # Train short-term predictor
            if len(short_term_data) >= 10:
                self._train_specific_predictor(short_term_data, 'short_term')
            
            # Train long-term predictor
            if len(long_term_data) >= 10:
                self._train_specific_predictor(long_term_data, 'long_term')
            
            # Record training
            cursor.execute('''
            INSERT INTO learning_history (date, predictor_type, model_version)
            VALUES (?, ?, ?)
            ''', (datetime.now().isoformat(), 'both', 1))
            
            conn.commit()
            conn.close()
            
            logger.info("Completed predictor training cycle")
            
        except Exception as e:
            logger.error(f"Error training predictors: {e}")
    
    def _train_specific_predictor(self, feedback_data, predictor_type):
        """Train a specific predictor with feedback data"""
        try:
            # Analyze pattern success rates
            pattern_performance = {}
            signal_performance = {}
            
            for _, score, actual_return, patterns_json, signals_json, _ in feedback_data:
                patterns = json.loads(patterns_json) if patterns_json else []
                signals = json.loads(signals_json) if signals_json else {}
                
                # Track pattern performance
                for pattern in patterns:
                    pattern_type = pattern.get('pattern', pattern.get('type', 'unknown'))
                    if pattern_type not in pattern_performance:
                        pattern_performance[pattern_type] = []
                    pattern_performance[pattern_type].append(actual_return)
                
                # Track signal performance
                for signal_name, signal_value in signals.items():
                    if signal_name not in signal_performance:
                        signal_performance[signal_name] = []
                    signal_performance[signal_name].append(actual_return)
            
            # Calculate average performance for each pattern/signal
            pattern_weights = {}
            for pattern, returns in pattern_performance.items():
                avg_return = np.mean(returns)
                success_rate = sum(1 for r in returns if r > 0) / len(returns)
                
                # Weight based on both average return and success rate
                weight = 1.0 + (avg_return * 10) + (success_rate - 0.5) * 2
                pattern_weights[pattern] = max(0.5, min(2.0, weight))
            
            signal_weights = {}
            for signal, returns in signal_performance.items():
                avg_return = np.mean(returns)
                success_rate = sum(1 for r in returns if r > 0) / len(returns)
                
                weight = 1.0 + (avg_return * 10) + (success_rate - 0.5) * 2
                signal_weights[signal] = max(0.5, min(2.0, weight))
            
            # Update the predictor's learning engine with new weights
            if predictor_type == 'short_term':
                self._predictor.short_term_weights = {
                    'patterns': pattern_weights,
                    'signals': signal_weights
                }
            else:
                self._predictor.long_term_weights = {
                    'patterns': pattern_weights,
                    'signals': signal_weights
                }
            
            logger.info(f"Updated {predictor_type} weights based on {len(feedback_data)} samples")
            
        except Exception as e:
            logger.error(f"Error training {predictor_type} predictor: {e}")
    
    def get_performance_summary(self, days=30):
        """Get performance summary for both predictors"""
        try:
            conn = sqlite3.connect(self.tracking_db_path)
            
            # Get recent performance metrics
            query = '''
            SELECT predictor_type, 
                   AVG(win_rate) as avg_win_rate,
                   AVG(avg_return) as avg_return,
                   AVG(sharpe_ratio) as avg_sharpe,
                   MIN(max_drawdown) as worst_drawdown,
                   SUM(total_predictions) as total_predictions,
                   SUM(successful_predictions) as total_successes
            FROM performance_metrics
            WHERE date > datetime('now', '-' || ? || ' days')
            GROUP BY predictor_type
            '''
            
            df = pd.read_sql_query(query, conn, params=(days,))
            
            # Add current active predictions
            active_query = '''
            SELECT prediction_type, COUNT(*) as active_count
            FROM prediction_tracking
            WHERE status = 'active'
            GROUP BY prediction_type
            '''
            
            active_df = pd.read_sql_query(active_query, conn)
            
            conn.close()
            
            # Combine results
            results = []
            for _, row in df.iterrows():
                result = row.to_dict()
                
                # Add active predictions count
                active_row = active_df[active_df['prediction_type'] == row['predictor_type']]
                result['active_predictions'] = active_row['active_count'].iloc[0] if len(active_row) > 0 else 0
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return []
    
    def get_active_predictions(self):
        """Get currently tracked active predictions"""
        try:
            conn = sqlite3.connect(self.tracking_db_path)
            
            query = '''
            SELECT symbol, prediction_type, predicted_score,
                   entry_price, target_price, stop_loss,
                   prediction_date,
                   JULIANDAY('now') - JULIANDAY(prediction_date) as days_held
            FROM prediction_tracking
            WHERE status = 'active'
            ORDER BY prediction_date DESC
            '''
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            # Add current prices and P&L
            results = []
            for _, row in df.iterrows():
                quote = self._fetcher.get_quote(row['symbol'])
                if quote:
                    result = row.to_dict()
                    result['current_price'] = quote['price']
                    result['unrealized_pnl'] = (quote['price'] - row['entry_price']) / row['entry_price']
                    result['unrealized_pnl_pct'] = result['unrealized_pnl'] * 100
                    
                    # Calculate progress to target
                    price_range = row['target_price'] - row['entry_price']
                    current_progress = quote['price'] - row['entry_price']
                    result['target_progress'] = (current_progress / price_range * 100) if price_range > 0 else 0
                    
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting active predictions: {e}")
            return []
    
    def get_historical_performance(self, symbol=None, predictor_type=None, days=90):
        """Get historical performance data"""
        try:
            conn = sqlite3.connect(self.tracking_db_path)
            
            query = '''
            SELECT symbol, prediction_type, prediction_date, exit_date,
                   entry_price, exit_price, actual_return, success,
                   predicted_score, status
            FROM prediction_tracking
            WHERE status != 'active'
            '''
            
            params = []
            if symbol:
                query += ' AND symbol = ?'
                params.append(symbol)
            
            if predictor_type:
                query += ' AND prediction_type = ?'
                params.append(predictor_type)
            
            query += ' AND exit_date > datetime("now", "-" || ? || " days")'
            params.append(days)
            
            query += ' ORDER BY exit_date DESC'
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            return df.to_dict('records')
            
        except Exception as e:
            logger.error(f"Error getting historical performance: {e}")
            return []