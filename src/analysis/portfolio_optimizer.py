# src/analysis/portfolio_optimizer.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import logging
from scipy.optimize import minimize
from scipy.stats import norm
import json

logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """Advanced portfolio optimization using Modern Portfolio Theory and risk management"""
    
    def __init__(self):
        self.risk_free_rate = 0.05  # 5% annual risk-free rate (Treasury)
        self.trading_days = 252
        self.min_position_size = 0.01  # 1% minimum position
        self.max_position_size = 0.30  # 30% maximum position
        
        # Risk profiles
        self.risk_profiles = {
            'conservative': {
                'target_volatility': 0.08,
                'max_drawdown': 0.10,
                'sharpe_target': 1.0,
                'allocation_constraints': {
                    'stocks': (0.3, 0.5),
                    'bonds': (0.4, 0.6),
                    'alternatives': (0, 0.2)
                }
            },
            'moderate': {
                'target_volatility': 0.12,
                'max_drawdown': 0.15,
                'sharpe_target': 1.2,
                'allocation_constraints': {
                    'stocks': (0.5, 0.7),
                    'bonds': (0.2, 0.4),
                    'alternatives': (0, 0.3)
                }
            },
            'aggressive': {
                'target_volatility': 0.18,
                'max_drawdown': 0.25,
                'sharpe_target': 1.5,
                'allocation_constraints': {
                    'stocks': (0.7, 0.95),
                    'bonds': (0, 0.2),
                    'alternatives': (0, 0.3)
                }
            },
            'growth': {
                'target_volatility': 0.25,
                'max_drawdown': 0.35,
                'sharpe_target': 1.3,
                'allocation_constraints': {
                    'stocks': (0.8, 1.0),
                    'bonds': (0, 0.1),
                    'alternatives': (0, 0.2)
                }
            }
        }
    
    def optimize_portfolio(self, symbols, capital=10000, risk_profile='moderate', 
                         constraints=None, period='1y'):
        """Optimize portfolio allocation using various strategies"""
        try:
            # Get historical data
            returns_data = self._get_returns_data(symbols, period)
            if returns_data.empty:
                logger.error("No returns data available for optimization")
                return None
            
            # Calculate statistics
            expected_returns = returns_data.mean() * self.trading_days
            cov_matrix = returns_data.cov() * self.trading_days
            
            # Get risk profile settings
            profile = self.risk_profiles.get(risk_profile, self.risk_profiles['moderate'])
            
            # Run different optimization strategies
            results = {
                'symbols': symbols,
                'capital': capital,
                'risk_profile': risk_profile,
                'optimizations': {}
            }
            
            # 1. Maximum Sharpe Ratio
            sharpe_result = self._optimize_sharpe_ratio(
                expected_returns, cov_matrix, constraints
            )
            results['optimizations']['max_sharpe'] = sharpe_result
            
            # 2. Minimum Volatility
            min_vol_result = self._optimize_min_volatility(
                expected_returns, cov_matrix, constraints
            )
            results['optimizations']['min_volatility'] = min_vol_result
            
            # 3. Risk Parity
            risk_parity_result = self._optimize_risk_parity(
                cov_matrix, constraints
            )
            results['optimizations']['risk_parity'] = risk_parity_result
            
            # 4. Target Return
            target_return = expected_returns.mean()
            target_result = self._optimize_target_return(
                expected_returns, cov_matrix, target_return, constraints
            )
            results['optimizations']['target_return'] = target_result
            
            # 5. Maximum Diversification
            max_div_result = self._optimize_max_diversification(
                cov_matrix, constraints
            )
            results['optimizations']['max_diversification'] = max_div_result
            
            # Calculate efficient frontier
            results['efficient_frontier'] = self._calculate_efficient_frontier(
                expected_returns, cov_matrix
            )
            
            # Recommend best portfolio based on risk profile
            results['recommended'] = self._select_best_portfolio(
                results['optimizations'], profile
            )
            
            # Add risk metrics for recommended portfolio
            if results['recommended']:
                results['recommended']['risk_metrics'] = self._calculate_risk_metrics(
                    results['recommended']['weights'],
                    returns_data,
                    expected_returns,
                    cov_matrix
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Error optimizing portfolio: {e}")
            return None
    
    def _get_returns_data(self, symbols, period='1y'):
        """Get historical returns data for symbols"""
        try:
            data = {}
            
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                
                if not hist.empty:
                    # Calculate daily returns
                    returns = hist['Close'].pct_change().dropna()
                    data[symbol] = returns
            
            # Create DataFrame with aligned dates
            if data:
                returns_df = pd.DataFrame(data)
                # Remove any rows with NaN values
                returns_df = returns_df.dropna()
                return returns_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error getting returns data: {e}")
            return pd.DataFrame()
    
    def _optimize_sharpe_ratio(self, expected_returns, cov_matrix, constraints=None):
        """Optimize for maximum Sharpe ratio"""
        n_assets = len(expected_returns)
        
        # Objective function (negative Sharpe ratio for minimization)
        def neg_sharpe(weights):
            port_return = np.dot(weights, expected_returns)
            port_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            sharpe = (port_return - self.risk_free_rate) / port_vol
            return -sharpe
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Sum to 1
        
        # Bounds
        bounds = tuple((self.min_position_size, self.max_position_size) 
                      for _ in range(n_assets))
        
        # Initial guess (equal weight)
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(neg_sharpe, x0, method='SLSQP', 
                         bounds=bounds, constraints=cons)
        
        if result.success:
            weights = result.x
            port_return = np.dot(weights, expected_returns)
            port_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            sharpe = (port_return - self.risk_free_rate) / port_vol
            
            return {
                'weights': dict(zip(expected_returns.index, weights)),
                'expected_return': port_return,
                'volatility': port_vol,
                'sharpe_ratio': sharpe,
                'success': True
            }
        else:
            return {'success': False, 'message': result.message}
    
    def _optimize_min_volatility(self, expected_returns, cov_matrix, constraints=None):
        """Optimize for minimum volatility"""
        n_assets = len(expected_returns)
        
        # Objective function (portfolio variance)
        def portfolio_variance(weights):
            return np.dot(weights, np.dot(cov_matrix, weights))
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # Bounds
        bounds = tuple((self.min_position_size, self.max_position_size) 
                      for _ in range(n_assets))
        
        # Initial guess
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(portfolio_variance, x0, method='SLSQP',
                         bounds=bounds, constraints=cons)
        
        if result.success:
            weights = result.x
            port_return = np.dot(weights, expected_returns)
            port_vol = np.sqrt(portfolio_variance(weights))
            sharpe = (port_return - self.risk_free_rate) / port_vol
            
            return {
                'weights': dict(zip(expected_returns.index, weights)),
                'expected_return': port_return,
                'volatility': port_vol,
                'sharpe_ratio': sharpe,
                'success': True
            }
        else:
            return {'success': False, 'message': result.message}
    
    def _optimize_risk_parity(self, cov_matrix, constraints=None):
        """Optimize for risk parity (equal risk contribution)"""
        n_assets = len(cov_matrix)
        
        # Objective function
        def risk_parity_objective(weights):
            # Calculate portfolio volatility
            port_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            
            # Calculate marginal contributions to risk
            marginal_contrib = np.dot(cov_matrix, weights) / port_vol
            
            # Calculate contributions to risk
            contrib = weights * marginal_contrib
            
            # We want equal risk contribution
            # Minimize the sum of squared differences from the mean contribution
            mean_contrib = np.mean(contrib)
            return np.sum((contrib - mean_contrib)**2)
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # Bounds
        bounds = tuple((0.01, 0.5) for _ in range(n_assets))
        
        # Initial guess
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(risk_parity_objective, x0, method='SLSQP',
                         bounds=bounds, constraints=cons)
        
        if result.success:
            weights = result.x
            port_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            
            # Calculate risk contributions
            marginal_contrib = np.dot(cov_matrix, weights) / port_vol
            risk_contribs = weights * marginal_contrib
            
            return {
                'weights': dict(zip(cov_matrix.index, weights)),
                'volatility': port_vol,
                'risk_contributions': dict(zip(cov_matrix.index, risk_contribs)),
                'success': True
            }
        else:
            return {'success': False, 'message': result.message}
    
    def _optimize_target_return(self, expected_returns, cov_matrix, 
                               target_return, constraints=None):
        """Optimize for minimum volatility given a target return"""
        n_assets = len(expected_returns)
        
        # Objective function (portfolio variance)
        def portfolio_variance(weights):
            return np.dot(weights, np.dot(cov_matrix, weights))
        
        # Constraints
        cons = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.dot(x, expected_returns) - target_return}
        ]
        
        # Bounds
        bounds = tuple((self.min_position_size, self.max_position_size) 
                      for _ in range(n_assets))
        
        # Initial guess
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(portfolio_variance, x0, method='SLSQP',
                         bounds=bounds, constraints=cons)
        
        if result.success:
            weights = result.x
            port_vol = np.sqrt(portfolio_variance(weights))
            sharpe = (target_return - self.risk_free_rate) / port_vol
            
            return {
                'weights': dict(zip(expected_returns.index, weights)),
                'expected_return': target_return,
                'volatility': port_vol,
                'sharpe_ratio': sharpe,
                'success': True
            }
        else:
            return {'success': False, 'message': result.message}
    
    def _optimize_max_diversification(self, cov_matrix, constraints=None):
        """Optimize for maximum diversification ratio"""
        n_assets = len(cov_matrix)
        
        # Get standard deviations
        std_devs = np.sqrt(np.diag(cov_matrix))
        
        # Objective function (negative diversification ratio)
        def neg_diversification_ratio(weights):
            # Weighted average of individual volatilities
            avg_vol = np.dot(weights, std_devs)
            # Portfolio volatility
            port_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            # Diversification ratio
            div_ratio = avg_vol / port_vol
            return -div_ratio
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # Bounds
        bounds = tuple((self.min_position_size, self.max_position_size) 
                      for _ in range(n_assets))
        
        # Initial guess
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(neg_diversification_ratio, x0, method='SLSQP',
                         bounds=bounds, constraints=cons)
        
        if result.success:
            weights = result.x
            avg_vol = np.dot(weights, std_devs)
            port_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            div_ratio = avg_vol / port_vol
            
            return {
                'weights': dict(zip(cov_matrix.index, weights)),
                'volatility': port_vol,
                'diversification_ratio': div_ratio,
                'success': True
            }
        else:
            return {'success': False, 'message': result.message}
    
    def _calculate_efficient_frontier(self, expected_returns, cov_matrix, n_points=50):
        """Calculate the efficient frontier"""
        # Get min and max possible returns
        min_ret = expected_returns.min()
        max_ret = expected_returns.max()
        
        # Generate target returns
        target_returns = np.linspace(min_ret, max_ret, n_points)
        
        frontier = {
            'returns': [],
            'volatilities': [],
            'sharpe_ratios': []
        }
        
        for target in target_returns:
            result = self._optimize_target_return(
                expected_returns, cov_matrix, target
            )
            
            if result['success']:
                frontier['returns'].append(result['expected_return'])
                frontier['volatilities'].append(result['volatility'])
                frontier['sharpe_ratios'].append(result['sharpe_ratio'])
        
        return frontier
    
    def _select_best_portfolio(self, optimizations, risk_profile):
        """Select best portfolio based on risk profile"""
        candidates = []
        
        # Filter successful optimizations
        for name, opt in optimizations.items():
            if opt.get('success', False):
                candidates.append((name, opt))
        
        if not candidates:
            return None
        
        # Score each portfolio based on risk profile preferences
        best_score = -np.inf
        best_portfolio = None
        
        for name, portfolio in candidates:
            score = 0
            
            # Sharpe ratio score
            if 'sharpe_ratio' in portfolio:
                sharpe_score = portfolio['sharpe_ratio'] / risk_profile['sharpe_target']
                score += sharpe_score * 0.4
            
            # Volatility score (inverse - lower is better)
            if 'volatility' in portfolio:
                vol_score = risk_profile['target_volatility'] / portfolio['volatility']
                score += vol_score * 0.3
            
            # Diversification score
            if 'diversification_ratio' in portfolio:
                score += portfolio['diversification_ratio'] * 0.2
            
            # Risk parity bonus
            if name == 'risk_parity':
                score += 0.1
            
            if score > best_score:
                best_score = score
                best_portfolio = portfolio.copy()
                best_portfolio['optimization_method'] = name
                best_portfolio['score'] = score
        
        return best_portfolio
    
    def _calculate_risk_metrics(self, weights, returns_data, expected_returns, cov_matrix):
        """Calculate comprehensive risk metrics for a portfolio"""
        try:
            # Convert weights dict to array aligned with returns data
            weight_array = np.array([weights[symbol] for symbol in returns_data.columns])
            
            # Portfolio returns time series
            portfolio_returns = returns_data.dot(weight_array)
            
            # Basic metrics
            metrics = {
                'expected_return': np.dot(weight_array, expected_returns),
                'volatility': np.sqrt(np.dot(weight_array, np.dot(cov_matrix, weight_array))),
                'skewness': portfolio_returns.skew(),
                'kurtosis': portfolio_returns.kurtosis()
            }
            
            # Sharpe ratio
            metrics['sharpe_ratio'] = (metrics['expected_return'] - self.risk_free_rate) / metrics['volatility']
            
            # Sortino ratio (downside deviation)
            downside_returns = portfolio_returns[portfolio_returns < 0]
            if len(downside_returns) > 0:
                downside_deviation = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(self.trading_days)
                metrics['sortino_ratio'] = (metrics['expected_return'] - self.risk_free_rate) / downside_deviation
            else:
                metrics['sortino_ratio'] = np.inf
            
            # Maximum drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - running_max) / running_max
            metrics['max_drawdown'] = drawdowns.min()
            
            # Value at Risk (VaR) - 95% confidence
            metrics['var_95'] = np.percentile(portfolio_returns, 5) * np.sqrt(self.trading_days)
            
            # Conditional Value at Risk (CVaR)
            var_threshold = np.percentile(portfolio_returns, 5)
            cvar_returns = portfolio_returns[portfolio_returns <= var_threshold]
            if len(cvar_returns) > 0:
                metrics['cvar_95'] = cvar_returns.mean() * np.sqrt(self.trading_days)
            else:
                metrics['cvar_95'] = metrics['var_95']
            
            # Calmar ratio
            if metrics['max_drawdown'] != 0:
                metrics['calmar_ratio'] = metrics['expected_return'] / abs(metrics['max_drawdown'])
            else:
                metrics['calmar_ratio'] = np.inf
            
            # Beta (vs market - using first symbol as proxy)
            if len(returns_data.columns) > 0:
                market_returns = returns_data.iloc[:, 0]  # First symbol as market proxy
                covariance = portfolio_returns.cov(market_returns)
                market_variance = market_returns.var()
                if market_variance > 0:
                    metrics['beta'] = covariance / market_variance
                else:
                    metrics['beta'] = 0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def rebalance_portfolio(self, current_holdings, target_weights, current_prices, 
                          capital=None, threshold=0.05):
        """Calculate rebalancing trades"""
        try:
            # Calculate current values and weights
            current_values = {}
            total_value = 0
            
            for symbol, shares in current_holdings.items():
                if symbol in current_prices:
                    value = shares * current_prices[symbol]
                    current_values[symbol] = value
                    total_value += value
            
            if capital:
                total_value = capital
            
            # Calculate current weights
            current_weights = {}
            if total_value > 0:
                current_weights = {
                    symbol: value / total_value 
                    for symbol, value in current_values.items()
                }
            
            # Calculate required trades
            trades = []
            
            for symbol in set(list(current_holdings.keys()) + list(target_weights.keys())):
                current_weight = current_weights.get(symbol, 0)
                target_weight = target_weights.get(symbol, 0)
                
                # Check if rebalancing is needed (exceeds threshold)
                if abs(current_weight - target_weight) > threshold:
                    current_value = current_values.get(symbol, 0)
                    target_value = target_weight * total_value
                    
                    trade_value = target_value - current_value
                    
                    if symbol in current_prices and current_prices[symbol] > 0:
                        shares_to_trade = int(trade_value / current_prices[symbol])
                        
                        if shares_to_trade != 0:
                            trades.append({
                                'symbol': symbol,
                                'action': 'BUY' if shares_to_trade > 0 else 'SELL',
                                'shares': abs(shares_to_trade),
                                'price': current_prices[symbol],
                                'value': abs(trade_value),
                                'current_weight': current_weight,
                                'target_weight': target_weight
                            })
            
            # Calculate transaction costs (simplified)
            total_trades_value = sum(trade['value'] for trade in trades)
            estimated_costs = total_trades_value * 0.001  # 0.1% transaction cost
            
            return {
                'trades': trades,
                'current_weights': current_weights,
                'target_weights': target_weights,
                'total_value': total_value,
                'estimated_costs': estimated_costs,
                'rebalancing_needed': len(trades) > 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating rebalancing: {e}")
            return None
    
    def analyze_portfolio_performance(self, holdings, start_date=None, end_date=None):
        """Analyze historical performance of a portfolio"""
        try:
            if not holdings:
                return None
            
            # Get historical data
            symbols = list(holdings.keys())
            weights = np.array(list(holdings.values()))
            weights = weights / weights.sum()  # Normalize
            
            # Fetch data
            data = {}
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date)
                if not hist.empty:
                    data[symbol] = hist['Close']
            
            if not data:
                return None
            
            # Create price DataFrame
            prices_df = pd.DataFrame(data)
            returns_df = prices_df.pct_change().dropna()
            
            # Calculate portfolio returns
            portfolio_returns = returns_df.dot(weights)
            
            # Calculate metrics
            total_return = (prices_df.iloc[-1] / prices_df.iloc[0] - 1).dot(weights)
            annualized_return = (1 + total_return) ** (252 / len(prices_df)) - 1
            volatility = portfolio_returns.std() * np.sqrt(252)
            
            if volatility > 0:
                sharpe_ratio = (annualized_return - self.risk_free_rate) / volatility
            else:
                sharpe_ratio = 0
            
            # Calculate cumulative returns
            cumulative_returns = (1 + portfolio_returns).cumprod()
            
            # Maximum drawdown
            running_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdowns.min()
            
            # Compare to benchmark (S&P 500)
            benchmark = yf.Ticker('^GSPC')
            benchmark_hist = benchmark.history(start=start_date, end=end_date)
            
            alpha = beta = benchmark_total = None
            
            if not benchmark_hist.empty:
                benchmark_returns = benchmark_hist['Close'].pct_change().dropna()
                benchmark_total = (benchmark_hist['Close'].iloc[-1] / 
                                 benchmark_hist['Close'].iloc[0] - 1)
                
                # Calculate alpha and beta
                aligned_dates = portfolio_returns.index.intersection(benchmark_returns.index)
                if len(aligned_dates) > 20:
                    port_ret_aligned = portfolio_returns[aligned_dates]
                    bench_ret_aligned = benchmark_returns[aligned_dates]
                    
                    covariance = port_ret_aligned.cov(bench_ret_aligned)
                    benchmark_variance = bench_ret_aligned.var()
                    
                    if benchmark_variance > 0:
                        beta = covariance / benchmark_variance
                        alpha = annualized_return - (self.risk_free_rate + 
                                                   beta * (benchmark_total - self.risk_free_rate))
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'benchmark_return': benchmark_total,
                'alpha': alpha,
                'beta': beta,
                'cumulative_returns': cumulative_returns.to_dict(),
                'period_start': prices_df.index[0].strftime('%Y-%m-%d'),
                'period_end': prices_df.index[-1].strftime('%Y-%m-%d'),
                'trading_days': len(prices_df)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing portfolio performance: {e}")
            return None
    
    def suggest_portfolio_improvements(self, current_portfolio, market_conditions=None):
        """Suggest improvements to current portfolio"""
        try:
            suggestions = {
                'rebalancing': [],
                'risk_adjustments': [],
                'diversification': [],
                'tax_optimization': []
            }
            
            # Analyze current portfolio
            if not current_portfolio:
                return suggestions
            
            # Get current weights
            total_value = sum(current_portfolio.values())
            if total_value > 0:
                weights = {k: v/total_value for k, v in current_portfolio.items()}
            else:
                return suggestions
            
            # Check concentration risk
            for symbol, weight in weights.items():
                if weight > 0.25:
                    suggestions['risk_adjustments'].append({
                        'issue': 'Concentration Risk',
                        'symbol': symbol,
                        'current_weight': weight,
                        'recommendation': f'Consider reducing {symbol} position to max 25%',
                        'priority': 'high'
                    })
            
            # Check diversification
            if len(weights) < 5:
                suggestions['diversification'].append({
                    'issue': 'Low Diversification',
                    'current_holdings': len(weights),
                    'recommendation': 'Consider adding more positions (minimum 5-7 for proper diversification)',
                    'priority': 'medium'
                })
            
            # Sector diversification
            # This would need sector mapping in production
            suggestions['diversification'].append({
                'issue': 'Sector Concentration Check',
                'recommendation': 'Ensure exposure across multiple sectors',
                'priority': 'medium'
            })
            
            # Tax loss harvesting opportunities
            # In production, this would check unrealized gains/losses
            suggestions['tax_optimization'].append({
                'recommendation': 'Review positions with losses for tax-loss harvesting opportunities',
                'priority': 'low'
            })
            
            # Market conditions adjustments
            if market_conditions:
                if market_conditions.get('vix', 20) > 30:
                    suggestions['risk_adjustments'].append({
                        'issue': 'High Market Volatility',
                        'recommendation': 'Consider reducing overall exposure or adding defensive positions',
                        'priority': 'high'
                    })
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error suggesting portfolio improvements: {e}")
            return {
                'rebalancing': [],
                'risk_adjustments': [],
                'diversification': [],
                'tax_optimization': []
            }