// src/web/static/js/stock-performance.js
// Performance Tracking Module
const PerformanceTracker = {
    chartInstance: null,
    updateInterval: null,
    
    // Initialize performance tracking
    init() {
        // Set up auto-refresh every 5 minutes
        this.updateInterval = setInterval(() => {
            if (StockUI.currentTab === 'performance') {
                this.loadPerformanceData();
            }
        }, 300000); // 5 minutes
    },
    
    // Load all performance data
    async loadPerformanceData() {
        try {
            // Load performance summary
            await this.loadPerformanceSummary();
            
            // Load active predictions
            await this.loadActivePredictions();
            
            // Load performance history chart
            await this.loadPerformanceChart();
            
            // Load detailed metrics
            await this.loadDetailedMetrics();
            
            // Update timestamp
            StockUI.updateLastUpdated('performance', new Date().toISOString());
            
        } catch (error) {
            console.error('Error loading performance data:', error);
        }
    },
    
    // Load performance summary
    async loadPerformanceSummary() {
        try {
            const response = await StockAPI.call('/performance/summary?days=30');
            
            if (response.data && response.data.predictors) {
                this.displayPerformanceSummary(response.data.predictors);
            }
        } catch (error) {
            console.error('Error loading performance summary:', error);
            StockUI.showError('performanceSummary', 'Failed to load performance summary');
        }
    },
    
    // Display performance summary cards
    displayPerformanceSummary(predictors) {
        let html = '';
        
        predictors.forEach(predictor => {
            const winRateClass = predictor.avg_win_rate >= 0.6 ? 'positive' : 
                               predictor.avg_win_rate >= 0.4 ? 'neutral' : 'negative';
            const returnClass = predictor.avg_return >= 0 ? 'positive' : 'negative';
            
            html += `
                <div class="indicator-card">
                    <div class="indicator-label">${predictor.predictor_type.replace('_', ' ').toUpperCase()}</div>
                    <div class="indicators-grid" style="margin-top: 10px;">
                        <div>
                            <div class="metric-label">Win Rate</div>
                            <div class="metric-value ${winRateClass}">${(predictor.avg_win_rate * 100).toFixed(1)}%</div>
                        </div>
                        <div>
                            <div class="metric-label">Avg Return</div>
                            <div class="metric-value ${returnClass}">${(predictor.avg_return * 100).toFixed(2)}%</div>
                        </div>
                        <div>
                            <div class="metric-label">Active</div>
                            <div class="metric-value">${predictor.active_predictions || 0}</div>
                        </div>
                        <div>
                            <div class="metric-label">Total Trades</div>
                            <div class="metric-value">${predictor.total_predictions}</div>
                        </div>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${predictor.avg_win_rate * 100}%"></div>
                    </div>
                </div>
            `;
        });
        
        if (predictors.length === 0) {
            html = '<div class="no-data">No performance data available yet. The system is actively tracking predictions.</div>';
        }
        
        document.getElementById('performanceSummary').innerHTML = html;
    },
    
    // Load active predictions
    async loadActivePredictions() {
        try {
            const response = await StockAPI.call('/performance/active');
            
            if (response.data && response.data.predictions) {
                this.displayActivePredictions(response.data.predictions);
            }
        } catch (error) {
            console.error('Error loading active predictions:', error);
            StockUI.showError('activePredictions', 'Failed to load active predictions');
        }
    },
    
    // Display active predictions table
    displayActivePredictions(predictions) {
        if (predictions.length === 0) {
            document.getElementById('activePredictions').innerHTML = 
                '<div class="no-data">No active predictions being tracked</div>';
            return;
        }
        
        let html = `
            <div class="prediction-row prediction-header">
                <div>Symbol</div>
                <div>Type</div>
                <div>Entry</div>
                <div>Current</div>
                <div>Target</div>
                <div>P/L</div>
                <div>Progress</div>
            </div>
        `;
        
        predictions.forEach(pred => {
            const pnlClass = pred.unrealized_pnl >= 0 ? 'prediction-profit' : 'prediction-loss';
            const progressPercent = Math.max(0, Math.min(100, pred.target_progress));
            
            html += `
                <div class="prediction-row" onclick="StockDetail.show('${pred.symbol}')">
                    <div class="stock-symbol">${pred.symbol}</div>
                    <div>${pred.prediction_type.replace('_', ' ')}</div>
                    <div>$${pred.entry_price.toFixed(2)}</div>
                    <div>$${pred.current_price.toFixed(2)}</div>
                    <div>$${pred.target_price.toFixed(2)}</div>
                    <div class="${pnlClass}">${pred.unrealized_pnl_pct.toFixed(2)}%</div>
                    <div>
                        <div class="progress-bar" style="height: 10px;">
                            <div class="progress-fill" style="width: ${progressPercent}%"></div>
                        </div>
                        <div style="font-size: 10px; margin-top: 2px;">${pred.days_held.toFixed(0)} days</div>
                    </div>
                </div>
            `;
        });
        
        document.getElementById('activePredictions').innerHTML = html;
    },
    
    // Load performance history chart
    async loadPerformanceChart() {
        try {
            const response = await StockAPI.call('/performance/history?days=30');
            
            if (response.data && response.data.history) {
                this.drawPerformanceChart(response.data.history);
            }
        } catch (error) {
            console.error('Error loading performance chart:', error);
        }
    },
    
    // Draw performance chart
    drawPerformanceChart(history) {
        const ctx = document.getElementById('performanceChart').getContext('2d');
        
        if (this.chartInstance) {
            this.chartInstance.destroy();
        }
        
        // Group by date and calculate cumulative returns
        const dailyReturns = {};
        const shortTermReturns = {};
        const longTermReturns = {};
        
        history.forEach(trade => {
            const exitDate = new Date(trade.exit_date).toISOString().split('T')[0];
            
            if (!dailyReturns[exitDate]) dailyReturns[exitDate] = [];
            dailyReturns[exitDate].push(trade.actual_return);
            
            if (trade.prediction_type === 'short_term') {
                if (!shortTermReturns[exitDate]) shortTermReturns[exitDate] = [];
                shortTermReturns[exitDate].push(trade.actual_return);
            } else {
                if (!longTermReturns[exitDate]) longTermReturns[exitDate] = [];
                longTermReturns[exitDate].push(trade.actual_return);
            }
        });
        
        // Calculate cumulative returns
        const dates = Object.keys(dailyReturns).sort();
        let cumulative = 1;
        let cumulativeShort = 1;
        let cumulativeLong = 1;
        
        const cumulativeData = [];
        const shortTermData = [];
        const longTermData = [];
        
        dates.forEach(date => {
            const dayReturn = dailyReturns[date].reduce((sum, r) => sum + r, 0) / dailyReturns[date].length;
            cumulative *= (1 + dayReturn);
            cumulativeData.push((cumulative - 1) * 100);
            
            if (shortTermReturns[date]) {
                const shortReturn = shortTermReturns[date].reduce((sum, r) => sum + r, 0) / shortTermReturns[date].length;
                cumulativeShort *= (1 + shortReturn);
            }
            shortTermData.push((cumulativeShort - 1) * 100);
            
            if (longTermReturns[date]) {
                const longReturn = longTermReturns[date].reduce((sum, r) => sum + r, 0) / longTermReturns[date].length;
                cumulativeLong *= (1 + longReturn);
            }
            longTermData.push((cumulativeLong - 1) * 100);
        });
        
        this.chartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: dates,
                datasets: [
                    {
                        label: 'Overall Performance',
                        data: cumulativeData,
                        borderColor: '#0f0',
                        backgroundColor: 'rgba(0, 255, 0, 0.1)',
                        borderWidth: 3,
                        tension: 0.4
                    },
                    {
                        label: 'Short-term Predictor',
                        data: shortTermData,
                        borderColor: '#0ff',
                        backgroundColor: 'rgba(0, 255, 255, 0.1)',
                        borderWidth: 2,
                        tension: 0.4
                    },
                    {
                        label: 'Long-term Predictor',
                        data: longTermData,
                        borderColor: '#ff0',
                        backgroundColor: 'rgba(255, 255, 0, 0.1)',
                        borderWidth: 2,
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                plugins: {
                    legend: {
                        display: true,
                        labels: { color: '#0f0' }
                    },
                    title: {
                        display: true,
                        text: 'Cumulative Performance (%)',
                        color: '#0f0'
                    }
                },
                scales: {
                    x: {
                        display: true,
                        grid: { color: 'rgba(0, 255, 0, 0.1)' },
                        ticks: { color: '#0f0', maxRotation: 45, minRotation: 45 }
                    },
                    y: {
                        display: true,
                        grid: { color: 'rgba(0, 255, 0, 0.1)' },
                        ticks: { 
                            color: '#0f0',
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                }
            }
        });
    },
    
    // Load detailed metrics
    async loadDetailedMetrics() {
        try {
            const response = await StockAPI.call('/performance/tracking-status');
            
            if (response.data) {
                this.displayDetailedMetrics(response.data);
            }
        } catch (error) {
            console.error('Error loading detailed metrics:', error);
            StockUI.showError('detailedMetrics', 'Failed to load detailed metrics');
        }
    },
    
    // Display detailed metrics
    displayDetailedMetrics(status) {
        const metrics = [
            {
                label: 'Tracking Status',
                value: status.is_tracking ? 'Active' : 'Inactive',
                subtitle: `Checking every ${status.check_interval / 60} minutes`,
                valueClass: status.is_tracking ? 'positive' : 'negative'
            },
            {
                label: 'Active Predictions',
                value: status.active_predictions,
                subtitle: 'Currently being tracked'
            },
            {
                label: 'Short-term Win Rate (7d)',
                value: status.performance_summary.find(p => p.predictor_type === 'short_term')?.avg_win_rate 
                    ? (status.performance_summary.find(p => p.predictor_type === 'short_term').avg_win_rate * 100).toFixed(1) + '%'
                    : 'N/A',
                subtitle: 'Last 7 days performance'
            },
            {
                label: 'Long-term Win Rate (7d)',
                value: status.performance_summary.find(p => p.predictor_type === 'long_term')?.avg_win_rate
                    ? (status.performance_summary.find(p => p.predictor_type === 'long_term').avg_win_rate * 100).toFixed(1) + '%'
                    : 'N/A',
                subtitle: 'Last 7 days performance'
            }
        ];
        
        let html = '';
        metrics.forEach(metric => {
            html += `
                <div class="metric-card">
                    <div class="metric-label">${metric.label}</div>
                    <div class="metric-value ${metric.valueClass || ''}">${metric.value}</div>
                    ${metric.subtitle ? `<div class="metric-subtitle">${metric.subtitle}</div>` : ''}
                </div>
            `;
        });
        
        document.getElementById('detailedMetrics').innerHTML = html;
    },
    
    // Manual tracking functions
    async startTracking(symbol, predictionType, score, price) {
        try {
            const response = await StockAPI.call('/performance/start-tracking', {
                method: 'POST',
                data: {
                    symbol: symbol,
                    prediction_type: predictionType,
                    score: score,
                    price: price
                }
            });
            
            if (response.status === 'success') {
                alert(`Started tracking ${predictionType} prediction for ${symbol}`);
                this.loadPerformanceData();
            }
        } catch (error) {
            console.error('Error starting tracking:', error);
            alert('Failed to start tracking');
        }
    }
};

// Make PerformanceTracker available globally
window.PerformanceTracker = PerformanceTracker;

// Make PerformanceTracker available globally
if (typeof window !== 'undefined') {
    window.PerformanceTracker = PerformanceTracker;
}