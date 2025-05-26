// stock-detail.js - Stock Detail Modal Module
const StockDetail = {
    chartInstance: null,

    // Show stock detail modal
    async show(symbol) {
        document.getElementById('stockModal').style.display = 'block';
        StockUI.showLoading('modalContent', 'Loading stock details...');

        try {
            // Fetch both detail and analysis
            const [detailResponse, analysisResponse] = await Promise.all([
                StockAPI.getStockDetail(symbol),
                StockAPI.getStockAnalysis(symbol)
            ]);

            console.log('Detail response:', detailResponse);
            console.log('Analysis response:', analysisResponse);

            // Extract data
            const stockData = detailResponse.data;
            const analysisData = analysisResponse.data;

            this.display(stockData, analysisData);
        } catch (error) {
            console.error('Error loading stock detail:', error);
            StockUI.showError('modalContent', 'Failed to load stock details', 
                error.response?.data?.error || error.message);
        }
    },

    // Display stock modal content
    display(stockData, analysisData) {
        if (!stockData || !stockData.quote) {
            StockUI.showError('modalContent', 'Invalid stock data received');
            return;
        }

        const changeClass = stockData.quote.change >= 0 ? 'positive' : 'negative';
        const arrow = stockData.quote.change >= 0 ? '▲' : '▼';

        // Check if in portfolio
        const inPortfolio = StockPortfolio.portfolio[stockData.symbol] !== undefined;
        const portfolioButton = inPortfolio
            ? `<button onclick="StockPortfolio.removeAndRefresh('${stockData.symbol}'); StockDetail.close();" style="background: #f00;">Remove from Portfolio</button>`
            : `<button onclick="StockDetail.addToPortfolio('${stockData.symbol}', ${stockData.quote.price});">Add to Portfolio</button>`;

        let html = `
            <h2>${stockData.symbol} - ${stockData.name}</h2>
            <div style="margin-bottom: 20px;">${portfolioButton}</div>
            
            <div class="indicators-grid">
                <div class="indicator-card">
                    <div class="indicator-label">Current Price</div>
                    <div class="indicator-value">$${stockData.quote.price.toFixed(2)}</div>
                    <div class="${changeClass}">
                        ${arrow} ${Math.abs(stockData.quote.change).toFixed(2)} (${stockData.quote.change_percent})
                    </div>
                </div>
                <div class="indicator-card">
                    <div class="indicator-label">Volume</div>
                    <div class="indicator-value">${(stockData.quote.volume / 1000000).toFixed(2)}M</div>
                </div>
                <div class="indicator-card">
                    <div class="indicator-label">RSI</div>
                    <div class="indicator-value">${
                        analysisData?.current_indicators?.rsi?.toFixed(2) || 'N/A'
                    }</div>
                </div>
                <div class="indicator-card">
                    <div class="indicator-label">AI Score</div>
                    <div class="indicator-value" style="color: ${
                        (analysisData?.score || 0) >= 7 ? '#0f0' :
                        (analysisData?.score || 0) >= 5 ? '#ff0' : '#f90'
                    }">
                        ${analysisData?.score || 0}/10
                    </div>
                </div>
            </div>
            
            <!-- Chart Container -->
            <div class="chart-container">
                <canvas id="stockChart"></canvas>
            </div>
        `;

        // Add AI Analysis if available
        if (analysisData) {
            html += this.renderAnalysis(analysisData);
        }

        // Add Fundamentals if available
        if (stockData.fundamentals) {
            html += this.renderFundamentals(stockData.fundamentals);
        }

        // Add News if available
        if (stockData.news && stockData.news.length > 0) {
            html += this.renderNews(stockData.news);
        }

        document.getElementById('modalContent').innerHTML = html;

        // Draw chart if we have data
        if (stockData.daily_data && stockData.daily_data.length > 0) {
            this.drawChart(stockData.daily_data);
        }
    },

    // Render analysis section
    renderAnalysis(analysisData) {
        return `
            <div class="analysis-section">
                <h3 class="analysis-header">AI Analysis</h3>
                <div style="margin-bottom: 15px;">
                    <strong>Signals Detected:</strong>
                    <ul style="margin-left: 20px; margin-top: 10px;">
                        ${(analysisData.reasons || []).map(r => `<li>${r}</li>`).join('')}
                    </ul>
                </div>
                
                <div style="margin-bottom: 15px;">
                    <strong>Entry/Exit Suggestions:</strong>
                    <div style="margin-top: 10px;">
                        Support Entry: $${analysisData.suggestions?.support_entry?.toFixed(2) || 'N/A'}<br>
                        Resistance Exit: $${analysisData.suggestions?.resistance_exit?.toFixed(2) || 'N/A'}<br>
                        Stop Loss: $${analysisData.suggestions?.stop_loss?.toFixed(2) || 'N/A'}
                    </div>
                </div>
            </div>
        `;
    },

    // Render fundamentals section
    renderFundamentals(fundamentals) {
        return `
            <div class="analysis-section">
                <h3 class="analysis-header">Fundamentals</h3>
                <div class="indicators-grid">
                    <div class="indicator-card">
                        <div class="indicator-label">Market Cap</div>
                        <div class="indicator-value">$${(fundamentals.market_cap / 1000000000).toFixed(2)}B</div>
                    </div>
                    <div class="indicator-card">
                        <div class="indicator-label">P/E Ratio</div>
                        <div class="indicator-value">${fundamentals.pe_ratio?.toFixed(2) || 'N/A'}</div>
                    </div>
                    <div class="indicator-card">
                        <div class="indicator-label">52W High</div>
                        <div class="indicator-value">$${fundamentals['52_week_high']?.toFixed(2) || 'N/A'}</div>
                    </div>
                    <div class="indicator-card">
                        <div class="indicator-label">52W Low</div>
                        <div class="indicator-value">$${fundamentals['52_week_low']?.toFixed(2) || 'N/A'}</div>
                    </div>
                </div>
            </div>
        `;
    },

    // Render news section
    renderNews(news) {
        return `
            <div class="analysis-section">
                <h3 class="analysis-header">Latest News</h3>
                <div class="news-section">
                    ${news.slice(0, 3).map(item => `
                        <div class="news-item">
                            <div class="news-title">${item.title}</div>
                            <div class="news-meta">${item.publisher} - ${new Date(item.timestamp * 1000).toLocaleDateString()}</div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    },

    // Draw stock chart
    drawChart(priceData) {
        const ctx = document.getElementById('stockChart').getContext('2d');

        // Destroy existing chart
        if (this.chartInstance) {
            this.chartInstance.destroy();
        }

        const labels = priceData.map(d => d.date);
        const prices = priceData.map(d => d.close);
        const volumes = priceData.map(d => d.volume);

        this.chartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Price',
                        data: prices,
                        borderColor: '#0f0',
                        backgroundColor: 'rgba(0, 255, 0, 0.1)',
                        borderWidth: 2,
                        tension: 0.4,
                        yAxisID: 'y-price'
                    },
                    {
                        label: 'Volume',
                        data: volumes,
                        type: 'bar',
                        backgroundColor: 'rgba(0, 255, 0, 0.3)',
                        yAxisID: 'y-volume'
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
                    }
                },
                scales: {
                    x: {
                        display: true,
                        grid: { color: 'rgba(0, 255, 0, 0.1)' },
                        ticks: { color: '#0f0', maxRotation: 45, minRotation: 45 }
                    },
                    'y-price': {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        grid: { color: 'rgba(0, 255, 0, 0.1)' },
                        ticks: { color: '#0f0' }
                    },
                    'y-volume': {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        grid: { drawOnChartArea: false },
                        ticks: { color: '#0f0' }
                    }
                }
            }
        });
    },

    // Add to portfolio helper
    addToPortfolio(symbol, price) {
        document.getElementById('portfolioAddSymbol').value = symbol;
        document.getElementById('portfolioAddPrice').value = price;
        this.close();
        StockApp.switchTab('portfolio');
    },

    // Close modal
    close() {
        document.getElementById('stockModal').style.display = 'none';
        if (this.chartInstance) {
            this.chartInstance.destroy();
            this.chartInstance = null;
        }
    }
};