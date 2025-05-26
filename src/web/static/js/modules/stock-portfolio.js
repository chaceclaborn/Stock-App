// stock-portfolio.js - Portfolio Management Module
const StockPortfolio = {
    // Portfolio data
    portfolio: {},
    
    // Initialize portfolio
    init() {
        this.portfolio = JSON.parse(localStorage.getItem('stockPortfolio')) || {};
    },

    // Save portfolio to localStorage
    save() {
        localStorage.setItem('stockPortfolio', JSON.stringify(this.portfolio));
    },

    // Add stock to portfolio
    add(symbol, shares, price) {
        symbol = symbol.toUpperCase();
        
        if (!symbol || !shares || !price || shares <= 0 || price <= 0) {
            alert('Please enter valid symbol, shares, and price');
            return false;
        }

        if (this.portfolio[symbol]) {
            // Update existing holding
            const totalShares = this.portfolio[symbol].shares + shares;
            const totalCost = this.portfolio[symbol].shares * this.portfolio[symbol].avgPrice + shares * price;
            this.portfolio[symbol] = {
                shares: totalShares,
                avgPrice: totalCost / totalShares
            };
        } else {
            // Add new holding
            this.portfolio[symbol] = {
                shares: shares,
                avgPrice: price
            };
        }

        this.save();
        return true;
    },

    // Remove stock from portfolio
    remove(symbol) {
        if (confirm(`Remove ${symbol} from portfolio?`)) {
            delete this.portfolio[symbol];
            this.save();
            return true;
        }
        return false;
    },

    // Get portfolio value
    async calculateValue(cachedStocks) {
        let totalValue = 0;
        let totalCost = 0;
        let dayChange = 0;

        for (const [symbol, holding] of Object.entries(this.portfolio)) {
            let currentPrice = 0;
            let change = 0;

            if (cachedStocks[symbol]) {
                currentPrice = cachedStocks[symbol].price;
                change = cachedStocks[symbol].change;
            }

            const currentValue = currentPrice * holding.shares;
            const costBasis = holding.avgPrice * holding.shares;

            totalValue += currentValue;
            totalCost += costBasis;
            dayChange += change * holding.shares;
        }

        const totalReturn = totalCost > 0 ? ((totalValue - totalCost) / totalCost) * 100 : 0;

        return {
            totalValue,
            totalCost,
            dayChange,
            totalReturn
        };
    },

    // Display portfolio
    async display(cachedStocks) {
        const container = document.getElementById('portfolioGrid');
        
        if (Object.keys(this.portfolio).length === 0) {
            container.innerHTML = '<div class="no-portfolio">No stocks in portfolio. Add some to get started!</div>';
            this.updateSummary(0, 0, 0);
            return;
        }

        let html = '';
        let totalValue = 0;
        let totalCost = 0;
        let dayChange = 0;

        for (const [symbol, holding] of Object.entries(this.portfolio)) {
            let currentPrice = 0;
            let change = 0;
            let changePercent = '0.00%';

            if (cachedStocks[symbol]) {
                currentPrice = cachedStocks[symbol].price;
                change = cachedStocks[symbol].change;
                changePercent = cachedStocks[symbol].change_percent;
            }

            const currentValue = currentPrice * holding.shares;
            const costBasis = holding.avgPrice * holding.shares;
            const gain = currentValue - costBasis;
            const gainPercent = costBasis > 0 ? (gain / costBasis) * 100 : 0;

            totalValue += currentValue;
            totalCost += costBasis;
            dayChange += change * holding.shares;

            const gainClass = gain >= 0 ? 'positive' : 'negative';

            html += `
                <div class="portfolio-stock-card">
                    <button class="remove-from-portfolio" onclick="StockPortfolio.removeAndRefresh('${symbol}')">×</button>
                    <div class="stock-symbol">${symbol}</div>
                    <div class="stock-name">${holding.shares} shares @ $${holding.avgPrice.toFixed(2)}</div>
                    <div class="stock-price">$${currentPrice.toFixed(2)}</div>
                    <div class="stock-change ${gainClass}">
                        P/L: $${gain.toFixed(2)} (${gainPercent.toFixed(2)}%)
                    </div>
                    <div style="margin-top: 10px; font-size: 14px;">
                        Value: $${currentValue.toFixed(2)}
                    </div>
                </div>
            `;
        }

        container.innerHTML = html;

        const totalReturn = totalCost > 0 ? ((totalValue - totalCost) / totalCost) * 100 : 0;
        this.updateSummary(totalValue, dayChange, totalReturn);
    },

    // Update portfolio summary
    updateSummary(totalValue, dayChange, totalReturn) {
        document.getElementById('portfolioTotalValue').textContent = `$${totalValue.toFixed(2)}`;
        
        const dayChangeEl = document.getElementById('portfolioDayChange');
        dayChangeEl.textContent = `$${dayChange.toFixed(2)}`;
        dayChangeEl.className = dayChange >= 0 ? 'indicator-value positive' : 'indicator-value negative';
        
        const totalReturnEl = document.getElementById('portfolioTotalReturn');
        totalReturnEl.textContent = `${totalReturn.toFixed(2)}%`;
        totalReturnEl.className = totalReturn >= 0 ? 'indicator-value positive' : 'indicator-value negative';
    },

    // Add from form
    addFromForm() {
        const symbol = document.getElementById('portfolioAddSymbol').value.toUpperCase();
        const shares = parseInt(document.getElementById('portfolioAddShares').value);
        const price = parseFloat(document.getElementById('portfolioAddPrice').value);

        if (this.add(symbol, shares, price)) {
            // Clear inputs
            document.getElementById('portfolioAddSymbol').value = '';
            document.getElementById('portfolioAddShares').value = '';
            document.getElementById('portfolioAddPrice').value = '';
            
            // Refresh display
            StockApp.loadPortfolio();
        }
    },

    // Remove and refresh
    removeAndRefresh(symbol) {
        if (this.remove(symbol)) {
            StockApp.loadPortfolio();
        }
    }
};