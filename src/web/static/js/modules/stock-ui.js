// stock-ui.js - UI Management Module
const StockUI = {
    // UI state
    currentTab: 'overview',
    
    // Show loading state
    showLoading(containerId, message = 'Loading...') {
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = `<div class="loading">${message}</div>`;
        }
    },

    // Show error message
    showError(containerId, message, details = null) {
        const container = document.getElementById(containerId);
        if (container) {
            let errorHtml = `<div class="error">${message}`;
            if (details) {
                errorHtml += `<br><small style="opacity: 0.7">${details}</small>`;
            }
            errorHtml += `</div>`;
            container.innerHTML = errorHtml;
        }
    },

    // Format time ago
    formatTimeAgo(timestamp) {
        const now = new Date();
        const then = new Date(timestamp);
        const seconds = Math.floor((now - then) / 1000);

        if (seconds < 60) return `${seconds} seconds ago`;
        const minutes = Math.floor(seconds / 60);
        if (minutes < 60) return `${minutes} minute${minutes > 1 ? 's' : ''} ago`;
        const hours = Math.floor(minutes / 60);
        if (hours < 24) return `${hours} hour${hours > 1 ? 's' : ''} ago`;
        const days = Math.floor(hours / 24);
        return `${days} day${days > 1 ? 's' : ''} ago`;
    },

    // Update last updated display
    updateLastUpdated(tabId, timestamp, isCache = false) {
        const element = document.getElementById(`${tabId}LastUpdated`);
        if (element) {
            const timeAgo = this.formatTimeAgo(timestamp);
            const cacheIndicator = isCache ? '<span class="data-status">Cached</span>' : '';
            element.innerHTML = `Last updated: ${timeAgo} ${cacheIndicator}`;
        }
    },

    // Create stock card HTML
    createStockCard(stock) {
        const changeClass = stock.change >= 0 ? 'positive' : 'negative';
        const arrow = stock.change >= 0 ? '▲' : '▼';

        return `
            <div class="stock-card" onclick="StockDetail.show('${stock.symbol}')">
                <div class="stock-symbol">${stock.symbol}</div>
                <div class="stock-name">${stock.name || stock.symbol}</div>
                <div class="stock-price">$${stock.price.toFixed(2)}</div>
                <div class="stock-change ${changeClass}">
                    ${arrow} ${Math.abs(stock.change).toFixed(2)} (${stock.change_percent})
                </div>
                <div style="margin-top: 10px; font-size: 12px; color: #aaa;">
                    Volume: ${(stock.volume / 1000000).toFixed(2)}M
                </div>
            </div>
        `;
    },

    // Display stocks grid
    displayStocks(stocks, containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;

        if (!stocks || stocks.length === 0) {
            container.innerHTML = '<div class="error">No stocks to display</div>';
            return;
        }

        container.innerHTML = stocks.map(stock => this.createStockCard(stock)).join('');
    },

    // Update sentiment meter
    updateSentimentMeter(fearGreedIndex) {
        const indicator = document.getElementById('sentimentIndicator');
        if (!indicator) return;

        const position = (fearGreedIndex / 100) * 80 + 10;
        indicator.style.left = `${position}%`;

        if (fearGreedIndex < 20) {
            indicator.style.background = '#f00';
            indicator.style.boxShadow = '0 0 10px #f00';
        } else if (fearGreedIndex < 40) {
            indicator.style.background = '#ff0';
            indicator.style.boxShadow = '0 0 10px #ff0';
        } else if (fearGreedIndex < 60) {
            indicator.style.background = '#fff';
            indicator.style.boxShadow = '0 0 10px #fff';
        } else if (fearGreedIndex < 80) {
            indicator.style.background = '#0f0';
            indicator.style.boxShadow = '0 0 10px #0f0';
        } else {
            indicator.style.background = '#0f0';
            indicator.style.boxShadow = '0 0 20px #0f0';
        }
    },

    // Display market indices
    displayMarketIndices(indices) {
        let html = '';

        for (const [name, data] of Object.entries(indices)) {
            if (name === 'fear_greed_index') continue;

            const changeClass = data.change >= 0 ? 'positive' : 'negative';
            const arrow = data.change >= 0 ? '▲' : '▼';

            html += `
                <div class="indicator-card">
                    <div class="indicator-label">${name}</div>
                    <div class="indicator-value">${data.value.toFixed(2)}</div>
                    <div class="${changeClass}">
                        ${arrow} ${Math.abs(data.change).toFixed(2)} (${data.change_percent.toFixed(2)}%)
                    </div>
                </div>
            `;
        }

        const container = document.getElementById('marketIndices');
        if (container) {
            container.innerHTML = html;
        }
    }
};