// stock-api.js - API Communication Module
const StockAPI = {
    // Base configuration
    config: {
        baseURL: '/api',
        timeout: 30000
    },

    // Generic API call handler
    async call(endpoint, options = {}) {
        try {
            const response = await axios.get(`${this.config.baseURL}${endpoint}`, options);
            
            if (response.data.status === 'success') {
                return response.data;
            } else if (response.data.status === 'error') {
                throw new Error(response.data.error || 'API Error');
            } else {
                return response.data;
            }
        } catch (error) {
            console.error(`API Error on ${endpoint}:`, error);
            throw error;
        }
    },

    // Stock endpoints
    async getStocks(category = 'all', forceRefresh = false) {
        return this.call(`/stocks?category=${category}&force=${forceRefresh}`);
    },

    async getCachedStocks() {
        return this.call('/stocks/cached');
    },

    async getStockDetail(symbol) {
        return this.call(`/stock/${symbol}`);
    },

    async getStockAnalysis(symbol) {
        return this.call(`/stock/${symbol}/analysis`);
    },

    async searchStocks(query) {
        return this.call(`/search?q=${encodeURIComponent(query)}`);
    },

    // Market endpoints
    async getMarketOverview() {
        return this.call('/market-overview');
    },

    // Predictions endpoints
    async getPredictions() {
        return this.call('/predictions');
    },

    // Debug endpoints
    async getDebugInfo() {
        return this.call('/debug');
    },

    async testStock(symbol) {
        return this.call(`/test-stock/${symbol}`);
    }
};