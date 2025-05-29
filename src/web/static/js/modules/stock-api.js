// stock-api.js - API Communication Module
const StockAPI = {
    config: {
        baseURL: '/api',
        timeout: 30000
    },

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

    async getMarketOverview() {
        return this.call('/market-overview');
    },

    async getPredictions() {
        return this.call('/predictions');
    },

    async getDebugInfo() {
        return this.call('/debug');
    },

    async testStock(symbol) {
        return this.call(`/test-stock/${symbol}`);
    }
};