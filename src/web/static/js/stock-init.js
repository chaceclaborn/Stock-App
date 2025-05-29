// stock-init.js - Initialize and expose global functions
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, initializing app...');
    
    // Initialize the app
    StockApp.init();
    
    // Expose functions globally for onclick handlers
    window.switchTab = function(tab) {
        StockApp.switchTab(tab);
    };
    
    window.searchStocks = function() {
        StockApp.searchStocks();
    };
    
    window.clearSearch = function() {
        StockApp.clearSearch();
    };
    
    window.showStockDetail = function(symbol) {
        StockDetail.show(symbol);
    };
    
    window.closeModal = function() {
        StockDetail.close();
    };
    
    window.addToPortfolio = function() {
        StockPortfolio.addFromForm();
    };
    
    window.removeFromPortfolio = function(symbol) {
        StockPortfolio.removeAndRefresh(symbol);
    };
    
    // Make modules available globally for debugging
    window.StockApp = StockApp;
    window.StockDetail = StockDetail;
    window.StockPortfolio = StockPortfolio;
    window.StockUI = StockUI;
    window.StockAPI = StockAPI;
});