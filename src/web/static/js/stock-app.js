// stock-app.js - Main Application Controller
const StockApp = {
    updateInterval: null,
    cachedStocks: {},
    
    async init() {
        console.log('Initializing Stock Analyzer...');
        
        StockPortfolio.init();
        
        await this.loadCachedStocks();
        this.loadMarketOverview();
        this.loadStocks('all');
        this.loadPredictions();
        this.loadPortfolio();
        
        this.updateInterval = setInterval(() => {
            const currentTab = StockUI.currentTab;
            if (currentTab === 'overview') {
                this.loadMarketOverview();
            } else if (currentTab === 'predictions') {
                this.loadPredictions();
            } else if (currentTab === 'portfolio') {
                this.loadPortfolio();
            } else {
                this.loadStocks(currentTab);
            }
        }, 30000);
        
        this.setupEventHandlers();
        setTimeout(() => this.checkAPIStatus(), 1000);
    },
    
    setupEventHandlers() {
        document.getElementById('stockSearch')?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.searchStocks();
            }
        });
        
        window.onclick = (event) => {
            const modal = document.getElementById('stockModal');
            if (event.target === modal) {
                StockDetail.close();
            }
        };
    },
    
    switchTab(tab) {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        
        const tabElements = document.querySelectorAll('.tab');
        tabElements.forEach(t => {
            const onclick = t.getAttribute('onclick');
            if (onclick && onclick.includes(`'${tab}'`)) {
                t.classList.add('active');
            }
        });
        
        document.querySelectorAll('.content-section').forEach(s => s.classList.remove('active'));
        const contentElement = document.getElementById(tab);
        if (contentElement) {
            contentElement.classList.add('active');
        }
        
        StockUI.currentTab = tab;
        
        switch(tab) {
            case 'overview':
                this.loadMarketOverview();
                break;
            case 'predictions':
                this.loadPredictions();
                break;
            case 'portfolio':
                this.loadPortfolio();
                break;
            default:
                this.loadStocks(tab);
        }
    },
    
    async loadCachedStocks() {
        try {
            console.log('Loading cached stocks...');
            const response = await StockAPI.getCachedStocks();
            
            if (response.data && response.data.length > 0) {
                this.cachedStocks = {};
                response.data.forEach(stock => {
                    this.cachedStocks[stock.symbol] = stock;
                });
                
                if (StockUI.currentTab === 'all') {
                    StockUI.displayStocks(response.data, 'allGrid');
                    StockUI.updateLastUpdated('all', response.last_updated || new Date().toISOString(), true);
                }
                
                console.log(`Loaded ${response.data.length} cached stocks`);
            }
        } catch (error) {
            console.error('Error loading cached stocks:', error);
        }
    },
    
    async loadStocks(category) {
        try {
            console.log(`Loading stocks for category: ${category}`);
            
            if (Object.keys(this.cachedStocks).length > 0) {
                const cachedArray = Object.values(this.cachedStocks);
                StockUI.displayStocks(cachedArray, 'allGrid');
                StockUI.updateLastUpdated('all', new Date().toISOString(), true);
            }
            
            const response = await StockAPI.getStocks(category);
            
            if (response.data && response.data.length > 0) {
                StockUI.displayStocks(response.data, 'allGrid');
                StockUI.updateLastUpdated(category, response.last_updated || new Date().toISOString(), response.from_cache);
                
                response.data.forEach(stock => {
                    this.cachedStocks[stock.symbol] = stock;
                });
                
                console.log(`Loaded ${response.data.length} stocks`);
            }
        } catch (error) {
            console.error('Error loading stocks:', error);
            
            if (Object.keys(this.cachedStocks).length > 0) {
                const cachedArray = Object.values(this.cachedStocks);
                StockUI.displayStocks(cachedArray, 'allGrid');
            } else {
                StockUI.showError('allGrid', 'Failed to load stock data', error.message);
            }
        }
    },
    
    async loadMarketOverview() {
        try {
            console.log('Loading market overview...');
            const response = await StockAPI.getMarketOverview();
            
            if (response.data) {
                if (response.data.indices) {
                    StockUI.displayMarketIndices(response.data.indices);
                    if (response.data.indices.fear_greed_index !== undefined) {
                        StockUI.updateSentimentMeter(response.data.indices.fear_greed_index);
                    }
                }
                StockUI.updateLastUpdated('overview', response.timestamp || new Date().toISOString());
            }
        } catch (error) {
            console.error('Error loading market overview:', error);
            StockUI.showError('marketIndices', 'Failed to load market data', error.message);
        }
    },
    
    async loadPredictions() {
        try {
            console.log('Loading AI predictions...');
            const response = await StockAPI.getPredictions();
            
            if (response.data?.opportunities?.length > 0) {
                this.displayPredictions(response.data.opportunities);
                StockUI.updateLastUpdated('predictions', response.data.generated_at || new Date().toISOString());
            } else {
                document.getElementById('predictionsGrid').innerHTML = 
                    '<div class="analysis-section">No strong trading signals detected at this time. The AI is continuously analyzing market patterns.</div>';
                StockUI.updateLastUpdated('predictions', new Date().toISOString());
            }
        } catch (error) {
            console.error('Error loading predictions:', error);
            StockUI.showError('predictionsGrid', 'Failed to load AI predictions', error.message);
        }
    },
    
    displayPredictions(opportunities) {
        let html = '';
        
        opportunities.forEach(opp => {
            const scoreColor = opp.score >= 7 ? '#0f0' : opp.score >= 5 ? '#ff0' : '#f90';
            
            html += `
                <div class="stock-card" onclick="StockDetail.show('${opp.ticker}')">
                    <div class="stock-symbol">${opp.ticker}</div>
                    <div class="stock-name">${opp.name || opp.ticker}</div>
                    <div class="stock-price">$${opp.price.toFixed(2)}</div>
                    <div style="margin: 10px 0; padding: 10px; background: rgba(0,255,0,0.1); border-radius: 5px;">
                        <div style="color: ${scoreColor}; font-weight: bold; margin-bottom: 5px;">
                            Score: ${opp.score}/10
                        </div>
                        <div style="font-size: 12px; color: #aaa;">
                            ${opp.reasons[0]}
                        </div>
                    </div>
                    <div style="font-size: 12px; color: #0f0;">
                        ${opp.strategy?.recommendation || 'BUY'}
                    </div>
                </div>
            `;
        });
        
        document.getElementById('predictionsGrid').innerHTML = html;
    },
    
    loadPortfolio() {
        StockPortfolio.display(this.cachedStocks);
        StockUI.updateLastUpdated('portfolio', new Date().toISOString());
    },
    
    async searchStocks() {
        const query = document.getElementById('stockSearch').value.trim();
        if (!query) return;
        
        try {
            const response = await StockAPI.searchStocks(query);
            const results = response.data.results;
            
            if (results.length > 0) {
                StockUI.displayStocks(results, 'allGrid');
                this.switchTab('all');
            } else {
                alert('No stocks found matching your search');
            }
        } catch (error) {
            console.error('Error searching stocks:', error);
            alert('Error searching stocks');
        }
    },
    
    clearSearch() {
        document.getElementById('stockSearch').value = '';
        this.loadStocks('all');
    },
    
    async checkAPIStatus() {
        try {
            const response = await StockAPI.getDebugInfo();
            console.log('API Debug Info:', response.data);
        } catch (error) {
            console.error('API Debug failed:', error);
        }
    }
};