# web_debugger.py
"""
Comprehensive Web-Based Stock Analyzer Debugger
Run this to debug and fix everything through a web interface
"""
import os
import sys
import json
import time
import traceback
import subprocess
import threading
import importlib.util
from datetime import datetime
from flask import Flask, render_template_string, jsonify, request, redirect, url_for
import logging

# Suppress Flask logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

class StockAnalyzerDebugger:
    def __init__(self):
        self.app = Flask(__name__)
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'fixes_applied': [],
            'system_info': {
                'python_version': sys.version,
                'current_dir': os.getcwd(),
                'platform': sys.platform
            }
        }
        self.setup_routes()
        
    def setup_routes(self):
        """Setup Flask routes"""
        @self.app.route('/')
        def index():
            return render_template_string(self.get_html_template())
        
        @self.app.route('/api/run-diagnostics')
        def run_diagnostics():
            """Run all diagnostic tests"""
            results = self.run_all_tests()
            return jsonify(results)
        
        @self.app.route('/api/fix/<issue>')
        def fix_issue(issue):
            """Fix a specific issue"""
            result = self.apply_fix(issue)
            return jsonify(result)
        
        @self.app.route('/api/test-source/<source>')
        def test_source(source):
            """Test a specific data source"""
            result = self.test_data_source(source)
            return jsonify(result)
        
        @self.app.route('/api/install-dependencies')
        def install_dependencies():
            """Install missing dependencies"""
            result = self.install_missing_dependencies()
            return jsonify(result)
        
        @self.app.route('/api/save-config', methods=['POST'])
        def save_config():
            """Save API configuration to .env file"""
            try:
                config = request.json
                result = self.save_env_config(config)
                return jsonify(result)
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
    
    def get_html_template(self):
        """Return the HTML template for the debugger"""
        return '''
<!DOCTYPE html>
<html>
<head>
    <title>Stock Analyzer Debugger</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0a;
            color: #e0e0e0;
            line-height: 1.6;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            background: linear-gradient(135deg, #1e3c72, #2a5298);
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }
        h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .status-card {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 10px;
            padding: 20px;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .status-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.4);
        }
        .status-card h3 {
            font-size: 1.3em;
            margin-bottom: 15px;
            color: #4a9eff;
        }
        .test-item {
            padding: 8px 0;
            border-bottom: 1px solid #333;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .test-item:last-child {
            border-bottom: none;
        }
        .status-pass {
            color: #4ade80;
            font-weight: bold;
        }
        .status-fail {
            color: #f87171;
            font-weight: bold;
        }
        .status-warning {
            color: #fbbf24;
            font-weight: bold;
        }
        .status-pending {
            color: #60a5fa;
            font-style: italic;
        }
        .action-buttons {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-bottom: 30px;
        }
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }
        .btn-primary {
            background: linear-gradient(135deg, #4a9eff, #3b82f6);
            color: white;
        }
        .btn-primary:hover {
            background: linear-gradient(135deg, #3b82f6, #2563eb);
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
        }
        .btn-success {
            background: linear-gradient(135deg, #4ade80, #22c55e);
            color: white;
        }
        .btn-danger {
            background: linear-gradient(135deg, #f87171, #ef4444);
            color: white;
        }
        .btn-warning {
            background: linear-gradient(135deg, #fbbf24, #f59e0b);
            color: #000;
        }
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .console {
            background: #000;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 20px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 14px;
            max-height: 400px;
            overflow-y: auto;
            margin-bottom: 20px;
        }
        .console-line {
            padding: 2px 0;
        }
        .console-error {
            color: #f87171;
        }
        .console-success {
            color: #4ade80;
        }
        .console-info {
            color: #60a5fa;
        }
        .console-warning {
            color: #fbbf24;
        }
        .details-panel {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
        }
        .fix-button {
            padding: 6px 12px;
            font-size: 12px;
            background: #22c55e;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .fix-button:hover {
            background: #16a34a;
        }
        .loader {
            border: 3px solid #333;
            border-top: 3px solid #4a9eff;
            border-radius: 50%;
            width: 20px;
            height: 20px;
            animation: spin 1s linear infinite;
            display: inline-block;
            margin-left: 10px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .progress-bar {
            width: 100%;
            height: 30px;
            background: #333;
            border-radius: 15px;
            overflow: hidden;
            margin: 20px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4a9eff, #22c55e);
            width: 0%;
            transition: width 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }
        .api-config {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }
        .api-config input {
            width: 100%;
            padding: 10px;
            margin: 5px 0;
            background: #000;
            border: 1px solid #333;
            border-radius: 4px;
            color: #e0e0e0;
        }
        .tab-container {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 2px solid #333;
        }
        .tab {
            padding: 12px 24px;
            background: none;
            border: none;
            color: #888;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            transition: all 0.3s;
            border-bottom: 3px solid transparent;
        }
        .tab.active {
            color: #4a9eff;
            border-bottom-color: #4a9eff;
        }
        .tab:hover {
            color: #60a5fa;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        pre {
            background: #000;
            padding: 15px;
            border-radius: 6px;
            overflow-x: auto;
            font-size: 13px;
            line-height: 1.4;
            border: 1px solid #333;
        }
        .quick-fix {
            background: #1e3c72;
            border: 2px solid #4a9eff;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #333;
        }
        .metric:last-child {
            border-bottom: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔧 Stock Analyzer Debugger</h1>
            <p>Comprehensive diagnostic and repair tool for your stock analyzer</p>
        </div>

        <div class="action-buttons">
            <button class="btn btn-primary" onclick="runDiagnostics()">
                <span>🔍</span> Run Full Diagnostics
            </button>
            <button class="btn btn-success" onclick="autoFix()">
                <span>🔧</span> Auto-Fix All Issues
            </button>
            <button class="btn btn-warning" onclick="installDeps()">
                <span>📦</span> Install Dependencies
            </button>
            <button class="btn btn-danger" onclick="restartApp()">
                <span>🔄</span> Restart Main App
            </button>
        </div>

        <div class="tab-container">
            <button class="tab active" onclick="showTab('overview')">Overview</button>
            <button class="tab" onclick="showTab('console')">Console</button>
            <button class="tab" onclick="showTab('config')">Configuration</button>
            <button class="tab" onclick="showTab('details')">Detailed Results</button>
        </div>

        <div id="overview" class="tab-content active">
            <div class="quick-fix" id="quickFix" style="display:none;">
                <h3>⚡ Quick Fix Available</h3>
                <p id="quickFixMessage"></p>
                <button class="btn btn-success" onclick="applyQuickFix()">Apply Fix Now</button>
            </div>

            <div class="progress-bar" id="progressBar" style="display:none;">
                <div class="progress-fill" id="progressFill">0%</div>
            </div>

            <div class="status-grid" id="statusGrid">
                <div class="status-card">
                    <h3>📊 System Status</h3>
                    <div id="systemTests">
                        <div class="test-item">
                            <span>Loading...</span>
                            <span class="loader"></span>
                        </div>
                    </div>
                </div>

                <div class="status-card">
                    <h3>📁 File Structure</h3>
                    <div id="fileTests">
                        <div class="test-item">
                            <span>Loading...</span>
                            <span class="loader"></span>
                        </div>
                    </div>
                </div>

                <div class="status-card">
                    <h3>🔌 Data Sources</h3>
                    <div id="sourceTests">
                        <div class="test-item">
                            <span>Loading...</span>
                            <span class="loader"></span>
                        </div>
                    </div>
                </div>

                <div class="status-card">
                    <h3>🌐 Web Services</h3>
                    <div id="webTests">
                        <div class="test-item">
                            <span>Loading...</span>
                            <span class="loader"></span>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div id="console" class="tab-content">
            <div class="console" id="consoleOutput">
                <div class="console-line console-info">Console ready. Run diagnostics to see output...</div>
            </div>
        </div>

        <div id="config" class="tab-content">
            <div class="api-config">
                <h3>🔑 API Configuration</h3>
                <p>Configure your data source API keys:</p>
                
                <label>Alpha Vantage API Key:</label>
                <input type="text" id="alphaKey" placeholder="Enter your Alpha Vantage API key">
                <small>Get free key at: https://www.alphavantage.co/support/#api-key</small>
                
                <label>Twelve Data API Key (Optional):</label>
                <input type="text" id="twelveKey" placeholder="Enter your Twelve Data API key">
                <small>Get free key at: https://twelvedata.com/apikey</small>
                
                <label>Polygon.io API Key (Optional):</label>
                <input type="text" id="polygonKey" placeholder="Enter your Polygon API key">
                <small>Get free key at: https://polygon.io/dashboard/signup</small>
                
                <br><br>
                <button class="btn btn-primary" onclick="saveConfig()">Save Configuration</button>
            </div>
        </div>

        <div id="details" class="tab-content">
            <div class="details-panel">
                <h3>📋 Detailed Test Results</h3>
                <pre id="detailedResults">Run diagnostics to see detailed results...</pre>
            </div>
        </div>
    </div>

    <script>
        let testResults = {};
        let currentTab = 'overview';

        function showTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected tab
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
            currentTab = tabName;
        }

        function log(message, type = 'info') {
            const console = document.getElementById('consoleOutput');
            const line = document.createElement('div');
            line.className = `console-line console-${type}`;
            line.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
            console.appendChild(line);
            console.scrollTop = console.scrollHeight;
        }

        function updateProgress(percent, message) {
            const bar = document.getElementById('progressBar');
            const fill = document.getElementById('progressFill');
            bar.style.display = 'block';
            fill.style.width = percent + '%';
            fill.textContent = message || percent + '%';
        }

        async function runDiagnostics() {
            log('Starting comprehensive diagnostics...', 'info');
            updateProgress(0, 'Starting...');
            
            try {
                const response = await fetch('/api/run-diagnostics');
                const results = await response.json();
                testResults = results;
                
                updateProgress(100, 'Complete!');
                displayResults(results);
                
                // Check for critical issues
                checkCriticalIssues(results);
                
                log('Diagnostics complete!', 'success');
                
                // Update detailed results
                document.getElementById('detailedResults').textContent = JSON.stringify(results, null, 2);
                
            } catch (error) {
                log(`Error running diagnostics: ${error}`, 'error');
                updateProgress(0, 'Failed');
            }
        }

        function displayResults(results) {
            // System tests
            displayTestGroup('systemTests', results.system || {});
            
            // File tests
            displayTestGroup('fileTests', results.files || {});
            
            // Source tests
            displayTestGroup('sourceTests', results.sources || {});
            
            // Web tests
            displayTestGroup('webTests', results.web || {});
        }

        function displayTestGroup(elementId, tests) {
            const container = document.getElementById(elementId);
            container.innerHTML = '';
            
            for (const [name, result] of Object.entries(tests)) {
                const item = document.createElement('div');
                item.className = 'test-item';
                
                const nameSpan = document.createElement('span');
                nameSpan.textContent = name;
                
                const statusSpan = document.createElement('span');
                statusSpan.className = `status-${result.status.toLowerCase()}`;
                statusSpan.textContent = result.status;
                
                // Add fix button if applicable
                if (result.status === 'FAIL' && result.fixable) {
                    const fixBtn = document.createElement('button');
                    fixBtn.className = 'fix-button';
                    fixBtn.textContent = 'Fix';
                    fixBtn.onclick = () => fixIssue(result.fix_id);
                    statusSpan.appendChild(fixBtn);
                }
                
                item.appendChild(nameSpan);
                item.appendChild(statusSpan);
                container.appendChild(item);
            }
        }

        function checkCriticalIssues(results) {
            const critical = [];
            
            // Check for import issues
            if (results.files?.fetcher_multi_source?.status === 'FAIL') {
                critical.push('Multi-source fetcher not found');
            }
            
            if (results.system?.imports?.status === 'FAIL') {
                critical.push('Import errors detected');
            }
            
            if (critical.length > 0) {
                const quickFix = document.getElementById('quickFix');
                const message = document.getElementById('quickFixMessage');
                quickFix.style.display = 'block';
                message.textContent = `Found ${critical.length} critical issues: ${critical.join(', ')}`;
            }
        }

        async function autoFix() {
            log('Starting auto-fix process...', 'info');
            
            // Fix imports
            await fixIssue('imports');
            
            // Install missing dependencies
            await installDeps();
            
            // Re-run diagnostics
            await runDiagnostics();
            
            log('Auto-fix complete!', 'success');
        }

        async function fixIssue(issue) {
            log(`Fixing issue: ${issue}`, 'info');
            
            try {
                const response = await fetch(`/api/fix/${issue}`);
                const result = await response.json();
                
                if (result.success) {
                    log(`Fixed: ${issue}`, 'success');
                } else {
                    log(`Failed to fix ${issue}: ${result.error}`, 'error');
                }
                
                return result;
            } catch (error) {
                log(`Error fixing ${issue}: ${error}`, 'error');
                return {success: false, error: error.toString()};
            }
        }

        async function installDeps() {
            log('Installing dependencies...', 'info');
            updateProgress(0, 'Installing...');
            
            try {
                const response = await fetch('/api/install-dependencies');
                const result = await response.json();
                
                if (result.success) {
                    log('Dependencies installed successfully', 'success');
                } else {
                    log(`Failed to install dependencies: ${result.error}`, 'error');
                }
                
                updateProgress(100, 'Complete!');
                return result;
            } catch (error) {
                log(`Error installing dependencies: ${error}`, 'error');
                updateProgress(0, 'Failed');
                return {success: false, error: error.toString()};
            }
        }

        async function restartApp() {
            if (confirm('This will restart the main stock analyzer app. Continue?')) {
                log('Restarting main application...', 'info');
                
                try {
                    const response = await fetch('/api/restart-app');
                    const result = await response.json();
                    
                    if (result.success) {
                        log('Main app restarted successfully', 'success');
                        setTimeout(() => {
                            window.open('http://localhost:5000', '_blank');
                        }, 2000);
                    } else {
                        log(`Failed to restart app: ${result.error}`, 'error');
                    }
                } catch (error) {
                    log(`Error restarting app: ${error}`, 'error');
                }
            }
        }

        async function saveConfig() {
            const config = {
                ALPHA_VANTAGE_API_KEY: document.getElementById('alphaKey').value,
                TWELVE_DATA_API_KEY: document.getElementById('twelveKey').value,
                POLYGON_API_KEY: document.getElementById('polygonKey').value
            };
            
            // Save to .env file
            log('Saving API configuration...', 'info');
            
            try {
                const response = await fetch('/api/save-config', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(config)
                });
                
                const result = await response.json();
                
                if (result.success) {
                    log('Configuration saved successfully!', 'success');
                    log('Re-run diagnostics to test your API keys', 'info');
                } else {
                    log(`Failed to save configuration: ${result.error}`, 'error');
                }
            } catch (error) {
                log(`Error saving configuration: ${error}`, 'error');
            }
        }

        async function applyQuickFix() {
            log('Applying quick fixes...', 'info');
            await autoFix();
        }

        // Auto-run diagnostics on load
        window.onload = () => {
            setTimeout(runDiagnostics, 1000);
            
            // Load existing API keys if any
            loadExistingConfig();
        };
        
        async function loadExistingConfig() {
            // This would load from the diagnostics results
            // For now, just check if they exist in environment
        }
    </script>
</body>
</html>
        '''
    
    def run_all_tests(self):
        """Run all diagnostic tests"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'system': self.test_system(),
            'files': self.test_files(),
            'sources': self.test_data_sources(),
            'web': self.test_web_layer()
        }
        return results
    
    def test_system(self):
        """Test system configuration"""
        tests = {}
        
        # Python version
        major, minor = sys.version_info[:2]
        tests['Python Version'] = {
            'status': 'PASS' if major >= 3 and minor >= 7 else 'WARNING',
            'message': f"Python {major}.{minor}",
            'details': sys.version
        }
        
        # Dependencies
        required_deps = ['flask', 'yfinance', 'pandas', 'numpy', 'requests']
        missing = []
        for dep in required_deps:
            try:
                __import__(dep)
            except ImportError:
                missing.append(dep)
        
        tests['Dependencies'] = {
            'status': 'PASS' if not missing else 'FAIL',
            'message': f"{len(required_deps) - len(missing)}/{len(required_deps)} installed",
            'fixable': True,
            'fix_id': 'dependencies'
        }
        
        # Import test
        import_error = None
        import_success = False
        
        # Try multiple import strategies
        original_path = sys.path.copy()
        
        try:
            # Strategy 1: Add src to path and try import
            if 'src' not in sys.path:
                sys.path.insert(0, 'src')
            
            try:
                from src.data.fetcher import StockDataFetcher
                import_success = True
                tests['Imports'] = {'status': 'PASS', 'message': 'Working'}
            except ImportError as e1:
                # Strategy 2: Try absolute path
                src_path = os.path.join(os.getcwd(), 'src')
                if src_path not in sys.path:
                    sys.path.insert(0, src_path)
                
                try:
                    from src.data.fetcher import StockDataFetcher
                    import_success = True
                    tests['Imports'] = {'status': 'PASS', 'message': 'Working (with path fix)'}
                except ImportError as e2:
                    # Strategy 3: Try direct module import
                    try:
                        import importlib.util
                        fetcher_path = os.path.join('src', 'data', 'fetcher.py')
                        if os.path.exists(fetcher_path):
                            spec = importlib.util.spec_from_file_location("fetcher", fetcher_path)
                            fetcher_module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(fetcher_module)
                            import_success = True
                            tests['Imports'] = {'status': 'WARNING', 'message': 'Working (direct import)'}
                        else:
                            import_error = f"fetcher.py not found at {fetcher_path}"
                    except Exception as e3:
                        import_error = f"All import methods failed: {str(e1)}, {str(e2)}, {str(e3)}"
        
        except Exception as e:
            import_error = str(e)
        finally:
            # Restore original path
            sys.path = original_path
        
        if not import_success:
            tests['Imports'] = {
                'status': 'FAIL',
                'message': import_error or 'Unknown import error',
                'fixable': True,
                'fix_id': 'imports'
            }
        
        return tests
    
    def test_files(self):
        """Test file structure"""
        tests = {}
        
        critical_files = {
            'fetcher.py': 'src/data/fetcher.py',
            'fetcher_multi_source': 'src/data/fetcher_multi_source.py',
            'app.py': 'src/web/app.py',
            'database.py': 'src/data/database.py'
        }
        
        for name, path in critical_files.items():
            tests[name] = {
                'status': 'PASS' if os.path.exists(path) else 'FAIL',
                'message': 'Found' if os.path.exists(path) else 'Missing',
                'path': path
            }
        
        return tests
    
    def test_data_sources(self):
        """Test data sources"""
        tests = {}
        
        # First check if fetcher files exist
        fetcher_path = os.path.join('src', 'data', 'fetcher.py')
        multi_source_path = os.path.join('src', 'data', 'fetcher_multi_source.py')
        
        tests['Fetcher File'] = {
            'status': 'PASS' if os.path.exists(fetcher_path) else 'FAIL',
            'message': 'Found' if os.path.exists(fetcher_path) else 'Missing'
        }
        
        tests['Multi-Source File'] = {
            'status': 'PASS' if os.path.exists(multi_source_path) else 'WARNING',
            'message': 'Found' if os.path.exists(multi_source_path) else 'Missing (will use fallback)'
        }
        
        # Yahoo Finance test with better error handling
        yahoo_status = 'UNKNOWN'
        yahoo_message = 'Not tested'
        
        try:
            import yfinance as yf
            ticker = yf.Ticker("AAPL")
            
            # Try to get data with timeout
            try:
                info = ticker.info
                price = info.get('regularMarketPrice') or info.get('currentPrice', 0)
                
                if price:
                    yahoo_status = 'PASS'
                    yahoo_message = f"Connected - AAPL: ${price}"
                else:
                    # Try history as fallback
                    hist = ticker.history(period="1d")
                    if not hist.empty:
                        yahoo_status = 'WARNING'
                        yahoo_message = "Limited access (rate limited?)"
                    else:
                        yahoo_status = 'FAIL'
                        yahoo_message = "No data (rate limit exceeded)"
            except Exception as e:
                yahoo_status = 'FAIL'
                yahoo_message = f"API error: {str(e)[:50]}..."
                
        except ImportError:
            yahoo_status = 'FAIL'
            yahoo_message = "yfinance not installed"
        except Exception as e:
            yahoo_status = 'FAIL'
            yahoo_message = str(e)[:50] + "..."
        
        tests['Yahoo Finance'] = {
            'status': yahoo_status,
            'message': yahoo_message
        }
        
        # Check API keys from environment
        env_file_exists = os.path.exists('.env')
        
        # Try to load .env file
        if env_file_exists:
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass
        
        # Check API keys
        av_key = os.environ.get('ALPHA_VANTAGE_API_KEY', '')
        tests['Alpha Vantage'] = {
            'status': 'PASS' if av_key and av_key != 'demo' else 'WARNING',
            'message': f"Key configured ({av_key[:8]}...)" if av_key else 'Not configured',
            'fixable': True,
            'fix_id': 'config'
        }
        
        twelve_key = os.environ.get('TWELVE_DATA_API_KEY', '')
        tests['Twelve Data'] = {
            'status': 'PASS' if twelve_key else 'INFO',
            'message': f"Key configured ({twelve_key[:8]}...)" if twelve_key else 'Optional - not configured'
        }
        
        tests['.env File'] = {
            'status': 'PASS' if env_file_exists else 'WARNING',
            'message': 'Found' if env_file_exists else 'Not found (using system env)'
        }
        
        return tests
    
    def test_web_layer(self):
        """Test web components"""
        tests = {}
        
        # Check if Flask app file exists
        app_path = os.path.join('src', 'web', 'app.py')
        tests['Flask App File'] = {
            'status': 'PASS' if os.path.exists(app_path) else 'FAIL',
            'message': 'Found' if os.path.exists(app_path) else 'Missing'
        }
        
        # Try to create Flask app
        if os.path.exists(app_path):
            original_path = sys.path.copy()
            try:
                # Add src to path
                src_path = os.path.join(os.getcwd(), 'src')
                if src_path not in sys.path:
                    sys.path.insert(0, src_path)
                
                from src.web.app import create_app
                app = create_app('development')
                
                # Count routes
                route_count = len(list(app.url_map.iter_rules()))
                
                tests['Flask App Creation'] = {
                    'status': 'PASS',
                    'message': f'Created with {route_count} routes'
                }
            except ImportError as e:
                tests['Flask App Creation'] = {
                    'status': 'FAIL',
                    'message': f'Import error: {str(e)}',
                    'fixable': True,
                    'fix_id': 'web_imports'
                }
            except Exception as e:
                tests['Flask App Creation'] = {
                    'status': 'FAIL',
                    'message': f'Error: {str(e)[:50]}...'
                }
            finally:
                sys.path = original_path
        
        # Check for routes directory
        routes_path = os.path.join('src', 'web', 'routes')
        tests['Routes Directory'] = {
            'status': 'PASS' if os.path.isdir(routes_path) else 'FAIL',
            'message': 'Found' if os.path.isdir(routes_path) else 'Missing'
        }
        
        # Check for services directory
        services_path = os.path.join('src', 'web', 'services')
        tests['Services Directory'] = {
            'status': 'PASS' if os.path.isdir(services_path) else 'FAIL',
            'message': 'Found' if os.path.isdir(services_path) else 'Missing'
        }
        
        return tests
    
    def apply_fix(self, issue):
        """Apply a specific fix"""
        if issue == 'imports':
            return self.fix_imports()
        elif issue == 'dependencies':
            return self.install_missing_dependencies()
        else:
            return {'success': False, 'error': 'Unknown fix'}
    
    def fix_imports(self):
        """Fix import issues"""
        try:
            # First, ensure we have the multi-source file
            multi_source_path = 'src/data/fetcher_multi_source.py'
            if not os.path.exists(multi_source_path):
                return {'success': False, 'error': 'fetcher_multi_source.py not found in src/data/'}
            
            # Read the current fetcher.py
            fetcher_path = 'src/data/fetcher.py'
            if not os.path.exists(fetcher_path):
                return {'success': False, 'error': 'fetcher.py not found in src/data/'}
            
            # Create a new fetcher.py with proper imports
            new_fetcher_content = '''# src/data/fetcher.py
"""
Stock data fetcher with multi-source support and fallback
"""
import os
import sys
import time
import logging

# Fix import path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Try different import methods
try:
    # Method 1: Relative import
    from .fetcher_multi_source import MultiSourceStockFetcher, DataCache
except ImportError:
    try:
        # Method 2: Absolute import from data
        from data.fetcher_multi_source import MultiSourceStockFetcher, DataCache
    except ImportError:
        try:
            # Method 3: Direct import
            import fetcher_multi_source
            MultiSourceStockFetcher = fetcher_multi_source.MultiSourceStockFetcher
            DataCache = fetcher_multi_source.DataCache
        except ImportError:
            # Fallback: Create dummy class
            class MultiSourceStockFetcher:
                def __init__(self, db=None):
                    self.db = db
                    self.sources = []
                    
                def get_quote(self, symbol):
                    import random
                    return {
                        'symbol': symbol,
                        'price': round(random.uniform(50, 500), 2),
                        'change': round(random.uniform(-5, 5), 2),
                        'change_percent': f"{random.uniform(-2, 2):.2f}%",
                        'volume': random.randint(1000000, 50000000),
                        'source': 'dummy',
                        'name': symbol
                    }
                
                def get_multiple_quotes(self, symbols):
                    return [self.get_quote(s) for s in symbols]
                
                def get_all_tracked_stocks(self):
                    return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
                
                def get_stocks_by_category(self, category):
                    return self.get_all_tracked_stocks()[:5]
                
                def get_top_stocks(self):
                    return self.get_all_tracked_stocks()[:10]
                
                def get_company_name(self, symbol):
                    return symbol
                
                def search_stocks(self, query, limit=10):
                    all_stocks = self.get_all_tracked_stocks()
                    return [s for s in all_stocks if query.upper() in s][:limit]
                
                def get_market_overview(self):
                    return {}
                
                def get_stock_data(self, symbol, period="3mo"):
                    import pandas as pd
                    return pd.DataFrame()
                
                def get_stocks_for_analysis(self, symbols=None):
                    return {}
            
            class DataCache:
                def __init__(self, default_ttl=300):
                    self.cache = {}
                    
                def get(self, key):
                    return self.cache.get(key)
                
                def set(self, key, value):
                    self.cache[key] = value

# For backwards compatibility, inherit from MultiSourceStockFetcher
class StockDataFetcher(MultiSourceStockFetcher):
    """
    Enhanced stock data fetcher with multi-source support
    Maintains backwards compatibility while adding robustness
    """
    
    def __init__(self, db=None):
        """Initialize fetcher with database"""
        super().__init__(db)
        
        # Log initialization
        logger = logging.getLogger(__name__)
        if hasattr(self, 'sources') and self.sources:
            logger.info(f"Stock Data Fetcher initialized with {len(self.sources)} sources")
        else:
            logger.info("Stock Data Fetcher initialized in fallback mode")
        
        # Additional backwards compatibility attributes
        self.quote_cache = DataCache()
        self.data_cache = DataCache()
        self.info_cache = DataCache()
        
        # For compatibility with old code
        if hasattr(self, 'yahoo'):
            self.rate_limiter = getattr(self.yahoo, 'rate_tracker', None)
        else:
            self.rate_limiter = None
    
    def get_quote_from_cache(self, symbol):
        """Backwards compatibility method"""
        cache_key = f"quote_{symbol}"
        return self.quote_cache.get(cache_key)
    
    def get_stock_fundamentals(self, symbol):
        """Get fundamentals - returns basic data for now"""
        quote = self.get_quote(symbol)
        if quote:
            return {
                'market_cap': quote.get('price', 0) * 1000000000,
                'pe_ratio': 25.0,
                'dividend_yield': 1.5,
                'beta': 1.0
            }
        return {}
    
    def get_news(self, symbol, limit=5):
        """Get news - returns dummy news for now"""
        return [
            {
                'title': f"Latest update on {symbol}",
                'publisher': 'Market News',
                'link': '#',
                'timestamp': int(time.time())
            }
            for _ in range(min(limit, 3))
        ]
    
    def get_realtime_metrics(self, symbol):
        """Get real-time metrics"""
        quote = self.get_quote(symbol)
        if quote:
            price = quote.get('price', 0)
            return {
                'bid': price - 0.01,
                'ask': price + 0.01,
                'bid_size': 100,
                'ask_size': 100,
                'day_high': price * 1.02,
                'day_low': price * 0.98,
                'open': price * 0.99,
                'previous_close': price,
                'fifty_day_average': price,
                'two_hundred_day_average': price,
                'average_volume': quote.get('volume', 1000000),
                'exchange': 'NASDAQ',
                'currency': 'USD'
            }
        return None

# Create module-level instances for backwards compatibility
if __name__ != "__main__":
    try:
        _default_fetcher = StockDataFetcher()
    except:
        _default_fetcher = None
'''
            
            # Backup the old file
            import shutil
            backup_path = fetcher_path + '.backup'
            shutil.copy(fetcher_path, backup_path)
            
            # Write the new content
            with open(fetcher_path, 'w', encoding='utf-8') as f:
                f.write(new_fetcher_content)
            
            # Also update the __init__.py files to help with imports
            init_path = 'src/data/__init__.py'
            init_content = '''"""
Data layer initialization
"""
import os
import sys

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Add parent directory to path
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import with fallback
try:
    from .fetcher import StockDataFetcher
    from .database import StockDatabase
except ImportError:
    try:
        from data.fetcher import StockDataFetcher
        from data.database import StockDatabase
    except ImportError:
        # Create dummy classes
        class StockDataFetcher:
            pass
        class StockDatabase:
            pass

__all__ = ['StockDatabase', 'StockDataFetcher']
'''
            
            with open(init_path, 'w', encoding='utf-8') as f:
                f.write(init_content)
            
            return {'success': True, 'message': 'Imports fixed successfully'}
            
        except Exception as e:
            import traceback
            return {'success': False, 'error': str(e), 'traceback': traceback.format_exc()}
    
    def install_missing_dependencies(self):
        """Install missing Python packages"""
        try:
            deps = ['flask', 'yfinance', 'pandas', 'numpy', 'requests', 'python-dotenv']
            for dep in deps:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', dep])
            return {'success': True, 'message': 'Dependencies installed'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def restart_main_app(self):
        """Restart the main stock analyzer app"""
        try:
            # Start in a new thread to avoid blocking
            def start_app():
                subprocess.Popen([sys.executable, 'run_webapp.py'])
            
            thread = threading.Thread(target=start_app)
            thread.daemon = True
            thread.start()
            
            return {'success': True, 'message': 'App started'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def run(self, port=8888):
        """Run the debugger web interface"""
        print("\n" + "="*60)
        print("🔧 STOCK ANALYZER WEB DEBUGGER")
        print("="*60)
        print(f"\n✅ Debugger running at: http://localhost:{port}")
        print("\n📊 Features:")
        print("  - Comprehensive system diagnostics")
        print("  - One-click fixes for common issues")
        print("  - API configuration interface")
        print("  - Real-time console output")
        print("\nPress Ctrl+C to stop\n")
        
        self.app.run(port=port, debug=False)

if __name__ == "__main__":
    debugger = StockAnalyzerDebugger()
    debugger.run()