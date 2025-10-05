#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MACD交易系统 - Railway Web监控界面
适用于Railway云部署的Web监控界面
"""

from flask import Flask, jsonify, render_template_string
import json
import os
import time
from datetime import datetime
import threading
import ccxt
import pandas as pd
import numpy as np

app = Flask(__name__)

# 全局变量存储交易数据
dashboard_data = {
    'account': {
        'totalBalance': 0.00,
        'freeBalance': 0.00,
        'dailyPnl': 0.00,
        'totalPnl': 0.00
    },
    'stats': {
        'totalTrades': 0,
        'winTrades': 0,
        'loseTrades': 0,
        'winRate': 0.0
    },
    'positions': [],
    'signals': [],
    'lastUpdate': datetime.now().isoformat()
}

# HTML模板
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MACD交易系统 - Railway监控</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 10px;
            color: #333;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        .header {
            text-align: center;
            margin-bottom: 15px;
        }

        .header h1 {
            color: white;
            font-size: 24px;
            margin-bottom: 5px;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }

        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            background: #4CAF50;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }

        .dashboard {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .top-info {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 15px;
            margin-bottom: 20px;
        }

        .info-card {
            background: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            border: 1px solid rgba(0, 0, 0, 0.1);
        }

        .info-card h3 {
            font-size: 14px;
            color: #666;
            margin-bottom: 8px;
        }

        .info-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            font-size: 12px;
        }

        .info-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .info-label {
            color: #666;
        }

        .info-value {
            font-weight: bold;
            color: #333;
        }

        .positive { color: #4CAF50; }
        .negative { color: #f44336; }

        .main-content {
            display: grid;
            grid-template-columns: 300px 1fr;
            gap: 20px;
        }

        .positions-section, .signals-section {
            background: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
            padding: 15px;
            border: 1px solid rgba(0, 0, 0, 0.1);
        }

        .positions-section h3, .signals-section h3 {
            font-size: 16px;
            margin-bottom: 10px;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 5px;
        }

        .position-item {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 8px;
            border-left: 4px solid #667eea;
            font-size: 11px;
        }

        .position-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
        }

        .position-symbol {
            font-weight: bold;
            font-size: 12px;
        }

        .position-side {
            padding: 2px 6px;
            border-radius: 4px;
            color: white;
            font-size: 10px;
        }

        .position-side.long { background: #4CAF50; }
        .position-side.short { background: #f44336; }

        .position-details {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 4px;
            font-size: 10px;
        }

        .position-detail {
            display: flex;
            justify-content: space-between;
        }

        .signals-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 8px;
        }

        .signal-item {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 6px;
            padding: 8px;
            border-left: 3px solid #ddd;
            font-size: 11px;
        }

        .signal-item.buy { border-left-color: #4CAF50; }
        .signal-item.sell { border-left-color: #f44336; }
        .signal-item.none { border-left-color: #9E9E9E; }

        .signal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 4px;
        }

        .signal-symbol {
            font-weight: bold;
            font-size: 11px;
        }

        .signal-status {
            padding: 1px 4px;
            border-radius: 3px;
            color: white;
            font-size: 9px;
        }

        .signal-status.buy { background: #4CAF50; }
        .signal-status.sell { background: #f44336; }
        .signal-status.none { background: #9E9E9E; }

        .signal-details {
            display: flex;
            justify-content: space-between;
            font-size: 10px;
            color: #666;
        }

        .last-update {
            text-align: center;
            margin-top: 15px;
            font-size: 11px;
            color: #666;
        }

        @media (max-width: 768px) {
            .top-info {
                grid-template-columns: 1fr;
                gap: 10px;
            }
            
            .main-content {
                grid-template-columns: 1fr;
                gap: 15px;
            }
            
            .signals-grid {
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><span class="status-indicator"></span>MACD交易系统 - Railway监控</h1>
        </div>

        <div class="dashboard">
            <div class="top-info">
                <div class="info-card">
                    <h3>💰 账户余额</h3>
                    <div class="info-grid">
                        <div class="info-item">
                            <span class="info-label">总余额:</span>
                            <span class="info-value" id="totalBalance">0.00</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">可用:</span>
                            <span class="info-value" id="freeBalance">0.00</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">今日:</span>
                            <span class="info-value" id="dailyPnl">0.00</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">总盈亏:</span>
                            <span class="info-value" id="totalPnl">0.00</span>
                        </div>
                    </div>
                </div>

                <div class="info-card">
                    <h3>📊 交易统计</h3>
                    <div class="info-grid">
                        <div class="info-item">
                            <span class="info-label">总交易:</span>
                            <span class="info-value" id="totalTrades">0</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">胜率:</span>
                            <span class="info-value" id="winRate">0.0%</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">盈利:</span>
                            <span class="info-value positive" id="winTrades">0</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">亏损:</span>
                            <span class="info-value negative" id="loseTrades">0</span>
                        </div>
                    </div>
                </div>

                <div class="info-card">
                    <h3>⚡ 系统状态</h3>
                    <div class="info-grid">
                        <div class="info-item">
                            <span class="info-label">持仓数:</span>
                            <span class="info-value" id="positionCount">0</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">信号数:</span>
                            <span class="info-value" id="signalCount">0</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">做多:</span>
                            <span class="info-value positive" id="buySignals">0</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">做空:</span>
                            <span class="info-value negative" id="sellSignals">0</span>
                        </div>
                    </div>
                </div>
            </div>

            <div class="main-content">
                <div class="positions-section">
                    <h3>📈 当前持仓</h3>
                    <div id="positionsList">
                        <div style="text-align: center; color: #666; font-size: 12px; padding: 20px;">
                            暂无持仓
                        </div>
                    </div>
                </div>

                <div class="signals-section">
                    <h3>🔍 交易信号</h3>
                    <div class="signals-grid" id="signalsList">
                        <!-- 信号项将通过JavaScript动态生成 -->
                    </div>
                </div>
            </div>

            <div class="last-update">
                最后更新: <span id="lastUpdate">--</span>
            </div>
        </div>
    </div>

    <script>
        function updateDashboard() {
            fetch('/api/data')
                .then(response => response.json())
                .then(data => {
                    console.log('获取到数据:', data);
                    
                    // 更新账户信息
                    document.getElementById('totalBalance').textContent = data.account.totalBalance.toFixed(2);
                    document.getElementById('freeBalance').textContent = data.account.freeBalance.toFixed(2);
                    
                    const dailyPnlElement = document.getElementById('dailyPnl');
                    dailyPnlElement.textContent = data.account.dailyPnl.toFixed(2);
                    dailyPnlElement.className = 'info-value ' + (data.account.dailyPnl >= 0 ? 'positive' : 'negative');
                    
                    const totalPnlElement = document.getElementById('totalPnl');
                    totalPnlElement.textContent = data.account.totalPnl.toFixed(2);
                    totalPnlElement.className = 'info-value ' + (data.account.totalPnl >= 0 ? 'positive' : 'negative');

                    // 更新交易统计
                    document.getElementById('totalTrades').textContent = data.stats.totalTrades;
                    document.getElementById('winRate').textContent = data.stats.winRate.toFixed(1) + '%';
                    document.getElementById('winTrades').textContent = data.stats.winTrades;
                    document.getElementById('loseTrades').textContent = data.stats.loseTrades;

                    // 更新系统状态
                    document.getElementById('positionCount').textContent = data.positions.length;
                    document.getElementById('signalCount').textContent = data.signals.length;
                    
                    const buySignals = data.signals.filter(s => s.status === 'buy').length;
                    const sellSignals = data.signals.filter(s => s.status === 'sell').length;
                    document.getElementById('buySignals').textContent = buySignals;
                    document.getElementById('sellSignals').textContent = sellSignals;

                    // 更新持仓列表
                    updatePositions(data.positions);

                    // 更新信号列表
                    updateSignals(data.signals);

                    // 更新时间
                    const updateTime = new Date(data.lastUpdate).toLocaleString('zh-CN');
                    document.getElementById('lastUpdate').textContent = updateTime;
                })
                .catch(error => {
                    console.error('获取数据失败:', error);
                    document.getElementById('lastUpdate').textContent = '数据获取失败';
                });
        }

        function updatePositions(positions) {
            const container = document.getElementById('positionsList');
            
            if (positions.length === 0) {
                container.innerHTML = '<div style="text-align: center; color: #666; font-size: 12px; padding: 20px;">暂无持仓</div>';
                return;
            }

            container.innerHTML = positions.map(pos => `
                <div class="position-item">
                    <div class="position-header">
                        <span class="position-symbol">${pos.symbol}</span>
                        <span class="position-side ${pos.side}">${pos.side === 'long' ? '做多' : '做空'}</span>
                    </div>
                    <div class="position-details">
                        <div class="position-detail">
                            <span>入场:</span>
                            <span>${pos.entryPrice.toFixed(4)}</span>
                        </div>
                        <div class="position-detail">
                            <span>当前:</span>
                            <span>${pos.currentPrice.toFixed(4)}</span>
                        </div>
                        <div class="position-detail">
                            <span>仓位:</span>
                            <span>${pos.size.toFixed(4)}</span>
                        </div>
                        <div class="position-detail">
                            <span>杠杆:</span>
                            <span>${pos.leverage}x</span>
                        </div>
                        <div class="position-detail">
                            <span>止损:</span>
                            <span>${pos.stopLoss.toFixed(4)}</span>
                        </div>
                        <div class="position-detail">
                            <span>止盈:</span>
                            <span>${pos.takeProfit.toFixed(4)}</span>
                        </div>
                        <div class="position-detail">
                            <span>盈亏:</span>
                            <span class="${pos.pnl >= 0 ? 'positive' : 'negative'}">${pos.pnl.toFixed(2)}</span>
                        </div>
                    </div>
                </div>
            `).join('');
        }

        function updateSignals(signals) {
            const container = document.getElementById('signalsList');
            
            container.innerHTML = signals.map(signal => `
                <div class="signal-item ${signal.status}">
                    <div class="signal-header">
                        <span class="signal-symbol">${signal.symbol}</span>
                        <span class="signal-status ${signal.status}">${signal.statusText}</span>
                    </div>
                    <div class="signal-details">
                        <span>价格: ${signal.price.toFixed(4)}</span>
                        <span>强度: ${signal.strength}</span>
                    </div>
                </div>
            `).join('');
        }

        // 页面加载时立即更新一次
        updateDashboard();

        // 每30秒自动更新
        setInterval(updateDashboard, 30000);
    </script>
</body>
</html>
'''

@app.route('/')
def dashboard():
    """主页 - 显示交易仪表板"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/data')
def get_data():
    """API接口 - 返回交易数据"""
    return jsonify(dashboard_data)

@app.route('/api/update', methods=['POST'])
def update_data():
    """API接口 - 更新交易数据（供主程序调用）"""
    global dashboard_data
    try:
        from flask import request
        new_data = request.get_json()
        if new_data:
            dashboard_data.update(new_data)
            dashboard_data['lastUpdate'] = datetime.now().isoformat()
            return jsonify({'status': 'success'})
        return jsonify({'status': 'error', 'message': 'No data provided'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/health')
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy', 
        'timestamp': datetime.now().isoformat(),
        'mode': 'railway'
    })

def update_dashboard_data(account_data, stats_data, positions_data, signals_data):
    """更新仪表板数据的函数（供主程序调用）"""
    global dashboard_data
    dashboard_data = {
        'account': account_data,
        'stats': stats_data,
        'positions': positions_data,
        'signals': signals_data,
        'lastUpdate': datetime.now().isoformat()
    }

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f"🚀 启动Railway MACD交易系统监控")
    print(f"📱 端口: {port}")
    print(f"🌐 访问地址: http://localhost:{port}")
    
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)