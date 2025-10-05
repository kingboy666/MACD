#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MACD交易系统 - Web仪表板启动器
独立启动Web界面，可以与主交易程序分离运行
"""

from flask import Flask, jsonify, send_from_directory
import json
import time
from datetime import datetime
import os
import random

app = Flask(__name__)

# 模拟交易数据
def generate_mock_data():
    """生成模拟交易数据"""
    
    # 模拟账户数据
    base_balance = 1000 + random.uniform(-100, 500)
    daily_pnl = random.uniform(-50, 100)
    total_pnl = random.uniform(-200, 800)
    
    # 模拟交易统计
    total_trades = random.randint(20, 100)
    win_trades = random.randint(int(total_trades * 0.4), int(total_trades * 0.8))
    lose_trades = total_trades - win_trades
    win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
    
    # 模拟持仓数据
    positions = []
    active_positions = random.randint(0, 3)
    
    position_templates = [
        {'symbol': 'BTC-USDT-SWAP', 'base_price': 67000, 'leverage': 50},
        {'symbol': 'ETH-USDT-SWAP', 'base_price': 2600, 'leverage': 30},
        {'symbol': 'SOL-USDT-SWAP', 'base_price': 140, 'leverage': 20},
    ]
    
    for i in range(active_positions):
        if i < len(position_templates):
            template = position_templates[i]
            side = random.choice(['long', 'short'])
            entry_price = template['base_price'] * random.uniform(0.98, 1.02)
            current_price = entry_price * random.uniform(0.95, 1.05)
            size = random.uniform(0.01, 1.0)
            
            if side == 'long':
                pnl = (current_price - entry_price) * size * template['leverage']
                stop_loss = entry_price * 0.98
                take_profit = entry_price * 1.04
            else:
                pnl = (entry_price - current_price) * size * template['leverage']
                stop_loss = entry_price * 1.02
                take_profit = entry_price * 0.96
            
            positions.append({
                'symbol': template['symbol'],
                'side': side,
                'entryPrice': entry_price,
                'currentPrice': current_price,
                'size': size,
                'leverage': template['leverage'],
                'stopLoss': stop_loss,
                'takeProfit': take_profit,
                'pnl': pnl
            })
    
    # 模拟信号数据
    trading_pairs = [
        'BTC-USDT-SWAP', 'ETH-USDT-SWAP', 'SOL-USDT-SWAP', 'BNB-USDT-SWAP',
        'XRP-USDT-SWAP', 'DOGE-USDT-SWAP', 'ADA-USDT-SWAP', 'AVAX-USDT-SWAP',
        'SHIB-USDT-SWAP', 'DOT-USDT-SWAP', 'FIL-USDT-SWAP', 'ZRO-USDT-SWAP',
        'WIF-USDT-SWAP', 'WLD-USDT-SWAP'
    ]
    
    price_bases = {
        'BTC-USDT-SWAP': 67000, 'ETH-USDT-SWAP': 2600, 'SOL-USDT-SWAP': 140,
        'BNB-USDT-SWAP': 590, 'XRP-USDT-SWAP': 0.52, 'DOGE-USDT-SWAP': 0.14,
        'ADA-USDT-SWAP': 0.35, 'AVAX-USDT-SWAP': 25, 'SHIB-USDT-SWAP': 0.000018,
        'DOT-USDT-SWAP': 4.2, 'FIL-USDT-SWAP': 3.8, 'ZRO-USDT-SWAP': 4.5,
        'WIF-USDT-SWAP': 2.1, 'WLD-USDT-SWAP': 2.3
    }
    
    signals = []
    for symbol in trading_pairs:
        base_price = price_bases.get(symbol, 100)
        current_price = base_price * random.uniform(0.98, 1.02)
        
        # 随机生成信号
        signal_type = random.choices(
            ['none', 'buy', 'sell'], 
            weights=[70, 15, 15]  # 70%无信号，15%做多，15%做空
        )[0]
        
        if signal_type == 'buy':
            status = 'buy'
            status_text = '做多'
            strength = random.randint(60, 90)
        elif signal_type == 'sell':
            status = 'sell'
            status_text = '做空'
            strength = random.randint(60, 90)
        else:
            status = 'none'
            status_text = '无信号'
            strength = 0
        
        signals.append({
            'symbol': symbol,
            'status': status,
            'statusText': status_text,
            'price': current_price,
            'strength': strength
        })
    
    return {
        'account': {
            'totalBalance': base_balance,
            'freeBalance': base_balance * 0.7,
            'dailyPnl': daily_pnl,
            'totalPnl': total_pnl
        },
        'stats': {
            'totalTrades': total_trades,
            'winTrades': win_trades,
            'loseTrades': lose_trades,
            'winRate': win_rate
        },
        'positions': positions,
        'signals': signals,
        'lastUpdate': datetime.now().isoformat()
    }

@app.route('/')
def dashboard():
    """主页 - 显示交易仪表板"""
    try:
        with open('trading_dashboard.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return """
        <html>
        <head><title>文件未找到</title></head>
        <body>
            <h1>错误：trading_dashboard.html 文件未找到</h1>
            <p>请确保 trading_dashboard.html 文件在当前目录中。</p>
        </body>
        </html>
        """, 404

@app.route('/api/dashboard')
def api_dashboard():
    """API接口 - 返回仪表板数据"""
    try:
        # 尝试读取真实交易数据
        if os.path.exists('dashboard_data.json'):
            with open('dashboard_data.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
            return jsonify(data)
        else:
            # 如果没有真实数据文件，返回模拟数据
            return jsonify(generate_mock_data())
    except Exception as e:
        print(f"读取数据失败: {e}")
        # 返回错误时使用模拟数据
        return jsonify(generate_mock_data())

@app.route('/api/status')
def api_status():
    """API接口 - 返回系统状态"""
    return jsonify({
        'status': 'running',
        'mode': 'demo',
        'timestamp': datetime.now().isoformat(),
        'message': '演示模式 - 使用模拟数据'
    })

@app.route('/health')
def health_check():
    """健康检查接口"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    print("🚀 启动MACD交易系统Web仪表板")
    print("=" * 50)
    print("📱 访问地址: http://localhost:8080")
    print("🔄 演示模式: 使用模拟数据")
    print("💡 提示: 数据每30秒自动更新")
    print("=" * 50)
    
    try:
        app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        input("按回车键退出...")