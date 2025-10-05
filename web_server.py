from flask import Flask, jsonify, render_template_string, send_from_directory
import json
import threading
import time
from datetime import datetime
import os

app = Flask(__name__)

# 全局数据存储
dashboard_data = {
    'account': {
        'totalBalance': 0,
        'freeBalance': 0,
        'dailyPnl': 0,
        'totalPnl': 0
    },
    'stats': {
        'totalTrades': 0,
        'winTrades': 0,
        'loseTrades': 0,
        'winRate': 0
    },
    'positions': [],
    'signals': [],
    'lastUpdate': datetime.now().isoformat()
}

class DashboardUpdater:
    """负责更新仪表板数据的类"""
    
    def __init__(self):
        self.running = False
        self.thread = None
    
    def start(self):
        """启动数据更新线程"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._update_loop, daemon=True)
            self.thread.start()
            print("📊 仪表板数据更新器已启动")
    
    def stop(self):
        """停止数据更新线程"""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _update_loop(self):
        """数据更新循环"""
        while self.running:
            try:
                self.update_data()
                time.sleep(30)  # 每30秒更新一次
            except Exception as e:
                print(f"❌ 更新仪表板数据失败: {e}")
                time.sleep(10)  # 出错时等待10秒再重试
    
    def update_data(self):
        """更新仪表板数据"""
        global dashboard_data
        
        try:
            # 从主程序的全局变量获取数据
            from main import (
                trade_stats, position_tracker, latest_signals, 
                get_account_info, exchange
            )
            
            # 更新账户信息
            account_info = get_account_info()
            if account_info:
                dashboard_data['account'] = {
                    'totalBalance': account_info.get('total_balance', 0),
                    'freeBalance': account_info.get('free_balance', 0),
                    'dailyPnl': trade_stats.get('daily_pnl', 0),
                    'totalPnl': trade_stats.get('total_profit_loss', 0)
                }
            
            # 更新交易统计
            total_trades = trade_stats.get('total_trades', 0)
            win_trades = trade_stats.get('winning_trades', 0)
            lose_trades = trade_stats.get('losing_trades', 0)
            win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
            
            dashboard_data['stats'] = {
                'totalTrades': total_trades,
                'winTrades': win_trades,
                'loseTrades': lose_trades,
                'winRate': win_rate
            }
            
            # 更新持仓信息
            positions = []
            for symbol, position in position_tracker['positions'].items():
                try:
                    # 获取当前价格
                    ticker = exchange.fetch_ticker(symbol)
                    current_price = ticker['last']
                    
                    positions.append({
                        'symbol': symbol,
                        'side': position['side'],
                        'entryPrice': position['entry_price'],
                        'currentPrice': current_price,
                        'size': position['size'],
                        'leverage': position.get('leverage', 20),
                        'stopLoss': position.get('sl', 0),
                        'takeProfit': position.get('tp', 0),
                        'pnl': position.get('pnl', 0)
                    })
                except Exception as e:
                    print(f"⚠️ 获取{symbol}价格失败: {e}")
            
            dashboard_data['positions'] = positions
            
            # 更新信号信息
            signals = []
            trading_pairs = [
                'BTC-USDT-SWAP', 'ETH-USDT-SWAP', 'SOL-USDT-SWAP', 'BNB-USDT-SWAP',
                'XRP-USDT-SWAP', 'DOGE-USDT-SWAP', 'ADA-USDT-SWAP', 'AVAX-USDT-SWAP',
                'SHIB-USDT-SWAP', 'DOT-USDT-SWAP', 'FIL-USDT-SWAP', 'ZRO-USDT-SWAP',
                'WIF-USDT-SWAP', 'WLD-USDT-SWAP'
            ]
            
            for symbol in trading_pairs:
                try:
                    # 获取当前价格
                    ticker = exchange.fetch_ticker(symbol)
                    current_price = ticker['last']
                    
                    # 获取最新信号
                    signal_info = latest_signals.get(symbol, (None, 0, None))
                    signal, strength, timestamp = signal_info
                    
                    if signal == "做多":
                        status = 'buy'
                        status_text = '做多'
                    elif signal == "做空":
                        status = 'sell'
                        status_text = '做空'
                    else:
                        status = 'none'
                        status_text = '无信号'
                    
                    signals.append({
                        'symbol': symbol,
                        'status': status,
                        'statusText': status_text,
                        'price': current_price,
                        'strength': strength or 0
                    })
                except Exception as e:
                    print(f"⚠️ 获取{symbol}信号失败: {e}")
            
            dashboard_data['signals'] = signals
            dashboard_data['lastUpdate'] = datetime.now().isoformat()
            
        except ImportError:
            # 如果无法导入主程序模块，使用模拟数据
            print("⚠️ 无法连接到主交易程序，使用模拟数据")
            self._load_mock_data()
        except Exception as e:
            print(f"❌ 更新数据时出错: {e}")
    
    def _load_mock_data(self):
        """加载模拟数据用于演示"""
        global dashboard_data
        
        dashboard_data.update({
            'account': {
                'totalBalance': 1250.75 + (time.time() % 100),
                'freeBalance': 850.25,
                'dailyPnl': 45.30,
                'totalPnl': 156.80
            },
            'stats': {
                'totalTrades': 28,
                'winTrades': 17,
                'loseTrades': 11,
                'winRate': 60.7
            },
            'positions': [
                {
                    'symbol': 'BTC-USDT-SWAP',
                    'side': 'long',
                    'entryPrice': 67500.50,
                    'currentPrice': 67850.25 + (time.time() % 50),
                    'size': 0.015,
                    'leverage': 50,
                    'stopLoss': 66150.49,
                    'takeProfit': 69525.65,
                    'pnl': 25.60
                }
            ],
            'signals': [
                {'symbol': 'BTC-USDT-SWAP', 'status': 'none', 'statusText': '无信号', 'price': 67850.25, 'strength': 0},
                {'symbol': 'ETH-USDT-SWAP', 'status': 'buy', 'statusText': '做多', 'price': 2635.45, 'strength': 75},
                {'symbol': 'SOL-USDT-SWAP', 'status': 'sell', 'statusText': '做空', 'price': 145.67, 'strength': 68}
            ]
        })

# 创建数据更新器实例
updater = DashboardUpdater()

@app.route('/')
def dashboard():
    """主页 - 显示交易仪表板"""
    try:
        with open('trading_dashboard.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "仪表板文件未找到", 404

@app.route('/api/dashboard')
def api_dashboard():
    """API接口 - 返回仪表板数据"""
    return jsonify(dashboard_data)

@app.route('/api/status')
def api_status():
    """API接口 - 返回系统状态"""
    return jsonify({
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'updater_running': updater.running
    })

def start_web_server(host='0.0.0.0', port=8080, debug=False):
    """启动Web服务器"""
    print(f"🌐 启动Web仪表板服务器...")
    print(f"📱 访问地址: http://localhost:{port}")
    print(f"🔄 数据每30秒自动更新")
    
    # 启动数据更新器
    updater.start()
    
    try:
        app.run(host=host, port=port, debug=debug, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 Web服务器已停止")
    finally:
        updater.stop()

if __name__ == '__main__':
    start_web_server()