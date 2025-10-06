import ccxt
import pandas as pd
import traceback
import numpy as np
from datetime import datetime, timedelta
import time
import json
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

# 自定义技术指标计算函数
def calculate_macd(close, fast=6, slow=32, signal=9):
    """计算MACD指标"""
    exp1 = close.ewm(span=fast).mean()
    exp2 = close.ewm(span=slow).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_atr(high, low, close, period=14):
    """计算ATR指标"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_adx(high, low, close, period=14):
    """计算ADX指标"""
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm = minus_dm.abs()
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    return adx

# ============================================
# API配置 - 从环境变量获取
# ============================================
def get_okx_config():
    """获取OKX API配置"""
    return {
        'apiKey': os.getenv('OKX_API_KEY'),
        'secret': os.getenv('OKX_SECRET_KEY'),
        'password': os.getenv('OKX_PASSPHRASE'),
        'sandbox': False,
        'enableRateLimit': True,
    }

# ============================================
# 交易所初始化 (OKX)
# ============================================
def initialize_exchange():
    """初始化OKX交易所连接"""
    try:
        config = get_okx_config()
        if not all([config['apiKey'], config['secret'], config['password']]):
            raise ValueError("缺少必要的API配置信息")
        
        exchange = ccxt.okx(config)
        exchange.set_sandbox_mode(False)  # 实盘模式
        return exchange
    except Exception as e:
        print(f"❌ 交易所初始化失败: {str(e)}")
        return None

# ============================================
# 全局配置常量
# ============================================
# 交易对配置
SYMBOLS = [
    'BTC-USDT-SWAP', 'ETH-USDT-SWAP', 'SOL-USDT-SWAP', 'BNB-USDT-SWAP',
    'XRP-USDT-SWAP', 'DOGE-USDT-SWAP', 'ADA-USDT-SWAP', 'AVAX-USDT-SWAP',
    'SHIB-USDT-SWAP', 'DOT-USDT-SWAP', 'FIL-USDT-SWAP', 'ZRO-USDT-SWAP',
    'WIF-USDT-SWAP', 'WLD-USDT-SWAP'
]

# ============================================
# 智能杠杆配置
# ============================================
MAX_LEVERAGE_BTC = 100                       # BTC最大杠杆
MAX_LEVERAGE_ETH = 50                        # ETH最大杠杆
MAX_LEVERAGE_MAJOR = 30                      # 主流币最大杠杆
MAX_LEVERAGE_OTHERS = 25                     # 其他币种最大杠杆
LEVERAGE_MIN = 20                            # 全局最低杠杆
DEFAULT_LEVERAGE = 25                        # 默认杠杆

# 主流币种定义
MAJOR_COINS = ['BNB', 'XRP', 'ADA', 'SOL', 'DOT', 'AVAX', 'DOGE']

# MACD指标配置
MACD_FAST = 6                             # MACD快线周期
MACD_SLOW = 32                            # MACD慢线周期
MACD_SIGNAL = 9                           # MACD信号线周期

# ATR动态止盈止损配置
USE_ATR_DYNAMIC_STOPS = True                 # 启用ATR动态止盈止损
ATR_PERIOD = 14                              # ATR计算周期
ATR_STOP_LOSS_MULTIPLIER = 2.0              # ATR止损倍数
ATR_TAKE_PROFIT_MULTIPLIER = 3.0            # ATR止盈倍数
ATR_TRAILING_ACTIVATION_MULTIPLIER = 1.5    # 移动止盈激活倍数
ATR_TRAILING_CALLBACK_MULTIPLIER = 1.0      # 移动止盈回调倍数
ATR_MIN_MULTIPLIER = 1.0                    # ATR最小倍数
ATR_MAX_MULTIPLIER = 5.0                    # ATR最大倍数

# ADX配置
ADX_PERIOD = 14                              # ADX计算周期
ADX_TREND_THRESHOLD = 25                     # ADX趋势阈值
ADX_SIDEWAYS_THRESHOLD = 20                  # ADX震荡阈值

# 风险管理配置
RISK_PER_TRADE = 0.02                        # 单笔风险2%
MAX_OPEN_POSITIONS = 5                       # 最大持仓数
COOLDOWN_PERIOD = 300                        # 冷却期5分钟
MAX_DAILY_TRADES = 20                        # 每日最大交易次数

# 主循环配置
MAIN_LOOP_DELAY = 30                         # 主循环延迟30秒

# ============================================
# 全局变量
# ============================================
exchange = None
position_tracker = {
    'positions': {},
    'trailing_stops': {},
    'last_trade_time': {},
    'daily_stats': {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'trades_count': 0,
        'total_pnl': 0
    }
}

# ============================================
# 交易统计
# ============================================
trade_stats = {
    'total_trades': 0,
    'winning_trades': 0,
    'losing_trades': 0,
    'total_pnl': 0,
    'total_profit': 0,
    'total_loss': 0,
    'win_rate': 0,
    'trade_history': [],
    'initial_balance': 0,
    'current_balance': 0
}

def log_message(level, message):
    """日志记录函数"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {level}: {message}")

def test_api_connection():
    """测试交易所API连接"""
    try:
        exchange.fetch_balance()
        return True
    except Exception as e:
        log_message("ERROR", f"API连接测试失败: {str(e)}")
        return False

def get_klines(symbol, timeframe, limit=100):
    """获取K线数据"""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        if not ohlcv:
            return None
        return ohlcv
    except Exception as e:
        log_message("ERROR", f"获取 {symbol} K线数据失败: {str(e)}")
        return None

def get_smart_leverage(symbol, account_balance, atr_percentage=None):
    """根据币种和账户大小智能计算杠杆倍数"""
    try:
        base_symbol = symbol.split('-')[0].upper()
        
        if base_symbol == 'BTC':
            max_leverage = MAX_LEVERAGE_BTC
            base_leverage = 60
        elif base_symbol == 'ETH':
            max_leverage = MAX_LEVERAGE_ETH
            base_leverage = 30
        elif base_symbol in MAJOR_COINS:
            max_leverage = MAX_LEVERAGE_MAJOR
            base_leverage = 25
        else:
            max_leverage = MAX_LEVERAGE_OTHERS
            base_leverage = 25
        
        # 根据账户大小调整杠杆
        if account_balance >= 10000:
            leverage_multiplier = 1.0
        elif account_balance >= 1000:
            leverage_multiplier = 0.8
        elif account_balance >= 100:
            leverage_multiplier = 0.6
        else:
            leverage_multiplier = 0.4
        
        # 根据ATR波动性调整杠杆
        volatility_multiplier = 1.0
        if atr_percentage:
            if atr_percentage > 0.05:
                volatility_multiplier = 0.6
            elif atr_percentage > 0.03:
                volatility_multiplier = 0.8
            elif atr_percentage > 0.015:
                volatility_multiplier = 1.0
            else:
                volatility_multiplier = 1.2
        
        calculated_leverage = int(base_leverage * leverage_multiplier * volatility_multiplier)
        final_leverage = min(calculated_leverage, max_leverage)
        final_leverage = max(final_leverage, LEVERAGE_MIN)
        
        return final_leverage
        
    except Exception as e:
        log_message("ERROR", f"智能杠杆计算失败: {str(e)}")
        return DEFAULT_LEVERAGE

def get_account_info():
    """获取账户信息"""
    try:
        balance = exchange.fetch_balance()
        total_balance = balance['total']['USDT'] if 'USDT' in balance['total'] else 0
        available_balance = balance['free']['USDT'] if 'USDT' in balance['free'] else 0
        
        return {
            'total_balance': total_balance,
            'available_balance': available_balance,
            'balance_info': balance
        }
    except Exception as e:
        log_message("ERROR", f"获取账户信息失败: {str(e)}")
        return None

def process_klines(ohlcv):
    """处理K线数据并计算技术指标"""
    try:
        if not ohlcv or len(ohlcv) < 50:
            return None
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # 计算MACD指标
        try:
            macd_line, signal_line, histogram = calculate_macd(df['close'], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
            df['MACD'] = macd_line
            df['MACD_SIGNAL'] = signal_line
            df['MACD_HIST'] = histogram
        except Exception as e:
            log_message("ERROR", f"MACD计算失败: {str(e)}")
            return None
        
        # 计算ATR
        try:
            df['ATR_14'] = calculate_atr(df['high'], df['low'], df['close'], period=ATR_PERIOD)
        except Exception as e:
            log_message("WARNING", f"ATR计算失败: {str(e)}")
            df['ATR_14'] = 0
        
        # 计算ADX
        try:
            df['ADX'] = calculate_adx(df['high'], df['low'], df['close'], period=ADX_PERIOD)
        except Exception as e:
            log_message("WARNING", f"ADX计算失败: {str(e)}")
            df['ADX'] = 30  # 默认值
        
        return df
        
    except Exception as e:
        log_message("ERROR", f"处理K线数据失败: {str(e)}")
        return None

def generate_signal(symbol):
    """基于MACD金叉/死叉生成交易信号"""
    try:
        ohlcv = get_klines(symbol, '30m', limit=100)
        if not ohlcv:
            return None
        
        df = process_klines(ohlcv)
        if df is None or len(df) < 2:
            return None
        
        current_macd = df['MACD'].iloc[-1]
        current_signal = df['MACD_SIGNAL'].iloc[-1]
        prev_macd = df['MACD'].iloc[-2]
        prev_signal = df['MACD_SIGNAL'].iloc[-2]
        current_adx = df['ADX'].iloc[-1]
        current_price = df['close'].iloc[-1]
        atr_value = df['ATR_14'].iloc[-1]
        
        # 检查MACD金叉死叉
        golden_cross = prev_macd <= prev_signal and current_macd > current_signal
        death_cross = prev_macd >= prev_signal and current_macd < current_signal
        
        signal = None
        
        # 趋势确认：ADX > 25
        if current_adx > ADX_TREND_THRESHOLD:
            if golden_cross:
                signal = {
                    'symbol': symbol,
                    'side': 'long',
                    'price': current_price,
                    'signal_strength': 'strong',
                    'atr_value': atr_value,
                    'adx_value': current_adx,
                    'macd_value': current_macd,
                    'signal_value': current_signal
                }
            elif death_cross:
                signal = {
                    'symbol': symbol,
                    'side': 'short',
                    'price': current_price,
                    'signal_strength': 'strong',
                    'atr_value': atr_value,
                    'adx_value': current_adx,
                    'macd_value': current_macd,
                    'signal_value': current_signal
                }
        
        return signal
        
    except Exception as e:
        log_message("ERROR", f"{symbol} 生成信号失败: {str(e)}")
        return None

def calculate_position_size(symbol, price, total_balance):
    """计算仓位大小"""
    try:
        smart_leverage = get_smart_leverage(symbol, total_balance)
        
        # 计算本次交易分配的资金
        total_trading_fund = total_balance * 0.8
        
        # 智能分配仓位资金
        open_positions = len(position_tracker['positions'])
        if open_positions == 0:
            position_fund = total_trading_fund * 0.5
        elif open_positions == 1:
            position_fund = total_trading_fund * 0.3
        else:
            remaining_fund = total_trading_fund * 0.2
            max_additional_positions = MAX_OPEN_POSITIONS - 2
            position_fund = remaining_fund / max_additional_positions if max_additional_positions > 0 else remaining_fund
        
        # 计算仓位大小
        position_value_with_leverage = position_fund * smart_leverage
        position_size = position_value_with_leverage / price
        
        return position_size
        
    except Exception as e:
        log_message("ERROR", f"仓位计算失败: {e}")
        return 0

def execute_trade(symbol, signal, signal_strength):
    """执行交易"""
    try:
        account_info = get_account_info()
        if not account_info:
            return False
        
        side = 'buy' if signal['side'] == 'long' else 'sell'
        price = signal['price']
        position_size = calculate_position_size(symbol, price, account_info['total_balance'])
        
        if position_size <= 0:
            log_message("WARNING", f"{symbol} 计算仓位大小为0，跳过交易")
            return False
        
        # 执行市价单
        order = exchange.create_market_order(symbol, side, position_size)
        
        if order:
            log_message("SUCCESS", f"{symbol} 交易成功: {side} {position_size} @ {price}")
            
            # 记录持仓
            position_tracker['positions'][symbol] = {
                'side': signal['side'],
                'size': position_size,
                'entry_price': price,
                'timestamp': datetime.now(),
                'atr_value': signal.get('atr_value', 0)
            }
            
            return True
        
        return False
        
    except Exception as e:
        log_message("ERROR", f"{symbol} 执行交易失败: {str(e)}")
        return False

def update_trade_stats(symbol, side, pnl, entry_price, exit_price):
    """更新交易统计数据"""
    try:
        trade_stats['total_trades'] += 1
        trade_stats['total_pnl'] += pnl
        
        if pnl > 0:
            trade_stats['winning_trades'] += 1
            trade_stats['total_profit'] += pnl
        else:
            trade_stats['losing_trades'] += 1
            trade_stats['total_loss'] += abs(pnl)
        
        if trade_stats['total_trades'] > 0:
            trade_stats['win_rate'] = (trade_stats['winning_trades'] / trade_stats['total_trades']) * 100
        
        trade_record = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_percentage': (pnl / abs(entry_price * 0.01)) * 100 if entry_price > 0 else 0
        }
        
        trade_stats['trade_history'].append(trade_record)
        
        if len(trade_stats['trade_history']) > 100:
            trade_stats['trade_history'] = trade_stats['trade_history'][-100:]
        
        log_message("INFO", f"交易统计更新: 总交易{trade_stats['total_trades']}, 胜率{trade_stats['win_rate']:.2f}%")
        
    except Exception as e:
        log_message("ERROR", f"更新交易统计失败: {str(e)}")

def check_positions():
    """检查持仓状态"""
    try:
        for symbol in list(position_tracker['positions'].keys()):
            signal = generate_signal(symbol)
            if not signal:
                continue
            
            position = position_tracker['positions'][symbol]
            current_price = signal['price']
            
            # 检查平仓条件
            should_close = False
            if position['side'] == 'long' and signal['side'] == 'short':
                should_close = True
            elif position['side'] == 'short' and signal['side'] == 'long':
                should_close = True
            
            if should_close:
                close_position(symbol, "MACD反向信号")
                
    except Exception as e:
        log_message("ERROR", f"检查持仓状态失败: {str(e)}")

def close_position(symbol, reason="手动平仓"):
    """平仓指定交易对"""
    try:
        if symbol not in position_tracker['positions']:
            return False
        
        position = position_tracker['positions'][symbol]
        side = 'sell' if position['side'] == 'long' else 'buy'
        size = abs(position['size'])
        
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        
        order = exchange.create_market_order(symbol, side, size, None, None, {
            'reduceOnly': True
        })
        
        if order:
            log_message("SUCCESS", f"平仓成功: {symbol} {reason}")
            
            # 计算盈亏
            entry_price = position['entry_price']
            if position['side'] == 'long':
                pnl = (current_price - entry_price) * size
            else:
                pnl = (entry_price - current_price) * size
            
            # 更新交易统计
            update_trade_stats(symbol, position['side'], pnl, entry_price, current_price)
            
            # 清理持仓记录
            del position_tracker['positions'][symbol]
            
            return True
        
        return False
        
    except Exception as e:
        log_message("ERROR", f"平仓 {symbol} 失败: {str(e)}")
        return False

def trading_loop():
    """主交易循环"""
    try:
        log_message("SUCCESS", "开始交易循环...")
        
        while True:
            try:
                # 检查现有持仓
                check_positions()
                
                # 获取账户信息
                account_info = get_account_info()
                if not account_info:
                    log_message("ERROR", "获取账户信息失败，等待下次循环")
                    time.sleep(60)
                    continue
                
                # 显示当前统计
                if trade_stats['total_trades'] > 0:
                    log_message("INFO", f"当前交易统计 - 总交易: {trade_stats['total_trades']}, "
                              f"胜率: {trade_stats['win_rate']:.2f}%, "
                              f"总盈亏: {trade_stats['total_pnl']:.2f} USDT")
                
                # 检查每个交易对
                for symbol in SYMBOLS:
                    try:
                        # 跳过已有持仓的交易对
                        if symbol in position_tracker['positions']:
                            continue
                        
                        # 检查持仓数量限制
                        if len(position_tracker['positions']) >= MAX_OPEN_POSITIONS:
                            break
                        
                        # 生成交易信号
                        signal = generate_signal(symbol)
                        if signal:
                            log_message("INFO", f"{symbol} 发现信号: {signal['side']} @ {signal['price']:.4f}")
                            execute_trade(symbol, signal, signal['signal_strength'])
                        
                        time.sleep(1)  # 短暂延迟避免API限制
                        
                    except Exception as e:
                        log_message("ERROR", f"处理 {symbol} 失败: {str(e)}")
                        continue
                
                # 主循环延迟
                log_message("INFO", f"交易循环完成，等待{MAIN_LOOP_DELAY}秒...")
                time.sleep(MAIN_LOOP_DELAY)
                
            except Exception as e:
                log_message("ERROR", f"交易循环中出错: {str(e)}")
                time.sleep(60)
                
    except KeyboardInterrupt:
        log_message("INFO", "收到退出信号，正在安全关闭...")
    except Exception as e:
        log_message("ERROR", f"交易循环启动失败: {str(e)}")
        traceback.print_exc()

def start_trading_system():
    """启动交易系统"""
    global exchange
    try:
        # 初始化交易所连接
        exchange = initialize_exchange()
        if not exchange:
            log_message("ERROR", "交易所初始化失败")
            return
        
        # 测试API连接
        if not test_api_connection():
            log_message("ERROR", "API连接测试失败，请检查配置")
            return
        
        # 显示启动信息
        log_message("SUCCESS", "MACD(6,32,9)策略交易系统启动成功")
        log_message("INFO", f"智能杠杆系统: BTC最大{MAX_LEVERAGE_BTC}x, ETH最大{MAX_LEVERAGE_ETH}x")
        log_message("INFO", f"交易对数量: {len(SYMBOLS)}")
        log_message("INFO", f"最大持仓数: {MAX_OPEN_POSITIONS}")
        
        # 启动交易循环
        trading_loop()
        
    except Exception as e:
        log_message("ERROR", f"启动交易系统失败: {str(e)}")
        traceback.print_exc()

# ============================================
# 主程序入口
# ============================================
if __name__ == "__main__":
    start_trading_system()