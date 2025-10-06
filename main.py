import ccxt
import pandas as pd
import traceback
import numpy as np
from datetime import datetime, timedelta, timezone
import time
import json
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

# 自定义技术指标计算函数
def calculate_macd(close, fast=6, slow=16, signal=9):
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

# 交易对最小数量配置
MIN_TRADE_AMOUNT = {
    'BTC-USDT-SWAP': 0.001,
    'ETH-USDT-SWAP': 0.01,
    'SOL-USDT-SWAP': 0.1,
    'BNB-USDT-SWAP': 1.0,  # 根据错误信息，BNB的最小精度是1
    'XRP-USDT-SWAP': 10,
    'DOGE-USDT-SWAP': 100,
    'ADA-USDT-SWAP': 10,
    'AVAX-USDT-SWAP': 0.1,
    'SHIB-USDT-SWAP': 1000000,
    'DOT-USDT-SWAP': 1,
    'FIL-USDT-SWAP': 0.1,
    'ZRO-USDT-SWAP': 10,
    'WIF-USDT-SWAP': 0.1,
    'WLD-USDT-SWAP': 0.1
}

# MACD指标配置
MACD_FAST = 6                             # MACD快线周期
MACD_SLOW = 16                            # MACD慢线周期
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

# 账户与保证金模式（用于 OKX 下单参数）
ACCOUNT_MODE = 'hedge'                       # 可选 'hedge'（双向持仓）或 'one-way'（单向持仓）
TD_MODE = 'cross'                            # 保证金模式：'cross' 全仓 或 'isolated' 逐仓

# ============================================
# 全局变量
# ============================================
exchange = None
position_tracker = {
    'positions': {},
    'trailing_stops': {},
    'last_trade_time': {},
    'pending_signals': {},  # 跟踪等待K线收盘的信号
    'daily_stats': {
        'date': datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d'),
        'trades_count': 0,
        'total_pnl': 0
    }
}

# 全局订单跟踪字典，防止重复设置止盈止损
order_tracking = {}

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
    # 使用UTC+8时区
    utc8_timezone = timezone(timedelta(hours=8))
    timestamp = datetime.now(utc8_timezone).strftime('%Y-%m-%d %H:%M:%S')
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
        
        # 计算K线阴阳线
        df['is_bullish'] = df['close'] > df['open']  # 阳线: 收盘价大于开盘价
        df['is_bearish'] = df['close'] < df['open']  # 阴线: 收盘价小于开盘价
        
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

def calculate_bollinger_bands(close, period=20, std_dev=2):
    """计算布林带指标"""
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

def calculate_rsi(close, period=14):
    """计算RSI指标"""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """计算随机指标"""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = ((close - lowest_low) / (highest_high - lowest_low)) * 100
    d = k.rolling(window=d_period).mean()
    return k, d

def generate_signal(symbol):
    """基于MACD趋势策略和布林带震荡策略生成交易信号"""
    try:
        ohlcv = get_klines(symbol, '30m', limit=100)
        if not ohlcv:
            return None
        
        df = process_klines(ohlcv)
        if df is None or len(df) < 50:
            return None
        
        # 计算布林带震荡策略指标
        df['BB_UPPER'], df['BB_MIDDLE'], df['BB_LOWER'] = calculate_bollinger_bands(df['close'])
        df['RSI'] = calculate_rsi(df['close'])
        df['STOCH_K'], df['STOCH_D'] = calculate_stochastic(df['high'], df['low'], df['close'])
        
        # 获取当前时间戳，检查K线是否已收盘
        import time
        current_timestamp = int(time.time() * 1000)
        current_kline = ohlcv[-1]
        current_kline_start = current_kline[0]
        current_kline_end = current_kline_start + 30 * 60 * 1000  # 30分钟K线结束时间
        kline_completed = current_timestamp >= current_kline_end
        
        # 获取当前数据
        current_macd = df['MACD'].iloc[-1]
        current_signal = df['MACD_SIGNAL'].iloc[-1]
        prev_macd = df['MACD'].iloc[-2]
        prev_signal = df['MACD_SIGNAL'].iloc[-2]
        current_adx = df['ADX'].iloc[-1]
        current_price = df['close'].iloc[-1]
        current_open = df['open'].iloc[-1]
        current_close = df['close'].iloc[-1]
        atr_value = df['ATR_14'].iloc[-1]
        
        # 布林带震荡策略数据
        current_bb_upper = df['BB_UPPER'].iloc[-1]
        current_bb_lower = df['BB_LOWER'].iloc[-1]
        current_bb_middle = df['BB_MIDDLE'].iloc[-1]
        current_rsi = df['RSI'].iloc[-1]
        current_stoch_k = df['STOCH_K'].iloc[-1]
        current_stoch_d = df['STOCH_D'].iloc[-1]
        
        # 检查MACD金叉死叉
        golden_cross = prev_macd <= prev_signal and current_macd > current_signal
        death_cross = prev_macd >= prev_signal and current_macd < current_signal
        
        # 检查K线阴阳线
        is_bullish = current_close > current_open  # 阳线：收盘价大于开盘价
        is_bearish = current_close < current_open  # 阴线：收盘价小于开盘价
        
        signal = None
        
        # 策略选择：根据ADX判断市场状态
        if current_adx > ADX_TREND_THRESHOLD:  # 趋势行情 - 使用MACD策略
            # 严格信号确认：ADX显示趋势 + MACD交叉 + K线确认 + K线收盘确认
            if golden_cross and is_bullish:
                if kline_completed:
                    signal = {
                        'symbol': symbol,
                        'side': 'long',
                        'price': current_price,
                        'signal_strength': 'strong',
                        'atr_value': atr_value,
                        'adx_value': current_adx,
                        'macd_value': current_macd,
                        'signal_value': current_signal,
                        'is_bullish': is_bullish,
                        'confirmation_type': 'MACD金叉+阳线确认+ADX趋势+K线收盘',
                        'strategy_type': 'trend'
                    }
                    log_message("DEBUG", f"{symbol} 趋势策略做多信号确认: ADX={current_adx:.2f}, 金叉确认, 阳线确认, K线已收盘")
                else:
                    time_remaining = (current_kline_end - current_timestamp) / 1000
                    log_message("DEBUG", f"{symbol} 做多信号条件满足但等待K线收盘 (还需等待{time_remaining:.0f}秒)")
                    # 不返回None，而是记录信号状态，等待K线收盘后重新检查
                    return {
                        'symbol': symbol,
                        'side': 'long',
                        'price': current_price,
                        'signal_strength': 'pending',
                        'atr_value': atr_value,
                        'adx_value': current_adx,
                        'macd_value': current_macd,
                        'signal_value': current_signal,
                        'is_bullish': is_bullish,
                        'confirmation_type': 'MACD金叉+阳线确认+ADX趋势+等待K线收盘',
                        'strategy_type': 'trend',
                        'kline_pending': True,
                        'time_remaining': time_remaining
                    }
                
            elif death_cross and is_bearish:
                if kline_completed:
                    signal = {
                        'symbol': symbol,
                        'side': 'short',
                        'price': current_price,
                        'signal_strength': 'strong',
                        'atr_value': atr_value,
                        'adx_value': current_adx,
                        'macd_value': current_macd,
                        'signal_value': current_signal,
                        'is_bearish': is_bearish,
                        'confirmation_type': 'MACD死叉+阴线确认+ADX趋势+K线收盘',
                        'strategy_type': 'trend'
                    }
                    log_message("DEBUG", f"{symbol} 趋势策略做空信号确认: ADX={current_adx:.2f}, 死叉确认, 阴线确认, K线已收盘")
                else:
                    time_remaining = (current_kline_end - current_timestamp) / 1000
                    log_message("DEBUG", f"{symbol} 做空信号条件满足但等待K线收盘 (还需等待{time_remaining:.0f}秒)")
                    # 不返回None，而是记录信号状态，等待K线收盘后重新检查
                    return {
                        'symbol': symbol,
                        'side': 'short',
                        'price': current_price,
                        'signal_strength': 'pending',
                        'atr_value': atr_value,
                        'adx_value': current_adx,
                        'macd_value': current_macd,
                        'signal_value': current_signal,
                        'is_bearish': is_bearish,
                        'confirmation_type': 'MACD死叉+阴线确认+ADX趋势+等待K线收盘',
                        'strategy_type': 'trend',
                        'kline_pending': True,
                        'time_remaining': time_remaining
                    }
            
            # 如果没有符合条件的信号，返回None
            if signal is None:
                return None
        
        elif current_adx < ADX_SIDEWAYS_THRESHOLD:  # 震荡行情 - 使用布林带策略
            # 布林带震荡策略信号
            bb_signal = None
            
            # 检查布林带位置和指标确认
            price_near_bb_lower = current_price <= current_bb_lower * 1.02  # 价格接近下轨
            price_near_bb_upper = current_price >= current_bb_upper * 0.98  # 价格接近上轨
            
            # RSI超卖超买确认
            rsi_oversold = current_rsi < 30  # RSI超卖
            rsi_overbought = current_rsi > 70  # RSI超买
            
            # 随机指标确认
            stoch_oversold = current_stoch_k < 20 and current_stoch_d < 20  # 随机指标超卖
            stoch_overbought = current_stoch_k > 80 and current_stoch_d > 80  # 随机指标超买
            
            # 震荡策略做多信号：价格接近下轨 + RSI超卖 + 随机指标超卖 + 阳线确认 + K线收盘确认
            if price_near_bb_lower and rsi_oversold and stoch_oversold and is_bullish:
                if kline_completed:
                    bb_signal = {
                        'symbol': symbol,
                        'side': 'long',
                        'price': current_price,
                        'signal_strength': 'medium',
                        'atr_value': atr_value,
                        'adx_value': current_adx,
                        'bb_upper': current_bb_upper,
                        'bb_lower': current_bb_lower,
                        'bb_middle': current_bb_middle,
                        'rsi_value': current_rsi,
                        'stoch_k': current_stoch_k,
                        'stoch_d': current_stoch_d,
                        'is_bullish': is_bullish,
                        'confirmation_type': '布林带下轨+RSI超卖+随机超卖+阳线确认+K线收盘',
                        'strategy_type': 'oscillation'
                    }
                    log_message("DEBUG", f"{symbol} 震荡策略做多信号确认: ADX={current_adx:.2f}, RSI={current_rsi:.1f}, 价格接近下轨, K线已收盘")
                else:
                    time_remaining = (current_kline_end - current_timestamp) / 1000
                    log_message("DEBUG", f"{symbol} 震荡做多信号条件满足但等待K线收盘 (还需等待{time_remaining:.0f}秒)")
                    return None  # 等待K线收盘，不返回信号
            
            # 震荡策略做空信号：价格接近上轨 + RSI超买 + 随机指标超买 + 阴线确认 + K线收盘确认
            elif price_near_bb_upper and rsi_overbought and stoch_overbought and is_bearish:
                if kline_completed:
                    bb_signal = {
                        'symbol': symbol,
                        'side': 'short',
                        'price': current_price,
                        'signal_strength': 'medium',
                        'atr_value': atr_value,
                        'adx_value': current_adx,
                        'bb_upper': current_bb_upper,
                        'bb_lower': current_bb_lower,
                        'bb_middle': current_bb_middle,
                        'rsi_value': current_rsi,
                        'stoch_k': current_stoch_k,
                        'stoch_d': current_stoch_d,
                        'is_bearish': is_bearish,
                        'confirmation_type': '布林带上轨+RSI超买+随机超买+阴线确认+K线收盘',
                        'strategy_type': 'oscillation'
                    }
                    log_message("DEBUG", f"{symbol} 震荡策略做空信号确认: ADX={current_adx:.2f}, RSI={current_rsi:.1f}, 价格接近上轨, K线已收盘")
                else:
                    time_remaining = (current_kline_end - current_timestamp) / 1000
                    log_message("DEBUG", f"{symbol} 震荡做空信号条件满足但等待K线收盘 (还需等待{time_remaining:.0f}秒)")
                    return None  # 等待K线收盘，不返回信号
            
            # 中等强度信号：缺少一个指标确认但其他条件满足 + K线收盘确认
            elif price_near_bb_lower and (rsi_oversold or stoch_oversold) and is_bullish:
                if kline_completed:
                    bb_signal = {
                        'symbol': symbol,
                        'side': 'long',
                        'price': current_price,
                        'signal_strength': 'weak',
                        'atr_value': atr_value,
                        'adx_value': current_adx,
                        'bb_upper': current_bb_upper,
                        'bb_lower': current_bb_lower,
                        'bb_middle': current_bb_middle,
                        'rsi_value': current_rsi,
                        'stoch_k': current_stoch_k,
                        'stoch_d': current_stoch_d,
                        'is_bullish': is_bullish,
                        'confirmation_type': '布林带下轨+部分指标确认+阳线确认+K线收盘',
                        'strategy_type': 'oscillation'
                    }
                    log_message("DEBUG", f"{symbol} 震荡策略弱做多信号确认: ADX={current_adx:.2f}, 价格接近下轨, K线已收盘")
                else:
                    time_remaining = (current_kline_end - current_timestamp) / 1000
                    log_message("DEBUG", f"{symbol} 震荡弱做多信号条件满足但等待K线收盘 (还需等待{time_remaining:.0f}秒)")
                    return None  # 等待K线收盘，不返回信号
            
            elif price_near_bb_upper and (rsi_overbought or stoch_overbought) and is_bearish:
                if kline_completed:
                    bb_signal = {
                        'symbol': symbol,
                        'side': 'short',
                        'price': current_price,
                        'signal_strength': 'weak',
                        'atr_value': atr_value,
                        'adx_value': current_adx,
                        'bb_upper': current_bb_upper,
                        'bb_lower': current_bb_lower,
                        'bb_middle': current_bb_middle,
                        'rsi_value': current_rsi,
                        'stoch_k': current_stoch_k,
                        'stoch_d': current_stoch_d,
                        'is_bearish': is_bearish,
                        'confirmation_type': '布林带上轨+部分指标确认+阴线确认+K线收盘',
                        'strategy_type': 'oscillation'
                    }
                    log_message("DEBUG", f"{symbol} 震荡策略弱做空信号确认: ADX={current_adx:.2f}, 价格接近上轨, K线已收盘")
                else:
                    time_remaining = (current_kline_end - current_timestamp) / 1000
                    log_message("DEBUG", f"{symbol} 震荡弱做空信号条件满足但等待K线收盘 (还需等待{time_remaining:.0f}秒)")
                    return None  # 等待K线收盘，不返回信号
            
            # 如果没有符合条件的信号，返回None
            if bb_signal is None:
                return None
            
            signal = bb_signal
        
        else:  # 中等趋势强度 - 优先使用趋势策略
            if golden_cross and is_bullish:
                log_message("DEBUG", f"{symbol} 中等趋势做多信号: ADX={current_adx:.2f}, 金叉确认, 阳线确认")
            elif death_cross and is_bearish:
                log_message("DEBUG", f"{symbol} 中等趋势做空信号: ADX={current_adx:.2f}, 死叉确认, 阴线确认")
        
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
        
        # 确保仓位大小不低于交易对的最小数量限制
        min_amount = MIN_TRADE_AMOUNT.get(symbol, 0.001)  # 默认最小数量
        
        # 计算购买最小数量所需的资金
        required_fund_with_leverage = min_amount * price / smart_leverage
        
        # 检查用户是否有足够资金购买最小数量
        if required_fund_with_leverage > position_fund:
            log_message("WARNING", f"{symbol} 资金不足，需要 {required_fund_with_leverage:.4f} U，但仅有 {position_fund:.4f} U 可用于本交易")
            return 0
        
        if position_size < min_amount:
            log_message("WARNING", f"{symbol} 计算的仓位大小 {position_size:.6f} 低于最小数量 {min_amount}，已调整为最小数量")
            position_size = min_amount
        
        # 检查金额精度：确保交易金额大于最小金额精度（1 USDT）
        trade_value = position_size * price
        if trade_value < 1.0:
            log_message("WARNING", f"{symbol} 交易金额 {trade_value:.4f} USDT 低于最小金额精度 1 USDT，已调整")
            # 调整到最小金额精度
            position_size = 1.0 / price
            # 再次检查调整后的数量是否满足最小数量要求
            if position_size < min_amount:
                position_size = min_amount
                # 如果调整后仍然不满足金额精度，则跳过交易
                trade_value = position_size * price
                if trade_value < 1.0:
                    log_message("ERROR", f"{symbol} 调整后交易金额 {trade_value:.4f} USDT 仍低于最小金额精度 1 USDT，跳过交易")
                    return 0
        
        return position_size
        
    except Exception as e:
        log_message("ERROR", f"仓位计算失败: {e}")
        return 0

def execute_trade(symbol, signal, signal_strength):
    """执行交易"""
    try:
        # 检查信号状态，如果是pending状态则不执行交易
        if signal_strength == 'pending':
            log_message("DEBUG", f"{symbol} 信号处于pending状态，等待K线收盘确认，不执行交易")
            return False
        
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
        order = exchange.create_order(
            symbol,
            'market',
            side,
            position_size,
            None,
            {
                'tdMode': 'cross',
                'posSide': 'long' if signal['side'] == 'long' else 'short'
            }
        )
        
        if order:
            log_message("SUCCESS", f"{symbol} 交易成功: {side} {position_size} @ {price}")
            
            # 记录持仓
            position_tracker['positions'][symbol] = {
                'symbol': symbol,
                'side': signal['side'],
                'size': position_size,
                'entry_price': price,
                'timestamp': datetime.now(timezone(timedelta(hours=8))),
                'atr_value': signal.get('atr_value', 0)
            }
            
            # 开仓后立即设置止盈止损
            try:
                setup_missing_stop_orders(position_tracker['positions'][symbol], symbol)
            except Exception as e:
                log_message("WARNING", f"同步设置止盈止损失败 {symbol}: {e}")
            
            return True
        
        return False
        
    except Exception as e:
        log_message("ERROR", f"{symbol} 执行交易失败: {str(e)}")
        return False

def calculate_stop_loss_take_profit(symbol, price, signal, atr_value):
    """计算止损止盈价格"""
    try:
        # 获取当前价格用于验证
        current_price = float(exchange.fetch_ticker(symbol)['last'])
        
        if USE_ATR_DYNAMIC_STOPS and atr_value and atr_value > 0:
            # 使用ATR动态计算
            atr_sl_multiplier = max(ATR_MIN_MULTIPLIER, min(ATR_MAX_MULTIPLIER, ATR_STOP_LOSS_MULTIPLIER))
            atr_tp_multiplier = max(ATR_MIN_MULTIPLIER, min(ATR_MAX_MULTIPLIER, ATR_TAKE_PROFIT_MULTIPLIER))
            
            if signal == 'long':
                stop_loss = max(price * 0.95, price - (atr_value * atr_sl_multiplier))  # 至少5%止损
                take_profit = price + (atr_value * atr_tp_multiplier)
            else:  # short
                stop_loss = min(price * 1.05, price + (atr_value * atr_sl_multiplier))  # 至少5%止损
                take_profit = max(price * 0.94, price - (atr_value * atr_tp_multiplier))  # 至少6%止盈
        else:
            # 固定百分比止损止盈
            if signal == 'long':
                stop_loss = price * 0.95  # 5%止损
                take_profit = price * 1.06  # 6%止盈
            else:  # short
                stop_loss = price * 1.05  # 5%止损
                take_profit = price * 0.94  # 6%止盈
        
        # 严格验证止盈止损价格合理性
        if signal == 'long':
            # 做多：止盈必须高于入场价，止损必须低于入场价
            if take_profit <= price:
                take_profit = price * 1.06  # 确保止盈高于入场价
            if stop_loss >= price:
                stop_loss = price * 0.95  # 确保止损低于入场价
            # 额外验证：止盈必须高于当前价格
            if take_profit <= current_price:
                take_profit = current_price * 1.02  # 设置比当前价高2%的止盈
        else:  # short
            # 做空：止盈必须低于入场价，止损必须高于入场价
            if take_profit >= price:
                take_profit = price * 0.94  # 确保止盈低于入场价
            if stop_loss <= price:
                stop_loss = price * 1.05  # 确保止损高于入场价
            # 额外验证：止盈必须低于当前价格
            if take_profit >= current_price:
                take_profit = current_price * 0.98  # 设置比当前价低2%的止盈
        
        log_message("DEBUG", f"{symbol} {signal} 止盈止损计算: 入场价={price:.4f}, 当前价={current_price:.4f}, 止损={stop_loss:.4f}, 止盈={take_profit:.4f}")
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
        
    except Exception as e:
        log_message("ERROR", f"计算止损止盈失败: {str(e)}")
        return None

def sync_exchange_positions():
    """同步交易所持仓，统一按MACD策略管理"""
    try:
        log_message("INFO", "正在同步交易所持仓...")
        
        positions = exchange.fetch_positions()
        active_positions = [pos for pos in positions if float(pos['contracts']) != 0]
        
        for position in active_positions:
            symbol = position['symbol']
            size = float(position['contracts'])
            side = 'long' if size > 0 else 'short'
            # entry price fallback: use entryPrice or avgPrice or last ticker
            entry_price = float(position.get('entryPrice') or position.get('avgPrice') or exchange.fetch_ticker(symbol)['last'])

            # 写入本地持仓跟踪
            position_tracker['positions'][symbol] = {
                'symbol': symbol,
                'side': side,
                'size': size,
                'entry_price': entry_price,
                'timestamp': datetime.now(timezone(timedelta(hours=8)))
            }

            # 同步后立即设置止盈止损
            try:
                setup_missing_stop_orders(position_tracker['positions'][symbol], symbol)
            except Exception as e:
                log_message("WARNING", f"同步设置止盈止损失败 {symbol}: {e}")
    except Exception as e:
        log_message("ERROR", f"同步交易所持仓失败: {str(e)}")
    return

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
            'timestamp': datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S'),
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
    """检查持仓状态，基于MACD金叉/死叉和K线阴阳线确认平仓"""
    try:
        for symbol in list(position_tracker['positions'].keys()):
            ohlcv = get_klines(symbol, '30m', limit=100)
            if not ohlcv:
                continue
            
            df = process_klines(ohlcv)
            if df is None or len(df) < 2:
                continue
            
            # 获取当前时间戳
            import time
            current_timestamp = int(time.time() * 1000)
            
            # 获取当前K线数据（MACD交叉的这根K线）
            current_kline = ohlcv[-1]
            current_kline_start = current_kline[0]
            current_kline_end = current_kline_start + 30 * 60 * 1000  # 30分钟K线结束时间
            
            # 获取当前K线的技术指标
            current_macd = df['MACD'].iloc[-1]
            current_signal = df['MACD_SIGNAL'].iloc[-1]
            current_adx = df['ADX'].iloc[-1]
            current_open = df['open'].iloc[-1]
            current_close = df['close'].iloc[-1]
            
            # 获取前一K线数据（用于判断交叉）
            prev_macd = df['MACD'].iloc[-2]
            prev_signal = df['MACD_SIGNAL'].iloc[-2]
            prev_adx = df['ADX'].iloc[-2]
            prev_open = df['open'].iloc[-2]
            prev_close = df['close'].iloc[-2]
            
            position = position_tracker['positions'][symbol]
            
            # 检查MACD金叉死叉（使用当前K线和前一K线判断交叉）
            golden_cross = prev_macd <= prev_signal and current_macd > current_signal
            death_cross = prev_macd >= prev_signal and current_macd < current_signal
            
            # 检查当前K线的阴阳线（收盘价确认）
            is_bullish = current_close > current_open  # 阳线：收盘价大于开盘价
            is_bearish = current_close < current_open  # 阴线：收盘价小于开盘价
            
            should_close = False
            close_reason = ""
            
            # 平仓条件：使用MACD交叉的这根K线收盘时确认
            if position['side'] == 'long' and death_cross and is_bearish and current_adx > ADX_TREND_THRESHOLD:
                should_close = True
                close_reason = f"MACD死叉+阴线确认平仓 (ADX={current_adx:.2f})"
                log_message("DEBUG", f"{symbol} 多头平仓条件满足: 死叉确认, 阴线确认, ADX趋势")
                
            elif position['side'] == 'short' and golden_cross and is_bullish and current_adx > ADX_TREND_THRESHOLD:
                should_close = True
                close_reason = f"MACD金叉+阳线确认平仓 (ADX={current_adx:.2f})"
                log_message("DEBUG", f"{symbol} 空头平仓条件满足: 金叉确认, 阳线确认, ADX趋势")
            
            # 检查当前K线是否已收盘
            kline_completed = current_timestamp >= current_kline_end
            
            if should_close:
                log_message("DEBUG", f"{symbol} 平仓检查: 当前K线开始时间={current_kline_start}, 结束时间={current_kline_end}, 当前时间={current_timestamp}, K线完成={kline_completed}")
                
                if kline_completed:
                    # 当前K线已收盘，执行平仓
                    close_position(symbol, close_reason)
                    log_message("INFO", f"{symbol} MACD交叉K线已收盘，执行平仓: {close_reason}")
                else:
                    # 当前K线未收盘，等待收盘
                    time_remaining = (current_kline_end - current_timestamp) / 1000
                    log_message("DEBUG", f"{symbol} 平仓条件满足但等待当前K线收盘 (还需等待{time_remaining:.0f}秒)")
            else:
                # 记录为什么没有满足平仓条件
                if position['side'] == 'long':
                    log_message("DEBUG", f"{symbol} 多头持仓未满足平仓条件: 死叉={death_cross}, 阴线={is_bearish}, ADX={current_adx:.2f}>{ADX_TREND_THRESHOLD}")
                elif position['side'] == 'short':
                    log_message("DEBUG", f"{symbol} 空头持仓未满足平仓条件: 金叉={golden_cross}, 阳线={is_bullish}, ADX={current_adx:.2f}>{ADX_TREND_THRESHOLD}")
                
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

        # 取消该交易对的所有未成交订单，避免与平仓冲突
        try:
            open_orders = exchange.fetch_open_orders(symbol)
            for o in open_orders:
                try:
                    exchange.cancel_order(o['id'], symbol)
                except:
                    pass
        except Exception as e:
            log_message("WARNING", f"取消未成交订单时出错 {symbol}: {e}")

        # 根据账户持仓模式构建下单参数
        params = {'tdMode': TD_MODE, 'reduceOnly': True}
        if ACCOUNT_MODE == 'hedge':
            params['posSide'] = 'long' if position['side'] == 'long' else 'short'
        
        order = exchange.createOrder(
            symbol,
            'market',
            side,
            abs(size),
            None,
            params
        )
        
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
            
            # 清理订单跟踪信息，防止重复设置止盈止损
            if symbol in order_tracking:
                del order_tracking[symbol]
                log_message("DEBUG", f"清理 {symbol} 订单跟踪信息")
            
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
                
                # 检查pending信号是否已经可以确认
                confirmed_signals = check_pending_signals()
                for symbol, signal in confirmed_signals:
                    log_message("INFO", f"执行pending信号确认的交易: {symbol} {signal['side']} @ {signal['price']:.4f}")
                    if execute_trade(symbol, signal, signal['signal_strength']):
                        position_tracker['daily_stats']['trades_count'] += 1
                        time.sleep(2)
                        if symbol in position_tracker['positions']:
                            setup_missing_stop_orders(position_tracker['positions'][symbol], symbol)
                
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

# =================================
# 附加：6个策略函数 + 可选 SYMBOLS 覆盖（不改动现有逻辑）
# 使用说明：
# - 这些函数供你的回测/外部加载器使用；当前 main.py 的回测仍按既有流程运行。
# - 若需用下面的 SYMBOLS 覆盖现有 SYMBOLS，请在环境变量设置 USE_APPENDED_SYMBOLS=1。
# =================================

from typing import Dict, Any, Optional

def _bool_series(s):
    # 将任意布尔条件安全地转换为布尔Series并与索引对齐
    return pd.Series(s, index=s.index).fillna(False).astype(bool)

def generate_signals_trend_ema_adx_rsi(df: pd.DataFrame, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, pd.Series]:
    cfg = cfg or {}
    ema_fast = int(cfg.get("ema_fast", 20))
    ema_slow = int(cfg.get("ema_slow", 50))
    adx_thr = float(cfg.get("adx_thr", 25))
    rsi_len = int(cfg.get("rsi_len", 14))
    rsi_os = float(cfg.get("rsi_os", 35))
    rsi_ob = float(cfg.get("rsi_ob", 65))

    ema_f = df["close"].ewm(span=ema_fast, adjust=False).mean()
    ema_s = df["close"].ewm(span=ema_slow, adjust=False).mean()
    adx_series = calculate_adx(df["high"], df["low"], df["close"], period=cfg.get("adx_period", 14))
    rsi_series = calculate_rsi(df["close"], period=rsi_len)

    long_entry = (ema_f > ema_s) & (adx_series > adx_thr) & (rsi_series > rsi_os)
    long_exit  = (ema_f < ema_s) | (rsi_series > rsi_ob)
    short_entry = (ema_f < ema_s) & (adx_series > adx_thr) & (rsi_series < (100 - rsi_os))
    short_exit  = (ema_f > ema_s) | (rsi_series < (100 - rsi_ob))
    return {
        "long_entry": _bool_series(long_entry),
        "long_exit": _bool_series(long_exit),
        "short_entry": _bool_series(short_entry),
        "short_exit": _bool_series(short_exit),
    }

def generate_signals_macd_rsi(df: pd.DataFrame, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, pd.Series]:
    cfg = cfg or {}
    fast = int(cfg.get("fast", 12)); slow = int(cfg.get("slow", 26)); signal = int(cfg.get("signal", 9))
    rsi_len = int(cfg.get("rsi_len", 14))
    rsi_os = float(cfg.get("rsi_os", 35)); rsi_ob = float(cfg.get("rsi_ob", 65))
    macd_line, signal_line, _ = calculate_macd(df["close"], fast=fast, slow=slow, signal=signal)
    rsi_series = calculate_rsi(df["close"], period=rsi_len)

    cross_up = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
    cross_dn = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))

    long_entry = cross_up & (rsi_series > rsi_os)
    long_exit  = cross_dn | (rsi_series > rsi_ob)
    short_entry = cross_dn & (rsi_series < (100 - rsi_os))
    short_exit  = cross_up | (rsi_series < (100 - rsi_ob))
    return {
        "long_entry": _bool_series(long_entry),
        "long_exit": _bool_series(long_exit),
        "short_entry": _bool_series(short_entry),
        "short_exit": _bool_series(short_exit),
    }

def generate_signals_bb_rsi(df: pd.DataFrame, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, pd.Series]:
    cfg = cfg or {}
    period = int(cfg.get("period", 20)); std_k = float(cfg.get("std_k", 2.0))
    rsi_len = int(cfg.get("rsi_len", 14))
    rsi_os = float(cfg.get("rsi_os", 30)); rsi_ob = float(cfg.get("rsi_ob", 70))
    upper, mid, lower = calculate_bollinger_bands(df["close"], period=period, std_dev=std_k)
    rsi_series = calculate_rsi(df["close"], period=rsi_len)

    long_entry = (df["close"] <= lower) & (rsi_series <= rsi_os)
    long_exit  = (df["close"] >= mid) | (rsi_series >= rsi_ob)
    short_entry = (df["close"] >= upper) & (rsi_series >= rsi_ob)
    short_exit  = (df["close"] <= mid) | (rsi_series <= rsi_os)
    return {
        "long_entry": _bool_series(long_entry),
        "long_exit": _bool_series(long_exit),
        "short_entry": _bool_series(short_entry),
        "short_exit": _bool_series(short_exit),
    }

def generate_signals_kdj_ma_volume(df: pd.DataFrame, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, pd.Series]:
    cfg = cfg or {}
    k_period = int(cfg.get("k_period", 9)); d_period = int(cfg.get("d_period", 3))
    ma_fast = int(cfg.get("ma_fast", 10)); ma_slow = int(cfg.get("ma_slow", 30))
    vol_len = int(cfg.get("vol_len", 20)); vol_k = float(cfg.get("vol_k", 1.2))
    k, d = calculate_stochastic(df["high"], df["low"], df["close"], k_period=k_period, d_period=d_period)
    ma_f = df["close"].rolling(ma_fast).mean()
    ma_s = df["close"].rolling(ma_slow).mean()
    vol_ma = df["volume"].rolling(vol_len).mean()

    cross_up = (k > d) & (k.shift(1) <= d.shift(1))
    cross_dn = (k < d) & (k.shift(1) >= d.shift(1))

    long_entry = cross_up & (ma_f > ma_s) & (df["volume"] > vol_ma * vol_k)
    long_exit  = cross_dn | (ma_f < ma_s)
    short_entry = cross_dn & (ma_f < ma_s) & (df["volume"] > vol_ma * vol_k)
    short_exit  = cross_up | (ma_f > ma_s)
    return {
        "long_entry": _bool_series(long_entry),
        "long_exit": _bool_series(long_exit),
        "short_entry": _bool_series(short_entry),
        "short_exit": _bool_series(short_exit),
    }

def generate_signals_atr_breakout(df: pd.DataFrame, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, pd.Series]:
    cfg = cfg or {}
    atr_p = int(cfg.get("atr_p", 14)); sma_p = int(cfg.get("sma_p", 20)); k = float(cfg.get("k", 1.5))
    atr_series = calculate_atr(df["high"], df["low"], df["close"], period=atr_p)
    sma = df["close"].rolling(sma_p).mean()

    long_entry = df["close"] > (sma + k * atr_series)
    long_exit  = df["close"] < sma
    short_entry = df["close"] < (sma - k * atr_series)
    short_exit  = df["close"] > sma
    return {
        "long_entry": _bool_series(long_entry),
        "long_exit": _bool_series(long_exit),
        "short_entry": _bool_series(short_entry),
        "short_exit": _bool_series(short_exit),
    }

def generate_signals_trend_pullback_bb_rsi(df: pd.DataFrame, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, pd.Series]:
    cfg = cfg or {}
    ema_fast = int(cfg.get("ema_fast", 20)); ema_slow = int(cfg.get("ema_slow", 50))
    bb_p = int(cfg.get("bb_p", 20)); bb_k = float(cfg.get("bb_k", 2.0))
    rsi_len = int(cfg.get("rsi_len", 14)); rsi_pullback = float(cfg.get("rsi_pullback", 45)); rsi_rebound = float(cfg.get("rsi_rebound", 55))
    ema_f = df["close"].ewm(span=ema_fast, adjust=False).mean()
    ema_s = df["close"].ewm(span=ema_slow, adjust=False).mean()
    upper, mid, lower = calculate_bollinger_bands(df["close"], period=bb_p, std_dev=bb_k)
    rsi_series = calculate_rsi(df["close"], period=rsi_len)

    long_entry = (ema_f > ema_s) & (df["close"] <= mid) & (rsi_series <= rsi_pullback)
    long_exit  = (rsi_series >= rsi_rebound) | (df["close"] >= upper)
    short_entry = (ema_f < ema_s) & (df["close"] >= mid) & (rsi_series >= (100 - rsi_pullback))
    short_exit  = (rsi_series <= (100 - rsi_rebound)) | (df["close"] <= lower)
    return {
        "long_entry": _bool_series(long_entry),
        "long_exit": _bool_series(long_exit),
        "short_entry": _bool_series(short_entry),
        "short_exit": _bool_series(short_exit),
    }

# 可选：覆盖 SYMBOLS（仅当设置 USE_APPENDED_SYMBOLS=1 时生效）
if os.getenv("USE_APPENDED_SYMBOLS", "0") == "1":
    SYMBOLS = [
        'BTC-USDT-SWAP', 'ETH-USDT-SWAP', 'SOL-USDT-SWAP', 'BNB-USDT-SWAP',
        'XRP-USDT-SWAP', 'DOGE-USDT-SWAP', 'ADA-USDT-SWAP', 'AVAX-USDT-SWAP',
        'SHIB-USDT-SWAP', 'DOT-USDT-SWAP', 'FIL-USDT-SWAP', 'ZRO-USDT-SWAP',
        'WIF-USDT-SWAP', 'WLD-USDT-SWAP'
    ]

# =================================
# 多策略回测模块（追加，不改动原有逻辑）
# - 依赖上面已追加的6个 generate_signals_xxx 函数
# - 在启动流程中自动执行：打印每个策略名的结果并保存独立报告
# =================================
from typing import Callable, Tuple

def generate_signals_combined_high_winrate_profit(df: pd.DataFrame, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, pd.Series]:
    """组合策略：趋势回调_布林带_RSI（高胜率）+ 趋势EMA_ADX_RSI（高盈利率）"""
    cfg = cfg or {}
    
    # 获取两个策略的信号
    signals_pullback = generate_signals_trend_pullback_bb_rsi(df, cfg)
    signals_trend = generate_signals_trend_ema_adx_rsi(df, cfg)
    
    # 信号叠加：两个策略同时满足条件才开仓
    long_entry = signals_pullback["long_entry"] & signals_trend["long_entry"]
    long_exit = signals_pullback["long_exit"] | signals_trend["long_exit"]
    short_entry = signals_pullback["short_entry"] & signals_trend["short_entry"]
    short_exit = signals_pullback["short_exit"] | signals_trend["short_exit"]
    
    return {
        "long_entry": _bool_series(long_entry),
        "long_exit": _bool_series(long_exit),
        "short_entry": _bool_series(short_entry),
        "short_exit": _bool_series(short_exit),
    }

# 高胜率指标（趋势回调_布林带_RSI）的搭配组合
def generate_signals_pullback_macd_rsi(df: pd.DataFrame, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, pd.Series]:
    """组合策略：趋势回调_布林带_RSI + MACD_RSI"""
    cfg = cfg or {}
    signals_pullback = generate_signals_trend_pullback_bb_rsi(df, cfg)
    signals_macd = generate_signals_macd_rsi(df, cfg)
    long_entry = signals_pullback["long_entry"] & signals_macd["long_entry"]
    long_exit = signals_pullback["long_exit"] | signals_macd["long_exit"]
    short_entry = signals_pullback["short_entry"] & signals_macd["short_entry"]
    short_exit = signals_pullback["short_exit"] | signals_macd["short_exit"]
    return {
        "long_entry": _bool_series(long_entry),
        "long_exit": _bool_series(long_exit),
        "short_entry": _bool_series(short_entry),
        "short_exit": _bool_series(short_exit),
    }

def generate_signals_pullback_bb_rsi(df: pd.DataFrame, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, pd.Series]:
    """组合策略：趋势回调_布林带_RSI + 布林带_RSI"""
    cfg = cfg or {}
    signals_pullback = generate_signals_trend_pullback_bb_rsi(df, cfg)
    signals_bb = generate_signals_bb_rsi(df, cfg)
    long_entry = signals_pullback["long_entry"] & signals_bb["long_entry"]
    long_exit = signals_pullback["long_exit"] | signals_bb["long_exit"]
    short_entry = signals_pullback["short_entry"] & signals_bb["short_entry"]
    short_exit = signals_pullback["short_exit"] | signals_bb["short_exit"]
    return {
        "long_entry": _bool_series(long_entry),
        "long_exit": _bool_series(long_exit),
        "short_entry": _bool_series(short_entry),
        "short_exit": _bool_series(short_exit),
    }

def generate_signals_pullback_kdj_ma_volume(df: pd.DataFrame, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, pd.Series]:
    """组合策略：趋势回调_布林带_RSI + KDJ_MA_成交量"""
    cfg = cfg or {}
    signals_pullback = generate_signals_trend_pullback_bb_rsi(df, cfg)
    signals_kdj = generate_signals_kdj_ma_volume(df, cfg)
    long_entry = signals_pullback["long_entry"] & signals_kdj["long_entry"]
    long_exit = signals_pullback["long_exit"] | signals_kdj["long_exit"]
    short_entry = signals_pullback["short_entry"] & signals_kdj["short_entry"]
    short_exit = signals_pullback["short_exit"] | signals_kdj["short_exit"]
    return {
        "long_entry": _bool_series(long_entry),
        "long_exit": _bool_series(long_exit),
        "short_entry": _bool_series(short_entry),
        "short_exit": _bool_series(short_exit),
    }

def generate_signals_pullback_atr_breakout(df: pd.DataFrame, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, pd.Series]:
    """组合策略：趋势回调_布林带_RSI + ATR突破"""
    cfg = cfg or {}
    signals_pullback = generate_signals_trend_pullback_bb_rsi(df, cfg)
    signals_atr = generate_signals_atr_breakout(df, cfg)
    long_entry = signals_pullback["long_entry"] & signals_atr["long_entry"]
    long_exit = signals_pullback["long_exit"] | signals_atr["long_exit"]
    short_entry = signals_pullback["short_entry"] & signals_atr["short_entry"]
    short_exit = signals_pullback["short_exit"] | signals_atr["short_exit"]
    return {
        "long_entry": _bool_series(long_entry),
        "long_exit": _bool_series(long_exit),
        "short_entry": _bool_series(short_entry),
        "short_exit": _bool_series(short_exit),
    }

def generate_signals_pullback_trend_macd(df: pd.DataFrame, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, pd.Series]:
    """组合策略：趋势回调_布林带_RSI + 趋势EMA_ADX_RSI + MACD_RSI"""
    cfg = cfg or {}
    signals_pullback = generate_signals_trend_pullback_bb_rsi(df, cfg)
    signals_trend = generate_signals_trend_ema_adx_rsi(df, cfg)
    signals_macd = generate_signals_macd_rsi(df, cfg)
    long_entry = signals_pullback["long_entry"] & signals_trend["long_entry"] & signals_macd["long_entry"]
    long_exit = signals_pullback["long_exit"] | signals_trend["long_exit"] | signals_macd["long_exit"]
    short_entry = signals_pullback["short_entry"] & signals_trend["short_entry"] & signals_macd["short_entry"]
    short_exit = signals_pullback["short_exit"] | signals_trend["short_exit"] | signals_macd["short_exit"]
    return {
        "long_entry": _bool_series(long_entry),
        "long_exit": _bool_series(long_exit),
        "short_entry": _bool_series(short_entry),
        "short_exit": _bool_series(short_exit),
    }

def generate_signals_pullback_trend_bb(df: pd.DataFrame, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, pd.Series]:
    """组合策略：趋势回调_布林带_RSI + 趋势EMA_ADX_RSI + 布林带_RSI"""
    cfg = cfg or {}
    signals_pullback = generate_signals_trend_pullback_bb_rsi(df, cfg)
    signals_trend = generate_signals_trend_ema_adx_rsi(df, cfg)
    signals_bb = generate_signals_bb_rsi(df, cfg)
    long_entry = signals_pullback["long_entry"] & signals_trend["long_entry"] & signals_bb["long_entry"]
    long_exit = signals_pullback["long_exit"] | signals_trend["long_exit"] | signals_bb["long_exit"]
    short_entry = signals_pullback["short_entry"] & signals_trend["short_entry"] & signals_bb["short_entry"]
    short_exit = signals_pullback["short_exit"] | signals_trend["short_exit"] | signals_bb["short_exit"]
    return {
        "long_entry": _bool_series(long_entry),
        "long_exit": _bool_series(long_exit),
        "short_entry": _bool_series(short_entry),
        "short_exit": _bool_series(short_exit),
    }

def generate_signals_pullback_trend_kdj(df: pd.DataFrame, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, pd.Series]:
    """组合策略：趋势回调_布林带_RSI + 趋势EMA_ADX_RSI + KDJ_MA_成交量"""
    cfg = cfg or {}
    signals_pullback = generate_signals_trend_pullback_bb_rsi(df, cfg)
    signals_trend = generate_signals_trend_ema_adx_rsi(df, cfg)
    signals_kdj = generate_signals_kdj_ma_volume(df, cfg)
    long_entry = signals_pullback["long_entry"] & signals_trend["long_entry"] & signals_kdj["long_entry"]
    long_exit = signals_pullback["long_exit"] | signals_trend["long_exit"] | signals_kdj["long_exit"]
    short_entry = signals_pullback["short_entry"] & signals_trend["short_entry"] & signals_kdj["short_entry"]
    short_exit = signals_pullback["short_exit"] | signals_trend["short_exit"] | signals_kdj["short_exit"]
    return {
        "long_entry": _bool_series(long_entry),
        "long_exit": _bool_series(long_exit),
        "short_entry": _bool_series(short_entry),
        "short_exit": _bool_series(short_exit),
    }

def generate_signals_pullback_trend_atr(df: pd.DataFrame, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, pd.Series]:
    """组合策略：趋势回调_布林带_RSI + 趋势EMA_ADX_RSI + ATR突破"""
    cfg = cfg or {}
    signals_pullback = generate_signals_trend_pullback_bb_rsi(df, cfg)
    signals_trend = generate_signals_trend_ema_adx_rsi(df, cfg)
    signals_atr = generate_signals_atr_breakout(df, cfg)
    long_entry = signals_pullback["long_entry"] & signals_trend["long_entry"] & signals_atr["long_entry"]
    long_exit = signals_pullback["long_exit"] | signals_trend["long_exit"] | signals_atr["long_exit"]
    short_entry = signals_pullback["short_entry"] & signals_trend["short_entry"] & signals_atr["short_entry"]
    short_exit = signals_pullback["short_exit"] | signals_trend["short_exit"] | signals_atr["short_exit"]
    return {
        "long_entry": _bool_series(long_entry),
        "long_exit": _bool_series(long_exit),
        "short_entry": _bool_series(short_entry),
        "short_exit": _bool_series(short_exit),
    }

# 高盈利指标（趋势EMA_ADX_RSI）的搭配组合
def generate_signals_trend_macd_rsi(df: pd.DataFrame, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, pd.Series]:
    """组合策略：趋势EMA_ADX_RSI + MACD_RSI"""
    cfg = cfg or {}
    signals_trend = generate_signals_trend_ema_adx_rsi(df, cfg)
    signals_macd = generate_signals_macd_rsi(df, cfg)
    long_entry = signals_trend["long_entry"] & signals_macd["long_entry"]
    long_exit = signals_trend["long_exit"] | signals_macd["long_exit"]
    short_entry = signals_trend["short_entry"] & signals_macd["short_entry"]
    short_exit = signals_trend["short_exit"] | signals_macd["short_exit"]
    return {
        "long_entry": _bool_series(long_entry),
        "long_exit": _bool_series(long_exit),
        "short_entry": _bool_series(short_entry),
        "short_exit": _bool_series(short_exit),
    }

def generate_signals_trend_bb_rsi(df: pd.DataFrame, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, pd.Series]:
    """组合策略：趋势EMA_ADX_RSI + 布林带_RSI"""
    cfg = cfg or {}
    signals_trend = generate_signals_trend_ema_adx_rsi(df, cfg)
    signals_bb = generate_signals_bb_rsi(df, cfg)
    long_entry = signals_trend["long_entry"] & signals_bb["long_entry"]
    long_exit = signals_trend["long_exit"] | signals_bb["long_exit"]
    short_entry = signals_trend["short_entry"] & signals_bb["short_entry"]
    short_exit = signals_trend["short_exit"] | signals_bb["short_exit"]
    return {
        "long_entry": _bool_series(long_entry),
        "long_exit": _bool_series(long_exit),
        "short_entry": _bool_series(short_entry),
        "short_exit": _bool_series(short_exit),
    }

def generate_signals_trend_kdj_ma_volume(df: pd.DataFrame, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, pd.Series]:
    """组合策略：趋势EMA_ADX_RSI + KDJ_MA_成交量"""
    cfg = cfg or {}
    signals_trend = generate_signals_trend_ema_adx_rsi(df, cfg)
    signals_kdj = generate_signals_kdj_ma_volume(df, cfg)
    long_entry = signals_trend["long_entry"] & signals_kdj["long_entry"]
    long_exit = signals_trend["long_exit"] | signals_kdj["long_exit"]
    short_entry = signals_trend["short_entry"] & signals_kdj["short_entry"]
    short_exit = signals_trend["short_exit"] | signals_kdj["short_exit"]
    return {
        "long_entry": _bool_series(long_entry),
        "long_exit": _bool_series(long_exit),
        "short_entry": _bool_series(short_entry),
        "short_exit": _bool_series(short_exit),
    }

def generate_signals_trend_atr_breakout(df: pd.DataFrame, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, pd.Series]:
    """组合策略：趋势EMA_ADX_RSI + ATR突破"""
    cfg = cfg or {}
    signals_trend = generate_signals_trend_ema_adx_rsi(df, cfg)
    signals_atr = generate_signals_atr_breakout(df, cfg)
    long_entry = signals_trend["long_entry"] & signals_atr["long_entry"]
    long_exit = signals_trend["long_exit"] | signals_atr["long_exit"]
    short_entry = signals_trend["short_entry"] & signals_atr["short_entry"]
    short_exit = signals_trend["short_exit"] | signals_atr["short_exit"]
    return {
        "long_entry": _bool_series(long_entry),
        "long_exit": _bool_series(long_exit),
        "short_entry": _bool_series(short_entry),
        "short_exit": _bool_series(short_exit),
    }

def generate_signals_trend_pullback_macd(df: pd.DataFrame, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, pd.Series]:
    """组合策略：趋势EMA_ADX_RSI + 趋势回调_布林带_RSI + MACD_RSI"""
    cfg = cfg or {}
    signals_trend = generate_signals_trend_ema_adx_rsi(df, cfg)
    signals_pullback = generate_signals_trend_pullback_bb_rsi(df, cfg)
    signals_macd = generate_signals_macd_rsi(df, cfg)
    long_entry = signals_trend["long_entry"] & signals_pullback["long_entry"] & signals_macd["long_entry"]
    long_exit = signals_trend["long_exit"] | signals_pullback["long_exit"] | signals_macd["long_exit"]
    short_entry = signals_trend["short_entry"] & signals_pullback["short_entry"] & signals_macd["short_entry"]
    short_exit = signals_trend["short_exit"] | signals_pullback["short_exit"] | signals_macd["short_exit"]
    return {
        "long_entry": _bool_series(long_entry),
        "long_exit": _bool_series(long_exit),
        "short_entry": _bool_series(short_entry),
        "short_exit": _bool_series(short_exit),
    }

def generate_signals_trend_pullback_bb(df: pd.DataFrame, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, pd.Series]:
    """组合策略：趋势EMA_ADX_RSI + 趋势回调_布林带_RSI + 布林带_RSI"""
    cfg = cfg or {}
    signals_trend = generate_signals_trend_ema_adx_rsi(df, cfg)
    signals_pullback = generate_signals_trend_pullback_bb_rsi(df, cfg)
    signals_bb = generate_signals_bb_rsi(df, cfg)
    long_entry = signals_trend["long_entry"] & signals_pullback["long_entry"] & signals_bb["long_entry"]
    long_exit = signals_trend["long_exit"] | signals_pullback["long_exit"] | signals_bb["long_exit"]
    short_entry = signals_trend["short_entry"] & signals_pullback["short_entry"] & signals_bb["short_entry"]
    short_exit = signals_trend["short_exit"] | signals_pullback["short_exit"] | signals_bb["short_exit"]
    return {
        "long_entry": _bool_series(long_entry),
        "long_exit": _bool_series(long_exit),
        "short_entry": _bool_series(short_entry),
        "short_exit": _bool_series(short_exit),
    }

def generate_signals_trend_pullback_kdj(df: pd.DataFrame, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, pd.Series]:
    """组合策略：趋势EMA_ADX_RSI + 趋势回调_布林带_RSI + KDJ_MA_成交量"""
    cfg = cfg or {}
    signals_trend = generate_signals_trend_ema_adx_rsi(df, cfg)
    signals_pullback = generate_signals_trend_pullback_bb_rsi(df, cfg)
    signals_kdj = generate_signals_kdj_ma_volume(df, cfg)
    long_entry = signals_trend["long_entry"] & signals_pullback["long_entry"] & signals_kdj["long_entry"]
    long_exit = signals_trend["long_exit"] | signals_pullback["long_exit"] | signals_kdj["long_exit"]
    short_entry = signals_trend["short_entry"] & signals_pullback["short_entry"] & signals_kdj["short_entry"]
    short_exit = signals_trend["short_exit"] | signals_pullback["short_exit"] | signals_kdj["short_exit"]
    return {
        "long_entry": _bool_series(long_entry),
        "long_exit": _bool_series(long_exit),
        "short_entry": _bool_series(short_entry),
        "short_exit": _bool_series(short_exit),
    }

def generate_signals_trend_pullback_atr(df: pd.DataFrame, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, pd.Series]:
    """组合策略：趋势EMA_ADX_RSI + 趋势回调_布林带_RSI + ATR突破"""
    cfg = cfg or {}
    signals_trend = generate_signals_trend_ema_adx_rsi(df, cfg)
    signals_pullback = generate_signals_trend_pullback_bb_rsi(df, cfg)
    signals_atr = generate_signals_atr_breakout(df, cfg)
    long_entry = signals_trend["long_entry"] & signals_pullback["long_entry"] & signals_atr["long_entry"]
    long_exit = signals_trend["long_exit"] | signals_pullback["long_exit"] | signals_atr["long_exit"]
    short_entry = signals_trend["short_entry"] & signals_pullback["short_entry"] & signals_atr["short_entry"]
    short_exit = signals_trend["short_exit"] | signals_pullback["short_exit"] | signals_atr["short_exit"]
    return {
        "long_entry": _bool_series(long_entry),
        "long_exit": _bool_series(long_exit),
        "short_entry": _bool_series(short_entry),
        "short_exit": _bool_series(short_exit),
    }

def _strategy_registry() -> list[Tuple[str, str, Callable]]:
    """返回策略名称、指标描述和策略函数"""
    return [
        # 原始单指标策略
        ("趋势EMA_ADX_RSI", "EMA+ADX+RSI趋势跟踪", generate_signals_trend_ema_adx_rsi),
        ("MACD_RSI", "MACD+RSI动量反转", generate_signals_macd_rsi),
        ("布林带_RSI", "布林带+RSI超买超卖", generate_signals_bb_rsi),
        ("KDJ_MA_成交量", "KDJ+MA+成交量突破", generate_signals_kdj_ma_volume),
        ("ATR突破", "ATR+威廉指标+动量", generate_signals_atr_breakout),
        ("趋势回调_布林带_RSI", "趋势回调+布林带+RSI", generate_signals_trend_pullback_bb_rsi),
        
        # 高胜率指标（趋势回调_布林带_RSI）的搭配组合
        ("高胜率+高盈利率组合", "趋势回调_布林带_RSI+趋势EMA_ADX_RSI", generate_signals_combined_high_winrate_profit),
        ("高胜率+MACD_RSI", "趋势回调_布林带_RSI+MACD_RSI", generate_signals_pullback_macd_rsi),
        ("高胜率+布林带_RSI", "趋势回调_布林带_RSI+布林带_RSI", generate_signals_pullback_bb_rsi),
        ("高胜率+KDJ_MA_成交量", "趋势回调_布林带_RSI+KDJ_MA_成交量", generate_signals_pullback_kdj_ma_volume),
        ("高胜率+ATR突破", "趋势回调_布林带_RSI+ATR突破", generate_signals_pullback_atr_breakout),
        ("高胜率+趋势+MACD", "趋势回调_布林带_RSI+趋势EMA_ADX_RSI+MACD_RSI", generate_signals_pullback_trend_macd),
        ("高胜率+趋势+布林带", "趋势回调_布林带_RSI+趋势EMA_ADX_RSI+布林带_RSI", generate_signals_pullback_trend_bb),
        ("高胜率+趋势+KDJ", "趋势回调_布林带_RSI+趋势EMA_ADX_RSI+KDJ_MA_成交量", generate_signals_pullback_trend_kdj),
        ("高胜率+趋势+ATR", "趋势回调_布林带_RSI+趋势EMA_ADX_RSI+ATR突破", generate_signals_pullback_trend_atr),
        
        # 高盈利指标（趋势EMA_ADX_RSI）的搭配组合
        ("高盈利+MACD_RSI", "趋势EMA_ADX_RSI+MACD_RSI", generate_signals_trend_macd_rsi),
        ("高盈利+布林带_RSI", "趋势EMA_ADX_RSI+布林带_RSI", generate_signals_trend_bb_rsi),
        ("高盈利+KDJ_MA_成交量", "趋势EMA_ADX_RSI+KDJ_MA_成交量", generate_signals_trend_kdj_ma_volume),
        ("高盈利+ATR突破", "趋势EMA_ADX_RSI+ATR突破", generate_signals_trend_atr_breakout),
        ("高盈利+高胜率+MACD", "趋势EMA_ADX_RSI+趋势回调_布林带_RSI+MACD_RSI", generate_signals_trend_pullback_macd),
        ("高盈利+高胜率+布林带", "趋势EMA_ADX_RSI+趋势回调_布林带_RSI+布林带_RSI", generate_signals_trend_pullback_bb),
        ("高盈利+高胜率+KDJ", "趋势EMA_ADX_RSI+趋势回调_布林带_RSI+KDJ_MA_成交量", generate_signals_trend_pullback_kdj),
        ("高盈利+高胜率+ATR", "趋势EMA_ADX_RSI+趋势回调_布林带_RSI+ATR突破", generate_signals_trend_pullback_atr),
    ]

def backtest_with_signals(symbol: str, days: int, initial_balance: float, signals: dict) -> dict:
    """
    基于给定信号字典 {long_entry,long_exit,short_entry,short_exit} 的简单持仓回测。
    单仓位、双向可切换，资金按 initial_balance*0.8 参与，收益以点对点价差近似。
    """
    df = get_historical_data(symbol, days)
    if df is None or len(df) < 100:
        return None

    # 对齐信号索引
    le = signals.get("long_entry", pd.Series(False, index=df.index)).reindex(df.index, fill_value=False)
    lx = signals.get("long_exit", pd.Series(False, index=df.index)).reindex(df.index, fill_value=False)
    se = signals.get("short_entry", pd.Series(False, index=df.index)).reindex(df.index, fill_value=False)
    sx = signals.get("short_exit", pd.Series(False, index=df.index)).reindex(df.index, fill_value=False)

    position = None  # 'long' | 'short' | None
    entry_price = 0.0
    entry_time = None
    balance = initial_balance
    trades = []
    peak = balance
    max_dd = 0.0

    for i in range(1, len(df)):
        price = float(df["close"].iloc[i])
        ts = df["timestamp"].iloc[i]

        # 更新回撤
        if balance > peak:
            peak = balance
        dd = (peak - balance) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

        # 平仓逻辑优先
        if position == 'long' and (lx.iloc[i] or se.iloc[i]):
            pnl = (price - entry_price) / entry_price * balance * 0.8
            balance += pnl
            trades.append({"type":"close_long","price":price,"pnl":pnl,"timestamp":ts})
            position = None

        elif position == 'short' and (sx.iloc[i] or le.iloc[i]):
            pnl = (entry_price - price) / entry_price * balance * 0.8
            balance += pnl
            trades.append({"type":"close_short","price":price,"pnl":pnl,"timestamp":ts})
            position = None

        # 开仓
        if position is None:
            if le.iloc[i]:
                position = 'long'
                entry_price = price
                entry_time = ts
                trades.append({"type":"open_long","price":price,"timestamp":ts})
            elif se.iloc[i]:
                position = 'short'
                entry_price = price
                entry_time = ts
                trades.append({"type":"open_short","price":price,"timestamp":ts})

    closed = [t for t in trades if 'pnl' in t]
    total_trades = len(closed)
    win = len([t for t in closed if t['pnl'] > 0])
    loss = len([t for t in closed if t['pnl'] < 0])
    total_pnl = sum(t['pnl'] for t in closed)
    win_rate = (win / total_trades * 100) if total_trades > 0 else 0.0
    pf = abs(sum(t['pnl'] for t in closed if t['pnl'] > 0) / sum(t['pnl'] for t in closed if t['pnl'] < 0)) if loss > 0 else float('inf')

    return {
        "symbol": symbol,
        "days": days,
        "initial_balance": initial_balance,
        "final_balance": balance,
        "total_return": (balance - initial_balance) / initial_balance * 100,
        "total_trades": total_trades,
        "winning_trades": win,
        "losing_trades": loss,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "profit_factor": pf,
        "max_drawdown": max_dd,
    }

def run_multi_strategy_backtests(symbols=None, days_list=[7,14,30], initial_balance=10000):
    if symbols is None:
        symbols = SYMBOLS[:5]
    all_reports = []
    report_lines = ["=== 多策略回测报告（按指标组合分组） ===", ""]

    for strat_name, indicator_desc, strat_fn in _strategy_registry():
        report_lines.append(f"--- 策略: {strat_name} ---")
        report_lines.append(f"指标组合: {indicator_desc}")
        for days in days_list:
            day_results = []
            for sym in symbols:
                try:
                    # 准备 DataFrame 后生成策略信号
                    df = get_historical_data(sym, days)
                    if df is None or len(df) < 100:
                        log_message("WARNING", f"[{strat_name}] {sym} 历史数据不足，跳过")
                        continue
                    signals = strat_fn(df, {})
                    result = backtest_with_signals(sym, days, initial_balance, signals)
                    if result:
                        day_results.append(result)
                        log_message("INFO", f"[{strat_name}] {sym} {days}天: 胜率{result['win_rate']:.1f}% 收益{result['total_return']:.2f}% 交易{result['total_trades']}")
                except Exception as e:
                    log_message("ERROR", f"[{strat_name}] 回测 {sym} 失败: {e}")
                    continue

            if day_results:
                avg_win_rate = sum(r['win_rate'] for r in day_results)/len(day_results)
                avg_return = sum(r['total_return'] for r in day_results)/len(day_results)
                total_trades = sum(r['total_trades'] for r in day_results)
                report_lines.extend([
                    f"{days}天汇总: 标的数={len(day_results)} 平均胜率={avg_win_rate:.1f}% 平均收益={avg_return:.2f}% 总交易={total_trades}",
                    ""
                ])
                all_reports.extend([dict(r, strategy=strat_name, indicator_desc=indicator_desc) for r in day_results])

    # 保存报告
    try:
        timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
        out = f"backtest_results_multi_{timestamp}.txt"
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        log_message("SUCCESS", f"多策略回测结果已保存到: {out}")
    except Exception as e:
        log_message("WARNING", f"保存多策略回测报告失败: {e}")

    # 控制台打印汇总
    
    return all_reports



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
        log_message("SUCCESS", "MACD(6,16,9)策略交易系统启动成功")
        log_message("INFO", f"智能杠杆系统: BTC最大{MAX_LEVERAGE_BTC}x, ETH最大{MAX_LEVERAGE_ETH}x")
        log_message("INFO", f"交易对数量: {len(SYMBOLS)}")
        log_message("INFO", f"最大持仓数: {MAX_OPEN_POSITIONS}")
        
        # 同步交易所现有持仓
        sync_exchange_positions()
        
        # 启动交易循环
        trading_loop()
        
    except Exception as e:
        log_message("ERROR", f"启动交易系统失败: {str(e)}")
        traceback.print_exc()

# =================================
# 历史回测模块
# =================================

def get_historical_data(symbol, days=30):
    """获取历史数据用于回测"""
    try:
        # 计算需要获取的K线数量
        limit = days * 24 * 2  # 30分钟K线，每天48根
        
        ohlcv = exchange.fetch_ohlcv(symbol, '30m', limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        return df
    except Exception as e:
        log_message("ERROR", f"获取历史数据失败 {symbol}: {e}")
        return None

def backtest_strategy(symbol, days=7, initial_balance=10000):
    """策略回测 - EMA5/EMA10 交叉 + 固定止盈0.0058 + 交叉前一根K线止损；统一50x杠杆"""
    try:
        log_message("INFO", f"开始回测 {symbol}，回测天数: {days}，初始资金: {initial_balance}")
        df = get_historical_data(symbol, days)
        if df is None or len(df) < 50:
            log_message("WARNING", f"历史数据不足，无法回测 {symbol}")
            return None

        # 计算 EMA5/EMA10
        df['ema5'] = df['close'].ewm(span=5, adjust=False).mean()
        df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
        # 计算 BB(20,2) 中轨（仅用于过滤）
        df['bb_mid'] = df['close'].rolling(window=20, min_periods=20).mean()
        bb_threshold = 0.005  # 0.5%

        position = None  # 'long' | 'short' | None
        entry_price = 0.0
        entry_time = None
        entry_size = 0.0  # 以50x杠杆计算的合约数量
        trades = []
        balance = initial_balance
        peak_balance = initial_balance
        max_drawdown = 0.0
        trade_details = []

        for i in range(1, len(df)):
            cur = df.iloc[i]
            prev = df.iloc[i - 1]

            # 更新最大回撤
            if balance > peak_balance:
                peak_balance = balance
            dd = (peak_balance - balance) / peak_balance * 100 if peak_balance > 0 else 0
            if dd > max_drawdown:
                max_drawdown = dd

            # 计算交叉
            cross_up = (prev['ema5'] <= prev['ema10']) and (cur['ema5'] > cur['ema10'])
            cross_dn = (prev['ema5'] >= prev['ema10']) and (cur['ema5'] < cur['ema10'])

            # 平仓检查（基于止盈/止损）
            if position == 'long':
                tp_price = entry_price * (1 + 0.0058)
                sl_price = float(prev['low'])  # 交叉前一根K线的低点
                # 触发 TP 或 SL
                if cur['high'] >= tp_price or cur['low'] <= sl_price:
                    exit_price = tp_price if cur['high'] >= tp_price else sl_price
                    pnl = (exit_price - entry_price) * entry_size
                    balance += pnl
                    trades.append({'type': 'close_long', 'price': exit_price, 'pnl': pnl, 'timestamp': cur['timestamp']})
                    trade_details.append({
                        'symbol': symbol, 'side': 'close_long',
                        'entry_price': entry_price, 'exit_price': exit_price,
                        'pnl': pnl, 'pnl_percentage': (pnl / initial_balance * 100) if initial_balance > 0 else 0,
                        'entry_time': entry_time, 'exit_time': cur['timestamp'],
                        'duration': (cur['timestamp'] - entry_time).total_seconds() / 3600 if entry_time else 0
                    })
                    position = None
                    entry_price = 0.0
                    entry_size = 0.0
                    entry_time = None

            elif position == 'short':
                tp_price = entry_price * (1 - 0.0058)
                sl_price = float(prev['high'])  # 交叉前一根K线的高点
                if cur['low'] <= tp_price or cur['high'] >= sl_price:
                    exit_price = tp_price if cur['low'] <= tp_price else sl_price
                    pnl = (entry_price - exit_price) * entry_size
                    balance += pnl
                    trades.append({'type': 'close_short', 'price': exit_price, 'pnl': pnl, 'timestamp': cur['timestamp']})
                    trade_details.append({
                        'symbol': symbol, 'side': 'close_short',
                        'entry_price': entry_price, 'exit_price': exit_price,
                        'pnl': pnl, 'pnl_percentage': (pnl / initial_balance * 100) if initial_balance > 0 else 0,
                        'entry_time': entry_time, 'exit_time': cur['timestamp'],
                        'duration': (cur['timestamp'] - entry_time).total_seconds() / 3600 if entry_time else 0
                    })
                    position = None
                    entry_price = 0.0
                    entry_size = 0.0
                    entry_time = None

            # 入场（仅在无持仓时，根据交叉开仓 + BB(20,2)过滤）
            if position is None:
                if cross_up:
                    # 做多过滤：close 在 [bb_mid, bb_mid*(1+0.5%)]
                    if pd.notna(cur['bb_mid']):
                        mid = float(cur['bb_mid'])
                        upper_near = mid * (1.0 + bb_threshold)
                        if (cur['close'] >= mid) and (cur['close'] <= upper_near):
                            position = 'long'
                            entry_price = float(cur['close'])
                            entry_time = cur['timestamp']
                            entry_size = (balance * 0.8 * 50) / entry_price
                            trades.append({'type': 'open_long', 'price': entry_price, 'timestamp': entry_time})
                elif cross_dn:
                    # 做空过滤：close 在 [bb_mid*(1-0.5%), bb_mid]
                    if pd.notna(cur['bb_mid']):
                        mid = float(cur['bb_mid'])
                        lower_near = mid * (1.0 - bb_threshold)
                        if (cur['close'] <= mid) and (cur['close'] >= lower_near):
                            position = 'short'
                            entry_price = float(cur['close'])
                            entry_time = cur['timestamp']
                            entry_size = (balance * 0.8 * 50) / entry_price
                            trades.append({'type': 'open_short', 'price': entry_price, 'timestamp': entry_time})

        # 结算
        closed = [t for t in trades if 'pnl' in t]
        total_trades = len(closed)
        winning_trades = len([t for t in closed if t['pnl'] > 0])
        losing_trades = len([t for t in closed if t['pnl'] < 0])
        total_pnl = sum(t['pnl'] for t in closed)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        profit_factor = abs(
            sum(t['pnl'] for t in closed if t['pnl'] > 0) / sum(t['pnl'] for t in closed if t['pnl'] < 0)
        ) if losing_trades > 0 else float('inf')

        result = {
            'symbol': symbol,
            'days': days,
            'initial_balance': initial_balance,
            'final_balance': balance,
            'total_return': ((balance - initial_balance) / initial_balance * 100),
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'trade_details': trade_details,
            'trades': closed
        }
        log_message("INFO", f"回测完成 {symbol}: {total_trades}笔交易，胜率{win_rate:.1f}%，收益率{result['total_return']:.2f}%，最大回撤{max_drawdown:.2f}%")
        return result

    except Exception as e:
        log_message("ERROR", f"策略回测失败 {symbol}: {e}")
        return None

def generate_backtest_report(backtest_results):
    """生成详细的回测报告"""
    try:
        if not backtest_results:
            return "暂无回测数据"
        
        report_lines = ["=== 策略回测报告 ===", ""]
        
        # 汇总统计
        total_symbols = len(backtest_results)
        total_trades = sum([r['total_trades'] for r in backtest_results if r])
        total_return = sum([r['total_return'] for r in backtest_results if r])
        avg_win_rate = sum([r['win_rate'] for r in backtest_results if r]) / total_symbols if total_symbols > 0 else 0
        
        report_lines.extend([
            f"回测标的数量: {total_symbols}",
            f"总交易次数: {total_trades}",
            f"平均胜率: {avg_win_rate:.1f}%",
            f"总收益率: {total_return:.2f}%",
            ""
        ])
        
        # 各标的详细结果
        report_lines.append("=== 各标的回测结果 ===")
        for result in backtest_results:
            if result:
                report_lines.extend([
                    f"标的: {result['symbol']}",
                    f"  回测天数: {result['days']}天",
                    f"  交易次数: {result['total_trades']}",
                    f"  胜率: {result['win_rate']:.1f}%",
                    f"  收益率: {result['total_return']:.2f}%",
                    f"  最大回撤: {result['max_drawdown']:.2f}%",
                    f"  盈利因子: {result['profit_factor']:.2f}",
                    ""
                ])
        
        return "\n".join(report_lines)
        
    except Exception as e:
        log_message("ERROR", f"生成回测报告失败: {e}")
        return "生成回测报告失败"

def run_comprehensive_backtest(symbols=None, days_list=[7, 14, 30]):
    """运行全面的回测分析"""
    try:
        if symbols is None:
            # 自动从交易所获取热度前10的 USDT 合约（按24h成交量/信息字段排序），并追加 FIL/ZRO/WIF/WLD
            try:
                hot = []
                if exchange:
                    tickers = exchange.fetch_tickers()
                    # 过滤 USDT 合约
                    for sym, tk in tickers.items():
                        if (sym.endswith('-USDT-SWAP') or sym.endswith(':USDT')) and ('SWAP' in sym or ':' in sym):
                            vol = None
                            # ccxt标准字段或OKX info字段
                            vol = tk.get('quoteVolume') or tk.get('baseVolume')
                            if vol is None and isinstance(tk.get('info'), dict):
                                info = tk['info']
                                # OKX 可能提供 24h成交量（计价币数量）
                                vol = float(info.get('volCcy24h')) if info.get('volCcy24h') else None
                            if vol:
                                hot.append((sym, float(vol)))
                    hot.sort(key=lambda x: x[1], reverse=True)
                    top10 = [s for s, _ in hot[:10]]
                else:
                    top10 = SYMBOLS[:10]
                # 统一成 OKX 合约格式 XXX-USDT-SWAP
                def norm_sym(s):
                    return s.replace(':USDT', '-USDT-SWAP') if ':USDT' in s else (s if s.endswith('-USDT-SWAP') else s + '-USDT-SWAP')
                base = [norm_sym(s) for s in top10]
                extras = ['FIL-USDT-SWAP', 'ZRO-USDT-SWAP', 'WIF-USDT-SWAP', 'WLD-USDT-SWAP']
                # 去重保持顺序
                seen = set()
                symbols = []
                for s in base + extras:
                    if s not in seen:
                        symbols.append(s)
                        seen.add(s)
                log_message("INFO", f"自动获取回测标的: {symbols[:10]} + extras")
            except Exception as e:
                log_message("WARNING", f"自动获取热门标的失败，使用默认列表: {str(e)}")
                symbols = SYMBOLS[:10] + ['FIL-USDT-SWAP', 'ZRO-USDT-SWAP', 'WIF-USDT-SWAP', 'WLD-USDT-SWAP']
        
        all_results = []
        
        for days in days_list:
            log_message("INFO", f"开始{days}天回测分析...")
            day_results = []
            
            for symbol in symbols:
                result = backtest_strategy(symbol, days)
                if result:
                    day_results.append(result)
            
            if day_results:
                # 生成该时间周期的回测报告
                report = generate_backtest_report(day_results)
                log_message("INFO", f"{days}天回测结果:\n{report}")
                all_results.extend(day_results)
        
        # 生成综合报告
        final_report = generate_backtest_report(all_results)
        log_message("INFO", f"综合回测报告:\n{final_report}")
        
        return all_results
        
    except Exception as e:
        log_message("ERROR", f"全面回测失败: {e}")
        return None

# =================================
# 动态止盈止损管理
# =================================

def check_trailing_stop(symbol, position_info):
    """检查并更新动态止损（修复：区分趋势和震荡策略，添加K线收盘确认）"""
    try:
        if not position_info or position_info['size'] == 0:
            return False
        
        # 获取K线数据，检查当前K线是否已收盘
        ohlcv = exchange.fetch_ohlcv(symbol, '30m', limit=2)
        if len(ohlcv) < 2:
            return False
            
        current_kline = ohlcv[-1]
        prev_kline = ohlcv[-2]
        
        # 检查当前K线是否已收盘（当前时间是否超过K线结束时间）
        current_time = exchange.milliseconds()
        kline_end_time = current_kline[0] + 30 * 60 * 1000  # 30分钟K线结束时间
        
        # 如果当前K线还未收盘，使用前一K线的收盘价作为参考
        if current_time < kline_end_time:
            current_price = prev_kline[4]  # 使用前一K线的收盘价
            log_message("DEBUG", f"{symbol} 当前K线未收盘，使用前一K线收盘价: {current_price}")
        else:
            current_price = float(exchange.fetch_ticker(symbol)['last'])
        
        entry_price = float(position_info['entry_price'])
        side = position_info['side']
        size = float(position_info['size'])
        strategy_type = position_info.get('strategy_type', 'trend')  # 默认为趋势策略
        
        # 获取ATR值
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        atr = calculate_atr(df['high'], df['low'], df['close']).iloc[-1]
        
        # 根据策略类型设置不同的动态止损参数
        if strategy_type == 'oscillation':
            # 震荡策略：更早启用动态止损，更紧的保护
            if side == 'long':
                unrealized_pnl = (current_price - entry_price) / entry_price
                profit_threshold = atr * 1.0 / entry_price  # 1.0倍ATR开始启用动态止损（更早）
                
                if unrealized_pnl > profit_threshold:
                    # 计算动态止损价格（保护60%利润，更紧）
                    profit_protection = unrealized_pnl * 0.4
                    new_stop_loss = entry_price * (1 + profit_protection)
            else:  # short position for oscillation strategy
                unrealized_pnl = (entry_price - current_price) / entry_price
                profit_threshold = atr * 1.0 / entry_price  # 1.0倍ATR开始启用动态止损（更早）
                
                if unrealized_pnl > profit_threshold:
                    profit_protection = unrealized_pnl * 0.4  # 保护60%利润，更紧
                    new_stop_loss = entry_price * (1 - profit_protection)
        else:
            # 趋势策略：原有参数
            if side == 'long':
                unrealized_pnl = (current_price - entry_price) / entry_price
                profit_threshold = atr * 1.5 / entry_price  # 1.5倍ATR开始启用动态止损
                
                if unrealized_pnl > profit_threshold:
                    # 计算动态止损价格（保护50%利润）
                    profit_protection = unrealized_pnl * 0.5
                    new_stop_loss = entry_price * (1 + profit_protection)
            else:  # short position for trend strategy
                unrealized_pnl = (entry_price - current_price) / entry_price
                profit_threshold = atr * 1.5 / entry_price
                
                if unrealized_pnl > profit_threshold:
                    profit_protection = unrealized_pnl * 0.5
                    new_stop_loss = entry_price * (1 - profit_protection)
        
        # 检查是否需要更新止损（修复：检查条件单而不是stop类型）
        current_orders = exchange.fetch_open_orders(symbol)
        stop_orders = [o for o in current_orders if o['type'] == 'conditional' and 'slTriggerPx' in o.get('info', {}).get('params', {})]
        
        should_update = True
        # 只有在有新的止损价格时才更新
        if 'new_stop_loss' in locals():
            for order in stop_orders:
                if abs(float(order['stopPrice']) - new_stop_loss) < new_stop_loss * 0.01:
                    should_update = False
                    break
        else:
            should_update = False
        
        if should_update and 'new_stop_loss' in locals():
            # 只有在K线已收盘时才更新止损单
            if current_time >= kline_end_time:
                # 取消旧的止损单
                for order in stop_orders:
                    try:
                        exchange.cancel_order(order['id'], symbol)
                    except:
                        pass
                
                # 下新的止损单（修复：使用条件单）
                side_action = 'sell' if side == 'long' else 'buy'
                pos_side = 'long' if side == 'long' else 'short'
                
                exchange.create_order(
                    symbol=symbol,
                    type='conditional',
                    side=side_action,
                    amount=abs(size),
                    price=new_stop_loss,
                    params={
                        'slTriggerPx': new_stop_loss,
                        'slOrdPx': new_stop_loss,
                        'tdMode': 'cross',
                        'posSide': pos_side,
                        'reduceOnly': True
                    }
                )
                
                log_message("INFO", f"更新动态止损 {symbol}: {new_stop_loss:.4f}")
                return True
            else:
                log_message("DEBUG", f"{symbol} 当前K线未收盘，跳过止损更新")
                return False
        
        return False
        
    except Exception as e:
        log_message("ERROR", f"检查动态止损失败 {symbol}: {e}")
        return False

def check_pending_signals():
    """检查pending信号是否已经可以确认"""
    current_time = datetime.now(timezone(timedelta(hours=8))).timestamp()
    confirmed_signals = []
    
    for symbol, pending_info in list(position_tracker['pending_signals'].items()):
        signal = pending_info['signal']
        kline_end_time = pending_info.get('kline_end_time', 0)
        
        # 检查K线是否已经收盘
        if current_time >= kline_end_time:
            # K线已经收盘，重新检查信号
            log_message("DEBUG", f"{symbol} K线已收盘，重新检查信号")
            
            # 重新生成信号
            new_signal = generate_signal(symbol)
            
            if new_signal and new_signal.get('signal_strength') in ['strong', 'medium', 'weak']:
                # 信号确认，添加到确认列表
                confirmed_signals.append((symbol, new_signal))
                log_message("INFO", f"{symbol} pending信号确认: {new_signal['side']} @ {new_signal['price']:.4f}")
            else:
                # 信号不再有效，清除pending信号
                log_message("DEBUG", f"{symbol} pending信号失效，已清除")
            
            # 无论是否确认，都清除pending信号
            del position_tracker['pending_signals'][symbol]
    
    return confirmed_signals

def setup_missing_stop_orders(position, symbol):
    """为现有持仓设置止盈止损（修复：防止重复设置，区分趋势和震荡策略）"""
    try:
        # symbol 由调用方传入，避免 position 缺少该字段导致的 KeyError
        entry_price = float(position['entry_price'])
        side = position['side']
        size = float(position['size'])
        strategy_type = position.get('strategy_type', 'trend')  # 默认为趋势策略
        
        # 检查是否最近已经设置过止盈止损（使用全局order_tracking字典）
        if symbol in order_tracking:
            last_setup_time = order_tracking[symbol]['last_setup_time']
            if time.time() - last_setup_time < 300:  # 5分钟内不重复设置
                log_message("DEBUG", f"{symbol} 最近已设置过止盈止损，跳过重复设置")
                return False
        
        # 检查是否最近已经设置过止盈止损（避免重复设置）
        current_time = datetime.now(timezone(timedelta(hours=8)))
        last_setup_time = position_tracker.get('last_stop_setup', {}).get(symbol)
        
        if last_setup_time and (current_time - last_setup_time).total_seconds() < 600:  # 10分钟内不重复设置
            log_message("DEBUG", f"{symbol} 最近已设置过止盈止损，跳过重复设置")
            return False
        
        # 更新最后设置时间
        if 'last_stop_setup' not in position_tracker:
            position_tracker['last_stop_setup'] = {}
        position_tracker['last_stop_setup'][symbol] = current_time
        
        # 添加全局订单跟踪，防止重复设置
        if 'active_stop_orders' not in position_tracker:
            position_tracker['active_stop_orders'] = {}
        
        # 检查是否已经有活跃的止盈止损订单
        if symbol in position_tracker['active_stop_orders']:
            active_orders = position_tracker['active_stop_orders'][symbol]
            order_time = active_orders.get('timestamp')
            if order_time and (current_time - order_time).total_seconds() < 600:  # 10分钟内不重复设置
                log_message("DEBUG", f"{symbol} 已有活跃的止盈止损订单，跳过重复设置")
                return False
        
        # 获取ATR
        ohlcv = exchange.fetch_ohlcv(symbol, '30m', limit=50)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        atr = calculate_atr(df['high'], df['low'], df['close']).iloc[-1]
        
        # 根据策略类型计算不同的止盈止损参数
        if strategy_type == 'oscillation':
            # 震荡策略：更紧的止损，更小的目标
            if USE_ATR_DYNAMIC_STOPS and atr and atr > 0:
                # 震荡策略使用更小的ATR倍数
                atr_sl_multiplier = max(ATR_MIN_MULTIPLIER, min(ATR_MAX_MULTIPLIER, ATR_STOP_LOSS_MULTIPLIER * 0.5))  # 减少50%
                atr_tp_multiplier = max(ATR_MIN_MULTIPLIER, min(ATR_MAX_MULTIPLIER, ATR_TAKE_PROFIT_MULTIPLIER * 0.6))  # 减少40%
                
                if side == 'long':
                    stop_loss = max(entry_price * 0.96, entry_price - (atr * atr_sl_multiplier))  # 至少4%止损
                    take_profit = entry_price + (atr * atr_tp_multiplier)
                else:  # short
                    stop_loss = min(entry_price * 1.04, entry_price + (atr * atr_sl_multiplier))  # 至少4%止损
                    take_profit = max(entry_price * 0.95, entry_price - (atr * atr_tp_multiplier))  # 至少5%止盈
            else:
                # 固定百分比止损止盈（震荡策略更紧）
                if side == 'long':
                    stop_loss = entry_price * 0.96  # 4%止损
                    take_profit = entry_price * 1.03  # 3%止盈
                else:  # short
                    stop_loss = entry_price * 1.04  # 4%止损
                    take_profit = entry_price * 0.95  # 5%止盈
            
            log_message("DEBUG", f"{symbol} 震荡策略止盈止损: 入场价={entry_price:.4f}, 止损={stop_loss:.4f}, 止盈={take_profit:.4f}")
        else:
            # 趋势策略：使用原有参数
            stop_loss_tp = calculate_stop_loss_take_profit(symbol, entry_price, side, atr)
            if not stop_loss_tp:
                return False
            
            stop_loss = stop_loss_tp['stop_loss']
            take_profit = stop_loss_tp['take_profit']
            log_message("DEBUG", f"{symbol} 趋势策略止盈止损: 入场价={entry_price:.4f}, 止损={stop_loss:.4f}, 止盈={take_profit:.4f}")
        
        # 检查是否已有止损单（修复：改进订单状态检查逻辑）
        current_orders = exchange.fetch_open_orders(symbol)
        
        # 改进的订单检查逻辑：检查订单类型、价格和详细信息
        has_stop_loss = False
        has_take_profit = False
        
        for order in current_orders:
            order_type = order.get('type', '')
            order_price = float(order.get('price', 0))
            order_side = order.get('side', '')
            order_amount = float(order.get('amount', 0))
            order_info = order.get('info', {})
            
            # 打印订单详细信息用于调试
            log_message("DEBUG", f"检查订单: 类型={order_type}, 价格={order_price:.4f}, 方向={order_side}, 数量={order_amount}")
            
            # 检查止损单：价格接近止损价，且是条件单或止损单
            if (order_type in ['conditional', 'stop', 'stop_loss']) and abs(order_price - stop_loss) < stop_loss * 0.01:
                has_stop_loss = True
                log_message("DEBUG", f"找到止损单: 价格={order_price:.4f}, 止损价={stop_loss:.4f}")
            
            # 检查止盈单：价格接近止盈价，且是条件单或限价单
            if (order_type in ['conditional', 'limit', 'take_profit']) and abs(order_price - take_profit) < take_profit * 0.01:
                has_take_profit = True
                log_message("DEBUG", f"找到止盈单: 价格={order_price:.4f}, 止盈价={take_profit:.4f}")
            
            # 额外检查：通过订单信息中的参数识别
            if 'slTriggerPx' in str(order_info) or 'stopPrice' in str(order_info):
                trigger_price = float(order_info.get('slTriggerPx') or order_info.get('stopPrice') or 0)
                if abs(trigger_price - stop_loss) < stop_loss * 0.01:
                    has_stop_loss = True
                    log_message("DEBUG", f"通过参数找到止损单: 触发价={trigger_price:.4f}")
            
            if 'tpTriggerPx' in str(order_info) or 'takeProfitPrice' in str(order_info):
                trigger_price = float(order_info.get('tpTriggerPx') or order_info.get('takeProfitPrice') or 0)
                if abs(trigger_price - take_profit) < take_profit * 0.01:
                    has_take_profit = True
                    log_message("DEBUG", f"通过参数找到止盈单: 触发价={trigger_price:.4f}")
        
        # 添加详细日志显示当前订单状态
        log_message("DEBUG", f"{symbol} 当前订单状态 - 止损单: {has_stop_loss}, 止盈单: {has_take_profit}, 总订单数: {len(current_orders)}")
        
        # 设置止损单（修复：使用条件单而不是stop类型）
        if not has_stop_loss:
            if side == 'long':
                exchange.create_order(
                    symbol=symbol,
                    type='conditional',
                    side='sell',
                    amount=abs(size),
                    price=stop_loss,
                    params={
                        'slTriggerPx': stop_loss,
                        'slOrdPx': stop_loss,
                        'tdMode': 'cross',
                        'posSide': 'long',
                        'reduceOnly': True
                    }
                )
            else:
                exchange.create_order(
                    symbol=symbol,
                    type='conditional',
                    side='buy',
                    amount=abs(size),
                    price=stop_loss,
                    params={
                        'slTriggerPx': stop_loss,
                        'slOrdPx': stop_loss,
                        'tdMode': 'cross',
                        'posSide': 'short',
                        'reduceOnly': True
                    }
                )
            
            log_message("INFO", f"设置止损单 {symbol}: {stop_loss:.4f}")
        
        # 设置止盈单（修复：使用条件单而不是限价单，避免"止盈触发价不能低于最新价格"错误）
        if not has_take_profit:
            # 获取当前价格进行严格验证
            current_price = float(exchange.fetch_ticker(symbol)['last'])
            
            # 严格验证止盈价格合理性，避免"止盈触发价不能低于最新价格"错误
            if side == 'long':
                # 做多：止盈必须高于当前价格和入场价
                if take_profit <= current_price:
                    log_message("WARNING", f"止盈价{take_profit:.4f}低于当前价{current_price:.4f}，重新计算止盈价")
                    # 确保止盈价高于当前价至少2%
                    take_profit = max(current_price * 1.02, entry_price * 1.06)
                if take_profit <= entry_price:
                    log_message("WARNING", f"止盈价{take_profit:.4f}低于入场价{entry_price:.4f}，重新计算止盈价")
                    take_profit = entry_price * 1.06
            else:  # short
                # 做空：止盈必须低于当前价格和入场价
                if take_profit >= current_price:
                    log_message("WARNING", f"止盈价{take_profit:.4f}高于当前价{current_price:.4f}，重新计算止盈价")
                    take_profit = min(current_price * 0.98, entry_price * 0.94)
                if take_profit >= entry_price:
                    log_message("WARNING", f"止盈价{take_profit:.4f}高于入场价{entry_price:.4f}，重新计算止盈价")
                    take_profit = entry_price * 0.94
            
            # 最终验证：确保止盈价与当前价有足够差距
            if side == 'long' and take_profit <= current_price * 1.01:
                take_profit = current_price * 1.03
                log_message("DEBUG", f"最终调整止盈价: {take_profit:.4f}")
            elif side == 'short' and take_profit >= current_price * 0.99:
                take_profit = current_price * 0.97
                log_message("DEBUG", f"最终调整止盈价: {take_profit:.4f}")
            
            # 使用条件单（conditional）而不是限价单，避免"止盈触发价不能低于最新价格"错误
            if side == 'long':
                exchange.create_order(
                    symbol=symbol,
                    type='conditional',
                    side='sell',
                    amount=abs(size),
                    price=take_profit,
                    params={
                        'tpTriggerPx': take_profit,
                        'tpOrdPx': take_profit,
                        'tdMode': 'cross',
                        'posSide': 'long',
                        'reduceOnly': True
                    }
                )
            else:
                exchange.create_order(
                    symbol=symbol,
                    type='conditional',
                    side='buy',
                    amount=abs(size),
                    price=take_profit,
                    params={
                        'tpTriggerPx': take_profit,
                        'tpOrdPx': take_profit,
                        'tdMode': 'cross',
                        'posSide': 'short',
                        'reduceOnly': True
                    }
                )
            
            log_message("INFO", f"设置止盈单 {symbol}: {take_profit:.4f} (当前价: {current_price:.4f}, 入场价: {entry_price:.4f})")
        
        # 更新全局订单跟踪信息
        position_tracker['active_stop_orders'][symbol] = {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'timestamp': current_time,
            'entry_price': entry_price
        }
        
        log_message("DEBUG", f"{symbol} 止盈止损订单跟踪已更新")
        
        # 更新全局订单跟踪字典，防止重复设置
        order_tracking[symbol] = {
            'last_setup_time': time.time(),
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_price': entry_price
        }
        log_message("DEBUG", f"{symbol} 全局订单跟踪已更新")
        
        return True
        
    except Exception as e:
        log_message("ERROR", f"设置止盈止损失败: {e}")
        return False

# =================================
# 详细统计分析模块
# =================================

def update_detailed_trade_stats(symbol, side, pnl, entry_price, exit_price, entry_time, exit_time):
    """更新详细的交易统计"""
    global trade_stats
    
    try:
        # 基础统计
        trade_stats['total_trades'] += 1
        
        # 盈亏统计
        if pnl > 0:
            trade_stats['winning_trades'] += 1
            trade_stats['total_profit'] += pnl
        else:
            trade_stats['losing_trades'] += 1
            trade_stats['total_loss'] += abs(pnl)
        
        # 交易详情
        trade_detail = {
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_percentage': (pnl / (entry_price * 0.8)) * 100,  # 基于仓位大小计算收益率
            'entry_time': entry_time,
            'exit_time': exit_time,
            'hold_duration': (exit_time - entry_time).total_seconds() / 3600,  # 持仓小时数
            'timestamp': datetime.now(timezone(timedelta(hours=8)))
        }
        
        # 保存交易历史（最多100笔）
        trade_stats['trade_history'].append(trade_detail)
        if len(trade_stats['trade_history']) > 100:
            trade_stats['trade_history'] = trade_stats['trade_history'][-100:]
        
        # 计算胜率
        if trade_stats['total_trades'] > 0:
            trade_stats['win_rate'] = (trade_stats['winning_trades'] / trade_stats['total_trades']) * 100
        
        # 打印统计信息
        profit_factor = (trade_stats['total_profit'] / trade_stats['total_loss']) if trade_stats['total_loss'] > 0 else float('inf')
        
        log_message("INFO", f"交易统计更新 - 总交易: {trade_stats['total_trades']}, "
                           f"胜率: {trade_stats['win_rate']:.1f}%, 盈利因子: {profit_factor:.2f}")
        
    except Exception as e:
        log_message("ERROR", f"更新交易统计失败: {e}")

def get_performance_report():
    """生成性能报告"""
    try:
        if trade_stats['total_trades'] == 0:
            return "暂无交易数据"
        
        win_rate = trade_stats['win_rate']
        profit_factor = (trade_stats['total_profit'] / trade_stats['total_loss']) if trade_stats['total_loss'] > 0 else float('inf')
        net_profit = trade_stats['total_profit'] - trade_stats['total_loss']
        
        # 平均盈亏
        avg_win = trade_stats['total_profit'] / trade_stats['winning_trades'] if trade_stats['winning_trades'] > 0 else 0
        avg_loss = trade_stats['total_loss'] / trade_stats['losing_trades'] if trade_stats['losing_trades'] > 0 else 0
        
        # 构建报告
        report_lines = [
            "=== 交易性能报告 ===",
            f"总交易次数: {trade_stats['total_trades']}",
            f"盈利交易: {trade_stats['winning_trades']}",
            f"亏损交易: {trade_stats['losing_trades']}",
            f"胜率: {win_rate:.2f}%",
            f"盈利因子: {profit_factor:.2f}",
            f"净利润: {net_profit:.2f} USDT",
            f"平均盈利: {avg_win:.2f} USDT",
            f"平均亏损: {avg_loss:.2f} USDT"
        ]
        
        # 最近5笔交易
        if trade_stats['trade_history']:
            report_lines.append("=== 最近5笔交易 ===")
            recent_trades = trade_stats['trade_history'][-5:]
            for trade in recent_trades:
                trade_line = f"{trade['symbol']} {trade['side']} PnL: {trade['pnl']:.2f} ({trade['pnl_percentage']:.2f}%)"
                report_lines.append(trade_line)
        
        return "\n".join(report_lines)

        
    except Exception as e:
        log_message("ERROR", f"生成性能报告失败: {e}")
        return "生成报告失败"

# =================================
# 条件单管理模块
# =================================

def manage_conditional_orders():
    """管理条件单"""
    try:
        for symbol in position_tracker['positions']:
            position = position_tracker['positions'][symbol]
            
            # 检查动态止损
            check_trailing_stop(symbol, position)
            
            # 设置或补齐止盈止损单（不依赖 open_orders 类型判断）
            setup_missing_stop_orders(position, symbol)
        
    except Exception as e:
        log_message("ERROR", f"管理条件单失败: {e}")

# =================================
# 风险管理模块
# =================================

def check_risk_limits():
    """检查风险限制"""
    try:
        # 检查最大持仓数
        if len(position_tracker['positions']) >= MAX_OPEN_POSITIONS:
            log_message("WARNING", f"已达到最大持仓数限制: {MAX_OPEN_POSITIONS}")
            return False
        
        # 检查每日交易次数
        today = datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d')
        if position_tracker['daily_stats']['date'] != today:
            # 重置每日统计
            position_tracker['daily_stats'] = {
                'date': today,
                'trades_count': 0,
                'total_pnl': 0
            }
        
        if position_tracker['daily_stats']['trades_count'] >= MAX_DAILY_TRADES:
            log_message("WARNING", f"已达到每日最大交易次数: {MAX_DAILY_TRADES}")
            return False
        
        return True
        
    except Exception as e:
        log_message("ERROR", f"检查风险限制失败: {e}")
        return True

# =================================
# 增强版交易循环
# =================================

def enhanced_trading_loop():
    """增强版主交易循环"""
    try:
        log_message("SUCCESS", "开始增强版交易循环...")
        
        # 启动时运行全面回测
        log_message("INFO", "正在运行全面策略回测分析...")
        backtest_results = run_comprehensive_backtest(None, days_list=[7, 14, 30])
        
        if backtest_results:
            # 保存回测结果到文件
            try:
                timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
                backtest_file = f"backtest_results_{timestamp}.txt"
                with open(backtest_file, 'w', encoding='utf-8') as f:
                    f.write(generate_backtest_report(backtest_results))
                log_message("SUCCESS", f"回测结果已保存到: {backtest_file}")
            except Exception as e:
                log_message("WARNING", f"保存回测结果失败: {e}")
        
        while True:
            try:
                # 检查风险限制
                if not check_risk_limits():
                    log_message("WARNING", "触发风险限制，跳过本轮交易")
                    time.sleep(300)  # 等待5分钟
                    continue
                
                # 管理条件单
                manage_conditional_orders()
                
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
                            # 检查信号状态
                            if signal.get('signal_strength') == 'pending':
                                # 信号处于pending状态，等待K线收盘
                                log_message("DEBUG", f"{symbol} 信号等待K线收盘: {signal['side']} @ {signal['price']:.4f}, 剩余时间: {signal.get('time_remaining', 0):.0f}秒")
                                
                                # 记录pending信号
                                position_tracker['pending_signals'][symbol] = {
                                    'signal': signal,
                                    'timestamp': datetime.now(timezone(timedelta(hours=8))),
                                    'kline_end_time': signal.get('kline_end_time', 0)
                                }
                                
                            elif signal.get('signal_strength') in ['strong', 'medium', 'weak']:
                                # 确认信号，执行交易
                                log_message("INFO", f"{symbol} 发现确认信号: {signal['side']} @ {signal['price']:.4f}")
                                
                                # 如果之前有pending信号，清除它
                                if symbol in position_tracker['pending_signals']:
                                    del position_tracker['pending_signals'][symbol]
                                
                                # 执行交易
                                if execute_trade(symbol, signal, signal['signal_strength']):
                                    # 更新每日统计
                                    position_tracker['daily_stats']['trades_count'] += 1
                                    
                                    # 设置止盈止损
                                    time.sleep(2)  # 等待订单确认
                                    if symbol in position_tracker['positions']:
                                        setup_missing_stop_orders(position_tracker['positions'][symbol], symbol)
                        
                        time.sleep(1)  # 短暂延迟避免API限制
                        
                    except Exception as e:
                        log_message("ERROR", f"处理 {symbol} 失败: {str(e)}")
                        continue
                
                # 每小时生成一次性能报告
                current_time = datetime.now(timezone(timedelta(hours=8)))
                if current_time.minute == 0:  # 整点时
                    report = get_performance_report()
                    log_message("INFO", f"性能报告:\n{report}")

                
                # 主循环延迟
                log_message("INFO", f"交易循环完成，等待{MAIN_LOOP_DELAY}秒...")
                time.sleep(MAIN_LOOP_DELAY)
                
            except Exception as e:
                log_message("ERROR", f"交易循环中出错: {str(e)}")
                time.sleep(60)
                
    except KeyboardInterrupt:
        log_message("INFO", "收到退出信号，正在安全关闭...")
        # 生成最终报告
        final_report = get_performance_report()
        log_message("INFO", f"最终交易报告:\n{final_report}")

    except Exception as e:
        log_message("ERROR", f"增强版交易循环启动失败: {str(e)}")
        traceback.print_exc()

# =================================
# 单策略回测入口（EMA5/EMA10 + BB(20,2)过滤，0.5%阈值）
# =================================
def run_ema_bb_backtest(symbols=None, days_list=[7, 14, 30], initial_balance=10000):
    """
    运行单策略综合回测（EMA5/EMA10 + BB(20,2)入场过滤，0.5%阈值），并输出汇总报告。
    - symbols=None: 自动抓取交易所USDT合约热度前10，并追加 FIL/ZRO/WIF/WLD
    - days_list: 回测天数列表
    - initial_balance: 初始资金
    """
    try:
        # 自动获取热门标的（与综合回测一致）
        if symbols is None:
            try:
                hot = []
                if 'exchange' in globals() and exchange:
                    tickers = exchange.fetch_tickers()
                    for sym, tk in tickers.items():
                        if (sym.endswith('-USDT-SWAP') or sym.endswith(':USDT')) and ('SWAP' in sym or ':' in sym):
                            vol = tk.get('quoteVolume') or tk.get('baseVolume')
                            if vol is None and isinstance(tk.get('info'), dict):
                                info = tk['info']
                                vol = float(info.get('volCcy24h')) if info.get('volCcy24h') else None
                            if vol:
                                hot.append((sym, float(vol)))
                    hot.sort(key=lambda x: x[1], reverse=True)
                    top10 = [s for s, _ in hot[:10]]
                else:
                    top10 = SYMBOLS[:10]
                def norm_sym(s):
                    return s.replace(':USDT', '-USDT-SWAP') if ':USDT' in s else (s if s.endswith('-USDT-SWAP') else s + '-USDT-SWAP')
                base = [norm_sym(s) for s in top10]
                extras = ['FIL-USDT-SWAP', 'ZRO-USDT-SWAP', 'WIF-USDT-SWAP', 'WLD-USDT-SWAP']
                seen, symbols = set(), []
                for s in base + extras:
                    if s not in seen:
                        symbols.append(s); seen.add(s)
                log_message("INFO", f"[EMA/BB] 自动获取回测标的: {symbols[:10]} + extras")
            except Exception as e:
                log_message("WARNING", f"[EMA/BB] 自动获取热门标的失败，使用默认: {str(e)}")
                symbols = SYMBOLS[:10] + ['FIL-USDT-SWAP', 'ZRO-USDT-SWAP', 'WIF-USDT-SWAP', 'WLD-USDT-SWAP']

        report_lines = ["=== EMA5/EMA10 + BB(20,2) 入场过滤（0.5%）回测报告 ===", f"标的数量: {len(symbols)}", ""]
        total_trades_all = 0
        sum_win_rate_all = 0.0
        sum_return_all = 0.0
        results_summary = []

        for days in days_list:
            log_message("INFO", f"[EMA/BB] 开始 {days} 天回测分析...")
            day_results = []
            for sym in symbols:
                try:
                    res = backtest_strategy(sym, days=days, initial_balance=initial_balance)
                    if res:
                        day_results.append(res)
                        total_trades_all += res.get('total_trades', 0)
                        sum_win_rate_all += res.get('win_rate', 0.0)
                        sum_return_all += res.get('total_return', 0.0)
                        log_message("INFO", f"[EMA/BB] {sym} {days}天: 胜率{res['win_rate']:.1f}% 收益率{res['total_return']:.2f}% 交易{res['total_trades']}")
                except Exception as e:
                    log_message("ERROR", f"[EMA/BB] 回测 {sym} 失败: {e}")
                    continue

            if day_results:
                avg_win = sum(r['win_rate'] for r in day_results) / len(day_results)
                avg_ret = sum(r['total_return'] for r in day_results) / len(day_results)
                report_lines.extend([
                    f"=== {days}天回测结果 ===",
                    f"标的: {len(day_results)}",
                    f"总交易: {sum(r['total_trades'] for r in day_results)}",
                    f"平均胜率: {avg_win:.1f}%",
                    f"平均收益率: {avg_ret:.2f}%",
                    ""
                ])
                results_summary.append({'days': days, 'avg_win': avg_win, 'avg_ret': avg_ret})

        # 汇总与保存
        timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
        out_file = f"backtest_results_ema_bb_{timestamp}.txt"
        with open(out_file, 'w', encoding='utf-8') as f:
            f.write("
".join(report_lines))
        log_message("SUCCESS", f"[EMA/BB] 回测结果已保存到: {out_file}")
        return results_summary

    except Exception as e:
        log_message("ERROR", f"[EMA/BB] 运行单策略回测失败: {e}")
        return None

# =================================
# 主程序入口 - 增强版
# =================================
# 保留原有增强循环：改为仅运行单一策略（EMA5/EMA10）的综合回测与交易
_orig_enhanced_trading_loop = enhanced_trading_loop

def enhanced_trading_loop():
    log_message("INFO", "正在运行单一策略综合回测（EMA5/EMA10），禁用多策略回测入口")
    try:
        # 直接继续原有增强循环（其内部已执行综合回测 run_comprehensive_backtest(None, ...)）
        return _orig_enhanced_trading_loop()
    except Exception as e:
        log_message("WARNING", f"单策略综合回测阶段出错: {e}")
        return _orig_enhanced_trading_loop()

if __name__ == "__main__":
    # 使用增强版交易循环替代原版本
    try:
        # 初始化交易所连接
        exchange = initialize_exchange()
        if not exchange:
            log_message("ERROR", "交易所初始化失败")
            exit(1)
        
        # 测试API连接
        if not test_api_connection():
            log_message("ERROR", "API连接测试失败，请检查配置")
            exit(1)
        
        # 显示启动信息
        log_message("SUCCESS", "MACD(6,16,9)策略交易系统启动成功")
        log_message("INFO", f"智能杠杆系统: BTC最大{MAX_LEVERAGE_BTC}x, ETH最大{MAX_LEVERAGE_ETH}x")
        log_message("INFO", f"交易对数量: {len(SYMBOLS)}")
        log_message("INFO", f"最大持仓数: {MAX_OPEN_POSITIONS}")
        log_message("INFO", f"ATR动态止盈止损: 启用, 止损{ATR_STOP_LOSS_MULTIPLIER}x, 止盈{ATR_TAKE_PROFIT_MULTIPLIER}x")
        
        # 同步交易所现有持仓
        sync_exchange_positions()
        
        # 启动增强版交易循环
        enhanced_trading_loop()
        
    except Exception as e:
        log_message("ERROR", f"启动交易系统失败: {str(e)}")
        traceback.print_exc()