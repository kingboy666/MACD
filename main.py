import ccxt
import pandas as pd
import traceback
import numpy as np
from datetime import datetime, timedelta, timezone
import time
import json
from dotenv import load_dotenv
import os
import smtplib
from email.message import EmailMessage

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
MAX_LEVERAGE_BTC = 20                        # BTC最大杠杆
MAX_LEVERAGE_ETH = 20                        # ETH最大杠杆
MAX_LEVERAGE_MAJOR = 20                      # 主流币最大杠杆
MAX_LEVERAGE_OTHERS = 20                     # 其他币种最大杠杆
LEVERAGE_MIN = 10                             # 全局最低杠杆
DEFAULT_LEVERAGE = 10                         # 默认杠杆

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
MACD_FAST = 8                             # MACD快线周期
MACD_SLOW = 21                            # MACD慢线周期
MACD_SIGNAL = 9                           # MACD信号线周期

# 时间框架配置
TIMEFRAME_MAIN = '5m'       # 主图
TIMEFRAME_CONFIRM = '15m'   # 确认

# Bollinger Bands配置
BB_PERIOD = 20
BB_STD = 1.5  # 震荡更敏感

# ATR动态止盈止损配置
USE_ATR_DYNAMIC_STOPS = True                 # 启用ATR动态止盈止损
ATR_PERIOD = 14                              # ATR计算周期
ATR_STOP_LOSS_MULTIPLIER = 2.0              # ATR止损倍数
ATR_TAKE_PROFIT_MULTIPLIER = 3.0            # ATR止盈倍数
ATR_TRAILING_ACTIVATION_MULTIPLIER = 1.5    # 移动止盈激活倍数
ATR_TRAILING_CALLBACK_MULTIPLIER = 1.0      # 移动止盈回调倍数
ATR_MIN_MULTIPLIER = 1.0                    # ATR最小倍数
ATR_MAX_MULTIPLIER = 5.0                    # ATR最大倍数

# ADX配置（用于震荡识别，阈值20）
ADX_TREND_THRESHOLD = 20

# 风险管理配置
RISK_PER_TRADE = 0.02                        # 单笔风险2%
MAX_OPEN_POSITIONS = 5                       # 最大持仓数
COOLDOWN_PERIOD = 300                        # 冷却期5分钟
MAX_DAILY_TRADES = 100                        # 每日最大交易次数

# 主循环配置
MAIN_LOOP_DELAY = 10                         # 主循环延迟30秒

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

def notify_email(subject: str, body: str, to_addr: str = None):
    """发送邮件通知（默认QQ SMTP），需在环境变量配置 SMTP_HOST/PORT/USER/PASS/SSL/EMAIL_TO"""
    try:
        host = os.getenv('SMTP_HOST', 'smtp.qq.com')
        port = int(os.getenv('SMTP_PORT', '465'))
        user = os.getenv('SMTP_USER')
        pwd = os.getenv('SMTP_PASS')
        use_ssl = os.getenv('SMTP_SSL', 'true').lower() in ['true','1','yes']
        to = to_addr or os.getenv('EMAIL_TO') or user
        if not all([host, port, user, pwd, to]):
            log_message("WARNING", "邮件通知未配置完整，跳过")
            return
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = user
        msg['To'] = to
        msg.set_content(body)
        if use_ssl:
            with smtplib.SMTP_SSL(host, port, timeout=10) as s:
                s.login(user, pwd)
                s.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=10) as s:
                s.starttls()
                s.login(user, pwd)
                s.send_message(msg)
    except Exception as e:
        log_message("WARNING", f"邮件通知失败: {e}")

def log_signal_overview(symbol, signal):
    """输出信号详细概览：侧向、价格、强度、策略、RSI/ATR、确认类型、pending剩余时间"""
    try:
        side = signal.get('side')
        price = signal.get('price')
        strength = signal.get('signal_strength')
        strat = signal.get('strategy_type')
        rsi = signal.get('RSI')
        atr = signal.get('atr_value')
        confirm = signal.get('confirmation_type')
        pending = bool(signal.get('kline_pending'))
        time_remaining = signal.get('time_remaining')
        line = f"{symbol} 信号: side={side}, price={price:.4f}, strength={strength}, strat={strat or 'NA'}"
        if rsi is not None:
            try: line += f", RSI={float(rsi):.1f}"
            except: line += f", RSI={rsi}"
        if atr is not None:
            try: line += f", ATR={float(atr):.4f}"
            except: line += f", ATR={atr}"
        if confirm:
            line += f", confirm={confirm}"
        if pending:
            try:
                tr = float(time_remaining) if time_remaining is not None else None
                line += f", pending剩余={tr:.0f}s" if tr is not None else ", pending"
            except:
                line += ", pending"
        log_message("SIGNAL", line)
    except Exception as e:
        try:
            log_message("SIGNAL", f"{symbol} 信号: {signal.get('side')} @ {signal.get('price')}")
        except:
            log_message("SIGNAL", f"{symbol} 信号: 概览输出异常 {e}")

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
    """根据币种、账户与波动性动态计算杠杆，最大不超过20x"""
    try:
        base_symbol = symbol.split('-')[0].upper()
        
        if base_symbol == 'BTC':
            max_leverage = MAX_LEVERAGE_BTC
            base_leverage = 10
        elif base_symbol == 'ETH':
            max_leverage = MAX_LEVERAGE_ETH
            base_leverage = 8
        elif base_symbol in MAJOR_COINS:
            max_leverage = MAX_LEVERAGE_MAJOR
            base_leverage = 7
        else:
            max_leverage = MAX_LEVERAGE_OTHERS
            base_leverage = 6
        
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
        final_leverage = min(calculated_leverage, max_leverage)  # 上限20x
        final_leverage = max(final_leverage, LEVERAGE_MIN)       # 下限5x
        
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

def calculate_adx(high, low, close, period=14):
    try:
        plus_dm = (high.diff()).clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        # 仅保留有效方向动量
        plus_dm[high.diff() < low.diff()] = 0
        minus_dm[low.diff() < high.diff()] = 0

        tr1 = (high - low)
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean()

        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr).replace([np.inf, -np.inf], np.nan)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr).replace([np.inf, -np.inf], np.nan)

        dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan)
        adx = dx.rolling(window=period).mean()
        return adx
    except Exception:
        return pd.Series([np.nan] * len(close))

def process_klines(ohlcv):
    """处理K线数据并计算技术指标（VWAP日内重置、MACD(12,26,9)、RSI(14)、VWAP±1SD、成交量均值）"""
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
        
        # 计算VWAP(日内重置)、RSI(14)、VWAP标准差带、成交量均值、布林带
        try:
            typical_price = (df['high'] + df['low'] + df['close']) / 3.0
            # 明确生成列，避免使用 Series.name=None 造成 KeyError
            df['vwap_value'] = typical_price * df['volume']
            df['day'] = df['timestamp'].dt.date
            # 日内累加
            df['cum_vwap'] = df.groupby('day')['vwap_value'].cumsum()
            df['cum_volume'] = df.groupby('day')['volume'].cumsum()
            # 安全计算VWAP（避免除零），并前向填充保证列存在
            safe_cum_volume = df['cum_volume'].replace(0, np.nan)
            df['VWAP'] = (df['cum_vwap'] / safe_cum_volume).ffill()
            # RSI计算
            df['RSI'] = calculate_rsi(df['close'], period=14)
            # VWAP标准差带（按日内重置）
            df['vwap_diff'] = (typical_price - df['VWAP'])
            df['VWAP_SD'] = df.groupby('day')['vwap_diff'].transform(lambda s: s.rolling(window=20, min_periods=5).std())
            df['VWAP_UP'] = df['VWAP'] + df['VWAP_SD']
            df['VWAP_DOWN'] = df['VWAP'] - df['VWAP_SD']
            # 成交量20期均值（用于过滤）
            df['vol_ma20'] = df['volume'].rolling(window=20, min_periods=5).mean()
            # 计算布林带（与实盘一致，用于震荡识别与回测逻辑）
            try:
                df = calculate_bb(df)
            except Exception as e:
                log_message("WARNING", f"布林带计算失败: {str(e)}")
        except Exception as e:
            log_message("WARNING", f"VWAP/RSI/成交量计算失败: {str(e)}")
        
        # 计算ATR
        try:
            df['ATR_14'] = calculate_atr(df['high'], df['low'], df['close'], period=ATR_PERIOD)
        except Exception as e:
            log_message("WARNING", f"ATR计算失败: {str(e)}")
            df['ATR_14'] = 0
        
        # 已移除ADX计算
        # 占位以兼容旧代码路径，避免KeyError
        try:
            df['ADX'] = calculate_adx(df['high'], df['low'], df['close'], period=14)
        except Exception as e:
            log_message("WARNING", f"ADX计算失败: {str(e)}")
            df['ADX'] = 25
        
        return df
        
    except Exception as e:
        log_message("ERROR", f"处理K线数据失败: {str(e)}")
        return None



def calculate_rsi(close, period=14):
    """计算RSI指标"""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi



# 计算布林带指标
def calculate_bb(df: pd.DataFrame, period: int = BB_PERIOD, std: float = BB_STD) -> pd.DataFrame:
    try:
        mid = df['close'].rolling(period).mean()
        std_dev = df['close'].rolling(period).std()
        df['BB_mid'] = mid
        df['BB_upper'] = mid + (std_dev * std)
        df['BB_lower'] = mid - (std_dev * std)
        # 防止除零
        safe_mid = mid.replace(0, np.nan)
        df['BB_width'] = ((df['BB_upper'] - df['BB_lower']) / safe_mid).ffill()
    except Exception:
        # 保持健壮性：失败不阻断流程
        pass
    return df

def calculate_bb_rsi_1m_stops(symbol: str, entry_price: float, side: str):
    """
    1m BB+RSI窄带回归策略专用SL/TP：
    - SL：BB外0.5%缓冲（long: BB_lower - 0.5%*close；short: BB_upper + 0.5%*close）
    - TP：BB中轨
    """
    try:
        ohlcv = get_klines(symbol, '1m', limit=60)
        if not ohlcv:
            return None
        df = process_klines(ohlcv)
        if df is None or len(df) < 20 or 'BB_mid' not in df.columns:
            return None
        close = float(df['close'].iloc[-1])
        bb_mid = float(df['BB_mid'].iloc[-1])
        bb_upper = float(df['BB_upper'].iloc[-1])
        bb_lower = float(df['BB_lower'].iloc[-1])

        if side == 'long':
            stop_loss = max(0.0, bb_lower - (close * 0.005))
            take_profit = bb_mid
        else:
            stop_loss = bb_upper + (close * 0.005)
            take_profit = bb_mid

        return {'stop_loss': stop_loss, 'take_profit': take_profit}
    except Exception as e:
        log_message("ERROR", f"{symbol} 1m BB+RSI 计算SL/TP失败: {e}")
        return None

def generate_signal(symbol):
    """基于VWAP+MACD(12,26,9)+RSI(14)的日内策略生成交易信号（仅收盘确认，含成交量过滤）"""
    try:
        ohlcv = get_klines(symbol, TIMEFRAME_MAIN, limit=100)
        if not ohlcv:
            return None
        
        df = process_klines(ohlcv)
        if df is None or len(df) < 50:
            return None
        
        # 使用日内VWAP与RSI(14)作为过滤，指标在process_klines已计算
        
        # 获取当前时间戳，检查K线是否已收盘
        import time
        current_timestamp = int(time.time() * 1000)
        current_kline = ohlcv[-1]
        current_kline_start = current_kline[0]
        current_kline_end = current_kline_start + 5 * 60 * 1000  # 5分钟K线结束时间
        kline_completed = current_timestamp >= current_kline_end
        
        # 获取当前数据
        current_macd = df['MACD'].iloc[-1]
        current_signal = df['MACD_SIGNAL'].iloc[-1]
        prev_macd = df['MACD'].iloc[-2]
        prev_signal = df['MACD_SIGNAL'].iloc[-2]
        current_adx = None
        current_price = df['close'].iloc[-1]
        current_open = df['open'].iloc[-1]
        current_close = df['close'].iloc[-1]
        atr_value = df['ATR_14'].iloc[-1]
        
        # 指标数据（RSI与成交量过滤）



        current_rsi = df['RSI'].iloc[-1]
        vol_ma20 = df['vol_ma20'].iloc[-1] if 'vol_ma20' in df.columns else None
        volume_ok = (vol_ma20 is None) or (df['volume'].iloc[-1] >= 0.7 * vol_ma20)
        # 周末低量避开：周六/周日或当前量低于均值则暂停入场
        try:
            now_utc8 = datetime.now(timezone(timedelta(hours=8)))
            is_weekend = now_utc8.weekday() >= 5  # 5=周六,6=周日
            if is_weekend or (vol_ma20 is not None and df['volume'].iloc[-1] < vol_ma20):
                return None
        except Exception:
            pass
        # 手动停（新闻事件）：环境变量 NEWS_PAUSE=true 时暂停
        try:
            if os.getenv('NEWS_PAUSE','').lower() in ['true', '1', 'yes']:
                return None
        except Exception:
            pass


        
        # 检查MACD金叉死叉（主图）
        golden_cross = prev_macd <= prev_signal and current_macd > current_signal
        death_cross = prev_macd >= prev_signal and current_macd < current_signal

        # 15m确认框架：MACD金叉/死叉
        macd_confirm_golden = False
        macd_confirm_death = False
        try:
            ohlcv_confirm = get_klines(symbol, TIMEFRAME_CONFIRM, limit=50)
            if ohlcv_confirm:
                df_confirm = pd.DataFrame(ohlcv_confirm, columns=['timestamp','open','high','low','close','volume'])
                macd_c, sig_c, _ = calculate_macd(df_confirm['close'], fast=MACD_FAST, slow=MACC_SLOW if 'MACC_SLOW' in globals() else MACD_SLOW, signal=MACD_SIGNAL)
                prev_c = macd_c.iloc[-2]; prev_sig_c = sig_c.iloc[-2]
                curr_c = macd_c.iloc[-1]; curr_sig_c = sig_c.iloc[-1]
                macd_confirm_golden = (prev_c <= prev_sig_c and curr_c > curr_sig_c)
                macd_confirm_death = (prev_c >= prev_sig_c and curr_c < curr_sig_c)
        except Exception:
            pass
        
        # 检查K线阴阳线
        is_bullish = current_close > current_open  # 阳线：收盘价大于开盘价
        is_bearish = current_close < current_open  # 阴线：收盘价小于开盘价
        
        # VWAP+MACD(12,26,9)+RSI(14) 日内策略（优先执行，含成交量过滤与阳/阴线收盘确认）
        current_vwap = df['VWAP'].iloc[-1] if 'VWAP' in df.columns else None

        # 市场状态判定：BB宽度<2%为震荡；>2.5%为趋势
        is_sideways = False
        is_trend = False
        bb_width = None
        adx_val = None
        try:
            bb_width = float(df['BB_width'].iloc[-1]) if 'BB_width' in df.columns else None
            adx_val = float(df['ADX'].iloc[-1]) if 'ADX' in df.columns else None
            is_sideways = (bb_width is not None) and (bb_width < 0.02)
            is_trend = (bb_width is not None) and (bb_width > 0.025)
        except Exception:
            pass
        # 显示市场状态识别结果
        try:
            bw_disp = f"{bb_width:.4f}" if isinstance(bb_width, float) else "NA"
            adx_disp = f"{adx_val:.2f}" if isinstance(adx_val, float) else "NA"
            log_message("INFO", f"{symbol} 市场状态识别: ADX={adx_disp}, BB宽度={bw_disp}, 震荡={is_sideways}, 趋势={is_trend}")
        except Exception:
            pass

        if is_sideways:
            # 先应用 1m BB(20,1.5SD) + RSI(14) 窄带回归策略
            try:
                ohlcv_1m = get_klines(symbol, '1m', limit=120)
                if ohlcv_1m:
                    df1m = process_klines(ohlcv_1m)
                    if df1m is not None and len(df1m) >= 30 and 'BB_mid' in df1m.columns:
                        c1 = float(df1m['close'].iloc[-1])
                        r1 = float(df1m['RSI'].iloc[-1]) if 'RSI' in df1m.columns else None
                        bb_l1 = float(df1m['BB_lower'].iloc[-1])
                        bb_u1 = float(df1m['BB_upper'].iloc[-1])
                        bb_m1 = float(df1m['BB_mid'].iloc[-1])

                        # 1m K线收盘确认
                        k1 = ohlcv_1m[-1]
                        k1_start = k1[0]; k1_end = k1_start + 60 * 1000
                        now_ms = int(time.time() * 1000)
                        k1_completed = now_ms >= k1_end

                        # Long：触BB下轨 + RSI<30
                        if (c1 <= bb_l1 * 1.0000) and (r1 is not None and r1 < 30):
                            if k1_completed:
                                return {
                                    'symbol': symbol,
                                    'side': 'long',
                                    'price': c1,
                                    'signal_strength': 'strong',
                                    'atr_value': float(df1m['ATR_14'].iloc[-1]) if 'ATR_14' in df1m.columns else 0,
                                    'RSI': r1,
                                    'confirmation_type': '1m触下轨+RSI<30+K线收盘',
                                    'strategy_type': 'bb_rsi_1m'
                                }
                            else:
                                return {
                                    'symbol': symbol,
                                    'side': 'long',
                                    'price': c1,
                                    'signal_strength': 'pending',
                                    'atr_value': float(df1m['ATR_14'].iloc[-1]) if 'ATR_14' in df1m.columns else 0,
                                    'RSI': r1,
                                    'confirmation_type': '1m触下轨+RSI<30+等待K线收盘',
                                    'strategy_type': 'bb_rsi_1m',
                                    'kline_pending': True,
                                    'time_remaining': (k1_end - now_ms) / 1000.0,
                                    'kline_end_time': k1_end / 1000.0
                                }

                        # Short：触BB上轨 + RSI>70
                        if (c1 >= bb_u1 * 1.0000) and (r1 is not None and r1 > 70):
                            if k1_completed:
                                return {
                                    'symbol': symbol,
                                    'side': 'short',
                                    'price': c1,
                                    'signal_strength': 'strong',
                                    'atr_value': float(df1m['ATR_14'].iloc[-1]) if 'ATR_14' in df1m.columns else 0,
                                    'RSI': r1,
                                    'confirmation_type': '1m触上轨+RSI>70+K线收盘',
                                    'strategy_type': 'bb_rsi_1m'
                                }
                            else:
                                return {
                                    'symbol': symbol,
                                    'side': 'short',
                                    'price': c1,
                                    'signal_strength': 'pending',
                                    'atr_value': float(df1m['ATR_14'].iloc[-1]) if 'ATR_14' in df1m.columns else 0,
                                    'RSI': r1,
                                    'confirmation_type': '1m触上轨+RSI>70+等待K线收盘',
                                    'strategy_type': 'bb_rsi_1m',
                                    'kline_pending': True,
                                    'time_remaining': (k1_end - now_ms) / 1000.0,
                                    'kline_end_time': k1_end / 1000.0
                                }
            except Exception:
                pass

            close_ = float(df['close'].iloc[-1])
            rsi_ = float(df['RSI'].iloc[-1]) if 'RSI' in df.columns else None
            # 使用MACD交叉而非仅信号线方向
            golden = (df['MACD'].iloc[-2] <= df['MACD_SIGNAL'].iloc[-2]) and (df['MACD'].iloc[-1] > df['MACD_SIGNAL'].iloc[-1])
            death = (df['MACD'].iloc[-2] >= df['MACD_SIGNAL'].iloc[-2]) and (df['MACD'].iloc[-1] < df['MACD_SIGNAL'].iloc[-1])
            bb_lower = float(df['BB_lower'].iloc[-1]) if 'BB_lower' in df.columns else None
            bb_upper = float(df['BB_upper'].iloc[-1]) if 'BB_upper' in df.columns else None

            # Long：5m触下轨 + RSI<40 + VWAP>价 + 15m MACD金叉确认
            if (bb_lower is not None and close_ <= bb_lower * 1.0005) and (rsi_ is not None and rsi_ < 40) and (current_vwap is not None and close_ > float(current_vwap)) and macd_confirm_golden:
                if kline_completed:
                    return {
                        'symbol': symbol,
                        'side': 'long',
                        'price': close_,
                        'signal_strength': 'medium',
                        'atr_value': atr_value,
                        'VWAP': current_vwap,
                        'RSI': rsi_,
                        'confirmation_type': 'BB下轨反弹+RSI<40+MACD金叉+K线收盘',
                        'strategy_type': 'sideways_bb'
                    }
                else:
                    time_remaining = (current_kline_end - current_timestamp) / 1000
                    return {
                        'symbol': symbol,
                        'side': 'long',
                        'price': close_,
                        'signal_strength': 'pending',
                        'atr_value': atr_value,
                        'VWAP': current_vwap,
                        'RSI': rsi_,
                        'confirmation_type': 'BB下轨反弹+RSI<40+MACD金叉+等待K线收盘',
                        'strategy_type': 'sideways_bb',
                        'kline_pending': True,
                        'time_remaining': time_remaining,
                        'kline_end_time': current_kline_end / 1000.0
                    }

            # Short：5m触上轨 + RSI>60 + VWAP<价
            if (bb_upper is not None and close_ >= bb_upper * 0.9995) and (rsi_ is not None and rsi_ > 60) and (current_vwap is not None and close_ < float(current_vwap)):
                if kline_completed:
                    return {
                        'symbol': symbol,
                        'side': 'short',
                        'price': close_,
                        'signal_strength': 'strong',
                        'atr_value': atr_value,
                        'VWAP': current_vwap,
                        'RSI': rsi_,
                        'confirmation_type': 'BB上轨回落+RSI>60+MACD死叉+K线收盘',
                        'strategy_type': 'sideways_bb'
                    }
                else:
                    time_remaining = (current_kline_end - current_timestamp) / 1000
                    return {
                        'symbol': symbol,
                        'side': 'short',
                        'price': close_,
                        'signal_strength': 'pending',
                        'atr_value': atr_value,
                        'VWAP': current_vwap,
                        'RSI': rsi_,
                        'confirmation_type': 'BB上轨回落+RSI>60+MACD死叉+等待K线收盘',
                        'strategy_type': 'sideways_bb',
                        'kline_pending': True,
                        'time_remaining': time_remaining,
                        'kline_end_time': current_kline_end / 1000.0
                    }
        # 趋势模式继续走原VWAP逻辑
        current_rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else None
        vol_ma20 = df['vol_ma20'].iloc[-1] if 'vol_ma20' in df.columns else None
        volume_ok = (vol_ma20 is None) or (df['volume'].iloc[-1] >= 1.0 * vol_ma20)
        # Funding过滤：正funding>0.03%时避免做多
        try:
            funding = exchange.fetch_funding_rate(symbol).get('fundingRate')
            if funding is not None and funding > 0.0003 and golden_cross:
                return None
        except Exception:
            pass
        # 1h框架确认：多单需1小时MACD>0，空单需<0
        h1_ok = True
        try:
            ohlcv_1h = get_klines(symbol, '1h', limit=100)
            if ohlcv_1h:
                df1h = pd.DataFrame(ohlcv_1h, columns=['timestamp','open','high','low','close','volume'])
                macd_h, sig_h, _ = calculate_macd(df1h['close'], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
                macd_h_last = macd_h.iloc[-1]
                if golden_cross:
                    h1_ok = macd_h_last > 0
                elif death_cross:
                    h1_ok = macd_h_last < 0
        except Exception:
            h1_ok = True  # 兜底：若获取失败不阻断
        
        if current_vwap is not None and current_rsi is not None and volume_ok and h1_ok:
            vwap_bias = abs(current_price - current_vwap) / current_vwap > 0.002
            if golden_cross and (current_close > current_vwap) and vwap_bias and (current_rsi > 50) and is_bullish:
                if kline_completed:
                    return {
                        'symbol': symbol,
                        'side': 'long',
                        'price': current_price,
                        'signal_strength': 'strong',
                        'atr_value': atr_value,
                        'macd_value': current_macd,
                        'signal_value': current_signal,
                        'VWAP': current_vwap,
                        'RSI': current_rsi,
                        'confirmation_type': 'VWAP+MACD金叉+RSI>50+K线收盘',
                        'strategy_type': 'intraday_vwap_macd_rsi'
                    }
                else:
                    time_remaining = (current_kline_end - current_timestamp) / 1000
                    return {
                        'symbol': symbol,
                        'side': 'long',
                        'price': current_price,
                        'signal_strength': 'pending',
                        'atr_value': atr_value,
                        'macd_value': current_macd,
                        'signal_value': current_signal,
                        'VWAP': current_vwap,
                        'RSI': current_rsi,
                        'confirmation_type': 'VWAP+MACD金叉+RSI>50+等待K线收盘',
                        'strategy_type': 'intraday_vwap_macd_rsi',
                        'kline_pending': True,
                        'time_remaining': time_remaining,
                        'kline_end_time': current_kline_end / 1000.0
                    }
            elif death_cross and (current_close < current_vwap) and vwap_bias and (current_rsi < 50) and is_bearish:
                if kline_completed:
                    return {
                        'symbol': symbol,
                        'side': 'short',
                        'price': current_price,
                        'signal_strength': 'strong',
                        'atr_value': atr_value,
                        'macd_value': current_macd,
                        'signal_value': current_signal,
                        'VWAP': current_vwap,
                        'RSI': current_rsi,
                        'confirmation_type': 'VWAP+MACD死叉+RSI<50+K线收盘',
                        'strategy_type': 'intraday_vwap_macd_rsi'
                    }
                else:
                    time_remaining = (current_kline_end - current_timestamp) / 1000
                    return {
                        'symbol': symbol,
                        'side': 'short',
                        'price': current_price,
                        'signal_strength': 'pending',
                        'atr_value': atr_value,
                        'macd_value': current_macd,
                        'signal_value': current_signal,
                        'VWAP': current_vwap,
                        'RSI': current_rsi,
                        'confirmation_type': 'VWAP+MACD死叉+RSI<50+等待K线收盘',
                        'strategy_type': 'intraday_vwap_macd_rsi',
                        'kline_pending': True,
                        'time_remaining': time_remaining,
                        'kline_end_time': current_kline_end / 1000.0
                    }

        return None
        
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
        position_size = position_size * 0.999  # 0.1%滑点缓冲
        # 总敞口限制：不超过余额的30%
        try:
            total_balance = account_info['total_balance']
            current_exposure = 0.0
            for sym, pos in position_tracker['positions'].items():
                try:
                    px = float(exchange.fetch_ticker(sym)['last'])
                    current_exposure += abs(pos.get('size',0)) * px
                except:
                    pass
            new_exposure = current_exposure + (abs(position_size) * price)
            if total_balance > 0 and (new_exposure / total_balance) > 0.30:
                log_message("WARNING", f"{symbol} 下单将使总敞口超30%（{new_exposure/total_balance:.2%}），跳过本次交易")
                return False
        except Exception as e:
            log_message("WARNING", f"{symbol} 敞口检查异常，继续但建议关注风险: {e}")
        
        if position_size <= 0:
            log_message("WARNING", f"{symbol} 计算仓位大小为0，跳过交易")
            return False
        
        # 执行市价单（同时挂条件止盈止损）
        sl_tp = calculate_stop_loss_take_profit(symbol, price, signal['side'], signal.get('atr_value', 0))
        attach_algo = []
        if sl_tp:
            # 同时附带止盈/止损条件单（OKX attachAlgoOrds）
            attach_algo = [
                {'algoOrdType': 'tp', 'tpTriggerPx': sl_tp['take_profit'], 'tpOrdPx': sl_tp['take_profit']},
                {'algoOrdType': 'sl', 'slTriggerPx': sl_tp['stop_loss'], 'slOrdPx': sl_tp['stop_loss']}
            ]
        
        order = exchange.create_order(
            symbol,
            'market',
            side,
            position_size,
            None,
            {
                'tdMode': TD_MODE,
                'posSide': 'long' if signal['side'] == 'long' else 'short',
                'attachAlgoOrds': attach_algo
            }
        )
        
        if order:
            log_message("SUCCESS", f"{symbol} 交易成功: {side} {position_size} @ {price}")
            try:
                notify_email(
                    f"{symbol} 下单成功",
                    f"{'做多' if signal['side']=='long' else '做空'} {position_size:.6f} @ {price:.4f}\n策略: {signal.get('strategy_type','NA')}"
                )
            except Exception as e:
                log_message("WARNING", f"下单邮件通知失败: {e}")
            # 成交后立即同步挂条件单（原子化保障）
            try:
                ok_bracket = place_bracket_orders(symbol, signal['side'], position_size, price, signal.get('atr_value', 0))
                if not ok_bracket:
                    time.sleep(1)
                    place_bracket_orders(symbol, signal['side'], position_size, price, signal.get('atr_value', 0))
            except Exception as e:
                log_message("WARNING", f"{symbol} 同步挂条件单异常: {e}")
            
            # 强制即时验证条件单是否已存在，缺失则立即补挂（最多重试3次）
            try:
                retries = 3
                for attempt in range(1, retries + 1):
                    orders = exchange.fetch_open_orders(symbol)
                    has_sl = False; has_tp = False
                    for o in orders:
                        info_str = str(o.get('info', {}))
                        t = o.get('type', '')
                        p = float(o.get('price') or o.get('stopPrice') or 0) if (o.get('price') or o.get('stopPrice')) else 0.0
                        if (t in ['conditional', 'stop', 'stop_loss']) and ('slTriggerPx' in info_str or 'stopPrice' in info_str):
                            has_sl = True
                        if (t in ['conditional', 'limit', 'take_profit']) and ('tpTriggerPx' in info_str or 'takeProfitPrice' in info_str):
                            has_tp = True
                    log_message("DEBUG", f"{symbol} 条件单即时验证: SL={has_sl}, TP={has_tp}, 尝试{attempt}/{retries}")
                    if has_sl and has_tp:
                        break
                    # 缺失则即时补挂
                    sl_tp_now = calculate_stop_loss_take_profit(symbol, price, signal['side'], signal.get('atr_value', 0))
                    if sl_tp_now:
                        side_action_sl = 'sell' if signal['side'] == 'long' else 'buy'
                        pos_side = 'long' if signal['side'] == 'long' else 'short'
                        try:
                            # 补挂止损
                            exchange.create_order(
                                symbol=symbol,
                                type='conditional',
                                side=side_action_sl,
                                amount=abs(position_size),
                                price=sl_tp_now['stop_loss'],
                                params={
                                    'slTriggerPx': sl_tp_now['stop_loss'],
                                    'slOrdPx': sl_tp_now['stop_loss'],
                                    'tdMode': TD_MODE,
                                    'posSide': pos_side,
                                    'reduceOnly': True
                                }
                            )
                        except Exception as e:
                            log_message("WARNING", f"{symbol} 补挂止损失败: {e}")
                        try:
                            # 补挂止盈
                            exchange.create_order(
                                symbol=symbol,
                                type='conditional',
                                side=side_action_sl,
                                amount=abs(position_size),
                                price=sl_tp_now['take_profit'],
                                params={
                                    'tpTriggerPx': sl_tp_now['take_profit'],
                                    'tpOrdPx': sl_tp_now['take_profit'],
                                    'tdMode': TD_MODE,
                                    'posSide': pos_side,
                                    'reduceOnly': True
                                }
                            )
                        except Exception as e:
                            log_message("WARNING", f"{symbol} 补挂止盈失败: {e}")
                    time.sleep(1)
            except Exception as e:
                log_message("WARNING", f"{symbol} 条件单即时验证异常: {e}")
            
            # 记录持仓
            position_tracker['positions'][symbol] = {
                'symbol': symbol,
                'side': signal['side'],
                'size': position_size,
                'entry_price': price,
                'timestamp': datetime.now(timezone(timedelta(hours=8))),
                'atr_value': signal.get('atr_value', 0),
            'strategy_type': signal.get('strategy_type', 'trend')
            }
            
            # 开仓后立即设置止盈止损（强制即时挂条件单 + 兜底重试与验证）
            # 立即计算并下条件单，避免延迟
            try:
                sl_tp = calculate_stop_loss_take_profit(symbol, price, signal['side'], signal.get('atr_value', 0))
                if sl_tp:
                    side_action_sl = 'sell' if signal['side'] == 'long' else 'buy'
                    pos_side = 'long' if signal['side'] == 'long' else 'short'
                    # 止损条件单
                    try:
                        exchange.create_order(
                            symbol=symbol,
                            type='conditional',
                            side=side_action_sl,
                            amount=abs(position_size),
                            price=sl_tp['stop_loss'],
                            params={
                                'slTriggerPx': sl_tp['stop_loss'],
                                'slOrdPx': sl_tp['stop_loss'],
                                'tdMode': TD_MODE,
                                'posSide': pos_side,
                                'reduceOnly': True
                            }
                        )
                        log_message("INFO", f"{symbol} 即时设置止损单: {sl_tp['stop_loss']:.4f}")
                    except Exception as e:
                        log_message("WARNING", f"{symbol} 即时设置止损单失败: {e}")
                    # 止盈条件单
                    try:
                        exchange.create_order(
                            symbol=symbol,
                            type='conditional',
                            side=side_action_sl,
                            amount=abs(position_size),
                            price=sl_tp['take_profit'],
                            params={
                                'tpTriggerPx': sl_tp['take_profit'],
                                'tpOrdPx': sl_tp['take_profit'],
                                'tdMode': TD_MODE,
                                'posSide': pos_side,
                                'reduceOnly': True
                            }
                        )
                        log_message("INFO", f"{symbol} 即时设置止盈单: {sl_tp['take_profit']:.4f}")
                    except Exception as e:
                        log_message("WARNING", f"{symbol} 即时设置止盈单失败: {e}")
                # 兜底：执行旧逻辑再尝试并验证
                ok = setup_missing_stop_orders(position_tracker['positions'][symbol], symbol)
                ok = setup_missing_stop_orders(position_tracker['positions'][symbol], symbol)
                if not ok:
                    log_message("WARNING", f"{symbol} 首次设置止盈止损未成功，准备重试")
                    time.sleep(1)
                    setup_missing_stop_orders(position_tracker['positions'][symbol], symbol)
                # 验证止损/止盈条件单是否已存在
                try:
                    orders = exchange.fetch_open_orders(symbol)
                    has_sl = False; has_tp = False
                    for o in orders:
                        info_str = str(o.get('info', {}))
                        t = o.get('type', '')
                        p = float(o.get('price') or o.get('stopPrice') or 0) if (o.get('price') or o.get('stopPrice')) else 0.0
                        if (t in ['conditional', 'stop', 'stop_loss']) and ('slTriggerPx' in info_str or 'stopPrice' in info_str):
                            has_sl = True
                        if (t in ['conditional', 'limit', 'take_profit']) and ('tpTriggerPx' in info_str or 'takeProfitPrice' in info_str):
                            has_tp = True
                    if not has_sl or not has_tp:
                        log_message("WARNING", f"{symbol} 条件单验证失败: SL={has_sl}, TP={has_tp}，尝试再次补齐")
                        setup_missing_stop_orders(position_tracker['positions'][symbol], symbol)
                except Exception as ve:
                    log_message("WARNING", f"{symbol} 条件单验证异常: {ve}")
            except Exception as e:
                log_message("WARNING", f"同步设置止盈止损失败 {symbol}: {e}")
            
            return True
        
        return False
        
    except Exception as e:
        log_message("ERROR", f"{symbol} 执行交易失败: {str(e)}")
        return False

def calculate_stop_loss_take_profit(symbol, price, signal, atr_value):
    """计算止损止盈价格（静态规则：VWAP保护 + 入场K线极值±0.5% + ATR固定倍数；BNB使用1.5x ATR TP）"""
    try:
        current_price = float(exchange.fetch_ticker(symbol)['last'])
        ohlcv = get_klines(symbol, '5m', limit=2)
        df_last = process_klines(ohlcv)
        last_vwap = df_last['VWAP'].iloc[-1] if df_last is not None and 'VWAP' in df_last.columns else None
        last_high = float(df_last['high'].iloc[-1]) if df_last is not None else None
        last_low = float(df_last['low'].iloc[-1]) if df_last is not None else None
        atr_used = float(df_last['ATR_14'].iloc[-1]) if df_last is not None and 'ATR_14' in df_last.columns else (atr_value or 0)

        base_symbol = symbol.split('-')[0].upper()
        # 震荡模式识别（ADX<20 或 BB宽度<2%）
        is_sideways = False
        try:
            bb_width = float(df_last['BB_width'].iloc[-1]) if df_last is not None and 'BB_width' in df_last.columns else None
            adx_val = float(df_last['ADX'].iloc[-1]) if df_last is not None and 'ADX' in df_last.columns else None
            is_sideways = ((bb_width is not None and bb_width < 0.02) or (adx_val is not None and adx_val < ADX_TREND_THRESHOLD))
        except Exception:
            is_sideways = False
        # TP倍数（震荡缩紧为1.2x）
        tp_mult = 1.5 if base_symbol == 'BNB' else (1.2 if is_sideways else 2.0)

        if signal == 'long':
            candidates = []
            if last_vwap:
                candidates.append(last_vwap - (atr_used if atr_used > 0 else last_vwap * 0.01))  # VWAP - 1x ATR 或约1%
            if last_low:
                candidates.append(last_low * 0.995)  # 入场K线低点下方0.5%
            stop_loss = min(candidates) if candidates else (price * 0.99)

            take_profit = price + (atr_used * tp_mult) if atr_used > 0 else price * 1.02
        else:  # short
            candidates = []
            if last_vwap:
                candidates.append(last_vwap + (atr_used if atr_used > 0 else last_vwap * 0.01))  # VWAP + 1x ATR 或约1%
            if last_high:
                candidates.append(last_high * 1.005)  # 入场K线高点上方0.5%
            stop_loss = max(candidates) if candidates else (price * 1.01)

            take_profit = price - (atr_used * tp_mult) if atr_used > 0 else price * 0.98

        # 合理性校验
        if signal == 'long':
            if take_profit <= price:
                take_profit = price * 1.02
            if stop_loss >= price:
                stop_loss = price * 0.99
            if take_profit <= current_price:
                take_profit = max(current_price * 1.02, price * 1.02)
        else:
            if take_profit >= price:
                take_profit = price * 0.98
            if stop_loss <= price:
                stop_loss = price * 1.01
            if take_profit >= current_price:
                take_profit = min(current_price * 0.98, price * 0.98)

        # 震荡模式SL/TP收紧：SL=1.5x ATR, TP=1.2x ATR（若有ATR）
        try:
            if is_sideways and atr_used and atr_used > 0:
                if signal == 'long':
                    stop_loss = price - (atr_used * 1.5)
                    take_profit = price + (atr_used * 1.2)
                else:
                    stop_loss = price + (atr_used * 1.5)
                    take_profit = price - (atr_used * 1.2)
        except Exception:
            pass

        log_message("DEBUG", f"{symbol} {signal} SL/TP: entry={price:.4f}, vwap={last_vwap if last_vwap else 0:.4f}, atr={atr_used:.4f}, SL={stop_loss:.4f}, TP={take_profit:.4f}")
        return {'stop_loss': stop_loss, 'take_profit': take_profit}
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

def check_positions_legacy():
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
        
        order = exchange.create_order(
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
            try:
                notify_email(
                    f"{symbol} 平仓成功",
                    f"{'平多' if position['side']=='long' else '平空'} {size:.6f} @ {current_price:.4f}
PnL: {pnl:.4f}
策略: {position.get('strategy_type','NA')}"
                )
            except Exception as e:
                log_message("WARNING", f"平仓邮件通知失败: {e}")
            
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
                            log_signal_overview(symbol, signal)
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
        log_message("SUCCESS", "VWAP+MACD(12,26,9)+RSI(14)策略交易系统启动成功")
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
        limit = days * 24 * 12  # 5分钟K线，每天288根
        
        ohlcv = exchange.fetch_ohlcv(symbol, '5m', limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        return df
    except Exception as e:
        log_message("ERROR", f"获取历史数据失败 {symbol}: {e}")
        return None

def backtest_strategy(symbol, days=7, initial_balance=10000):
    """策略回测 - 增强版"""
    try:
        log_message("INFO", f"开始回测 {symbol}，回测天数: {days}，初始资金: {initial_balance}")
        
        # 获取历史数据
        df = get_historical_data(symbol, days)
        if df is None or len(df) < 100:
            log_message("WARNING", f"历史数据不足，无法回测 {symbol}")
            return None
        
        # 计算技术指标
        # VWAP(日内) + MACD(12,26,9) + RSI(14)
        df['macd'], df['macd_signal'], df['macd_histogram'] = calculate_macd(df['close'], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
        typical_price = (df['high'] + df['low'] + df['close']) / 3.0
        # 修复：明确生成列，避免使用 Series.name(None) 导致 KeyError
        df['vwap_value'] = typical_price * df['volume']
        df['day'] = df['timestamp'].dt.date
        df['cum_vwap'] = df.groupby('day')['vwap_value'].cumsum()
        df['cum_volume'] = df.groupby('day')['volume'].cumsum()
        safe_cum_volume = df['cum_volume'].replace(0, np.nan)
        df['VWAP'] = (df['cum_vwap'] / safe_cum_volume).ffill()
        df['RSI'] = calculate_rsi(df['close'], period=14)
        
        # 回测变量
        position = None
        entry_price = 0
        entry_time = None
        trades = []
        balance = initial_balance
        max_drawdown = 0
        peak_balance = initial_balance
        trade_details = []
        
        for i in range(1, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            
            # 更新最大回撤
            if balance > peak_balance:
                peak_balance = balance
            current_drawdown = (peak_balance - balance) / peak_balance * 100
            if current_drawdown > max_drawdown:
                max_drawdown = current_drawdown
            
            # 检查信号
            if pd.notna(current['macd']):
                # 做多信号
                if (prev['macd'] <= prev['macd_signal'] and 
                    current['macd'] > current['macd_signal'] and 
                    current['close'] > current['VWAP'] and 
                    current['RSI'] > 50 and position != 'long'):
                    
                    if position == 'short':
                        # 平空仓
                        pnl = (entry_price - current['close']) / entry_price * balance * 0.8
                        trades.append({
                            'type': 'close_short',
                            'price': current['close'],
                            'pnl': pnl,
                            'timestamp': current['timestamp']
                        })
                        trade_details.append({
                            'symbol': symbol,
                            'side': 'close_short',
                            'entry_price': entry_price,
                            'exit_price': current['close'],
                            'pnl': pnl,
                            'pnl_percentage': (pnl / balance * 100),
                            'entry_time': entry_time,
                            'exit_time': current['timestamp'],
                            'duration': (current['timestamp'] - entry_time).total_seconds() / 3600 if entry_time else 0
                        })
                        balance += pnl
                    
                    # 开多仓
                    position = 'long'
                    entry_price = current['close']
                    entry_time = current['timestamp']
                    trades.append({
                        'type': 'open_long',
                        'price': entry_price,
                        'timestamp': current['timestamp']
                    })
                
                # 做空信号
                elif (prev['macd'] >= prev['macd_signal'] and 
                      current['macd'] < current['macd_signal'] and 
                      current['close'] < current['VWAP'] and 
                      current['RSI'] < 50 and position != 'short'):
                    
                    if position == 'long':
                        # 平多仓
                        pnl = (current['close'] - entry_price) / entry_price * balance * 0.8
                        trades.append({
                            'type': 'close_long',
                            'price': current['close'],
                            'pnl': pnl,
                            'timestamp': current['timestamp']
                        })
                        trade_details.append({
                            'symbol': symbol,
                            'side': 'close_long',
                            'entry_price': entry_price,
                            'exit_price': current['close'],
                            'pnl': pnl,
                            'pnl_percentage': (pnl / balance * 100),
                            'entry_time': entry_time,
                            'exit_time': current['timestamp'],
                            'duration': (current['timestamp'] - entry_time).total_seconds() / 3600 if entry_time else 0
                        })
                        balance += pnl
                    
                    # 开空仓
                    position = 'short'
                    entry_price = current['close']
                    entry_time = current['timestamp']
                    trades.append({
                        'type': 'open_short',
                        'price': entry_price,
                        'timestamp': current['timestamp']
                    })
        
        # 计算回测结果
        closed_trades = [t for t in trades if 'pnl' in t]
        total_trades = len(closed_trades)
        winning_trades = len([t for t in closed_trades if t['pnl'] > 0])
        losing_trades = len([t for t in closed_trades if t['pnl'] < 0])
        total_pnl = sum([t['pnl'] for t in closed_trades])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        profit_factor = abs(sum([t['pnl'] for t in closed_trades if t['pnl'] > 0]) / 
                          sum([t['pnl'] for t in closed_trades if t['pnl'] < 0])) if losing_trades > 0 else float('inf')
        
        # 计算平均盈亏
        avg_win = sum([t['pnl'] for t in closed_trades if t['pnl'] > 0]) / winning_trades if winning_trades > 0 else 0
        avg_loss = sum([t['pnl'] for t in closed_trades if t['pnl'] < 0]) / losing_trades if losing_trades > 0 else 0
        
        backtest_result = {
            'symbol': symbol,
            'days': days,
            'initial_balance': initial_balance,
            'final_balance': balance,
            'total_return': ((balance - initial_balance) / initial_balance * 100),
            'return_rate': ((balance - initial_balance) / initial_balance * 100),
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown,
            'trade_details': trade_details,
            'trades': closed_trades
        }
        
        log_message("INFO", f"回测完成 {symbol}: {total_trades}笔交易，胜率{win_rate:.1f}%，收益率{backtest_result['total_return']:.2f}%，最大回撤{max_drawdown:.2f}%")
        return backtest_result
        
    except Exception as e:
        log_message("ERROR", f"策略回测失败 {symbol}: {e}")
        return None

def generate_backtest_report(backtest_results):
    """生成详细的回测报告"""
    try:
        if not backtest_results:
            return "暂无回测数据"
        
        report_lines = ["=== 策略回测报告 ===", ""]
        
        # 汇总统计（容错）
        total_symbols = len(backtest_results)
        total_trades = sum([r.get('total_trades', 0) for r in backtest_results if r])
        total_return = sum([r.get('total_return', 0.0) for r in backtest_results if r])
        avg_win_rate = (sum([r.get('win_rate', 0.0) for r in backtest_results if r]) / total_symbols) if total_symbols > 0 else 0.0
        
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
            symbols = SYMBOLS[:5]  # 默认回测前5个标的
        
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
                        'tdMode': TD_MODE,
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
        if strategy_type == 'bb_rsi_1m':
            # 1m BB+RSI窄带回归：SL=BB外0.5%，TP=BB中轨
            sltp = calculate_bb_rsi_1m_stops(symbol, entry_price, side)
            if not sltp:
                return False
            stop_loss = sltp['stop_loss']
            take_profit = sltp['take_profit']
            log_message("DEBUG", f"{symbol} 1m BB+RSI 止盈止损: 入场={entry_price:.4f}, SL={stop_loss:.4f}, TP={take_profit:.4f}")
        elif strategy_type == 'oscillation':
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
                        'tdMode': TD_MODE,
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
                        'tdMode': TD_MODE,
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
                        'tdMode': TD_MODE,
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

def place_bracket_orders(symbol, side, size, entry_price, atr_value=0):
    """同步挂条件止盈止损，记录订单ID到 order_tracking"""
    try:
        # 若为1m BB+RSI策略，优先使用专用SL/TP
        strategy_type = None
        try:
            pos = position_tracker['positions'].get(symbol)
            strategy_type = pos.get('strategy_type') if pos else None
        except Exception:
            strategy_type = None

        if strategy_type == 'bb_rsi_1m':
            sl_tp = calculate_bb_rsi_1m_stops(symbol, entry_price, side)
        else:
            sl_tp = calculate_stop_loss_take_profit(symbol, entry_price, side, atr_value or 0)

        if not sl_tp:
            return False
        pos_side = 'long' if side == 'long' else 'short'
        side_action = 'sell' if side == 'long' else 'buy'
        # 止损
        o_sl = exchange.create_order(
            symbol=symbol,
            type='conditional',
            side=side_action,
            amount=abs(size),
            price=sl_tp['stop_loss'],
            params={
                'slTriggerPx': sl_tp['stop_loss'],
                'slOrdPx': sl_tp['stop_loss'],
                'tdMode': TD_MODE,
                'posSide': pos_side,
                'reduceOnly': True
            }
        )
        # 止盈
        o_tp = exchange.create_order(
            symbol=symbol,
            type='conditional',
            side=side_action,
            amount=abs(size),
            price=sl_tp['take_profit'],
            params={
                'tpTriggerPx': sl_tp['take_profit'],
                'tpOrdPx': sl_tp['take_profit'],
                'tdMode': TD_MODE,
                'posSide': pos_side,
                'reduceOnly': True
            }
        )
        # 记录ID以便后续管理
        order_tracking[symbol] = {
            'last_setup_time': time.time(),
            'stop_loss': sl_tp['stop_loss'],
            'take_profit': sl_tp['take_profit'],
            'sl_id': (o_sl.get('id') if isinstance(o_sl, dict) else None),
            'tp_id': (o_tp.get('id') if isinstance(o_tp, dict) else None),
            'entry_price': entry_price,
        }
        log_message("INFO", f"{symbol} 同步挂条件单: SL={sl_tp['stop_loss']:.4f}, TP={sl_tp['take_profit']:.4f}")
        return True
    except Exception as e:
        log_message("ERROR", f"{symbol} 条件单提交失败: {e}")
        return False

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

def report_symbols_status():
    """输出每个交易对的详细状态：做多/做空/等待信号、价格、VWAP、RSI、未实现盈亏、SL/TP、敞口占比"""
    try:
        acct = get_account_info()
        total_balance = acct['total_balance'] if acct else 0.0
        for symbol in SYMBOLS:
            try:
                pos = position_tracker['positions'].get(symbol)
                pending = position_tracker.get('pending_signals', {}).get(symbol)
                status = 'idle'
                if pending:
                    status = f"pending({pending['signal']['side']})"
                if pos:
                    status = pos['side']

                # 最新价格与5m指标
                last = None; vwap = None; rsi = None
                try:
                    last = float(exchange.fetch_ticker(symbol)['last'])
                    ohlcv = get_klines(symbol, '5m', limit=60)
                    df = process_klines(ohlcv) if ohlcv else None
                    if df is not None and len(df) >= 20:
                        vwap = float(df['VWAP'].iloc[-1]) if 'VWAP' in df.columns else None
                        rsi = float(df['RSI'].iloc[-1]) if 'RSI' in df.columns else None
                except Exception:
                    pass

                # 未实现盈亏与敞口
                upnl = 0.0; exposure_pct = 0.0
                if pos and last is not None:
                    size = float(pos.get('size', 0))
                    entry = float(pos.get('entry_price', last))
                    upnl = (last - entry) * size if pos['side'] == 'long' else (entry - last) * size
                    if total_balance > 0:
                        exposure_pct = (abs(size) * last) / total_balance * 100.0

                # SL/TP（来自跟踪器或占位）
                sl = None; tp = None
                active = position_tracker.get('active_stop_orders', {}).get(symbol)
                if active:
                    sl = active.get('stop_loss'); tp = active.get('take_profit')

                # 状态行
                line = (
                    f"{symbol}: status={status}"
                    f", last={last:.4f}" if last is not None else f"{symbol}: status={status}, last=NA"
                )
                line += f", VWAP={vwap:.4f}" if vwap is not None else ", VWAP=NA"
                line += f", RSI={rsi:.1f}" if rsi is not None else ", RSI=NA"
                line += f", uPnL={upnl:.4f}"
                line += f", SL={sl:.4f}" if sl is not None else ", SL=NA"
                line += f", TP={tp:.4f}" if tp is not None else ", TP=NA"
                line += f", exposure={exposure_pct:.2f}%"
                log_message("INFO", line)
            except Exception as e:
                log_message("WARNING", f"{symbol} 状态报告失败: {e}")
    except Exception as e:
        log_message("ERROR", f"生成状态报告失败: {e}")

def enhanced_trading_loop():
    """增强版主交易循环"""
    try:
        log_message("SUCCESS", "开始增强版交易循环...")
        
        # 启动时运行全面回测
        log_message("INFO", "正在运行全面策略回测分析...")
        backtest_results = run_comprehensive_backtest(SYMBOLS[:5], days_list=[7, 14, 30])
        
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

                # 逐交易对状态报告
                report_symbols_status()
                
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
                                log_signal_overview(symbol, signal)
                                # 记录pending信号
                                position_tracker['pending_signals'][symbol] = {
                                    'signal': signal,
                                    'timestamp': datetime.now(timezone(timedelta(hours=8))),
                                    'kline_end_time': signal.get('kline_end_time', 0)
                                }
                                
                            elif signal.get('signal_strength') in ['strong', 'medium', 'weak']:
                                # 确认信号，执行交易
                                log_signal_overview(symbol, signal)
                                
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
# 平仓管理模块（日内 VWAP + MACD + RSI）
# =================================
def check_positions():
    """每分钟检查持仓并执行退出规则：VWAP反转、RSI极值分批/全平、MACD背离、时间止损、ATR追踪止损"""
    try:
        if not position_tracker['positions']:
            return
        for symbol, pos in list(position_tracker['positions'].items()):
            try:
                # 1m BB+RSI策略专用平仓规则（优先处理）
                if pos.get('strategy_type') == 'bb_rsi_1m':
                    ohlcv_1m = get_klines(symbol, '1m', limit=120)
                    if ohlcv_1m:
                        df1m = process_klines(ohlcv_1m)
                        if df1m is not None and len(df1m) >= 30 and 'BB_mid' in df1m.columns:
                            last_close_1m = float(df1m['close'].iloc[-1])
                            bb_mid_1m = float(df1m['BB_mid'].iloc[-1])
                            rsi_1m = float(df1m['RSI'].iloc[-1]) if 'RSI' in df1m.columns else None

                            # 平仓：动态回中轨0.3%阈值
                            mid_revert = (abs(last_close_1m - bb_mid_1m) / bb_mid_1m) < 0.003

                            # RSI极值反转：long RSI>70平，short RSI<30平
                            side = pos['side']
                            rsi_revert = (side == 'long' and rsi_1m is not None and rsi_1m > 70) or (side == 'short' and rsi_1m is not None and rsi_1m < 30)

                            # 时间>3min强制平
                            force_time = False
                            try:
                                et = pos.get('timestamp')
                                if et:
                                    from datetime import datetime as dt
                                    now_ = dt.now(timezone(timedelta(hours=8)))
                                    force_time = (now_ - et).total_seconds() > 180
                            except Exception:
                                pass

                            if mid_revert or rsi_revert or force_time:
                                reason = "BB中轨0.3%回归" if mid_revert else ("RSI极值反转" if rsi_revert else "超过3分钟强制平")
                                close_position(symbol, f"1m BB+RSI {reason}")
                                # 进入下一循环项
                                continue

                # 拉取最新5m数据
                ohlcv = get_klines(symbol, '5m', limit=120)
                if not ohlcv:
                    continue
                df = process_klines(ohlcv)
                if df is None or len(df) < 50:
                    continue

                last_close = float(df['close'].iloc[-1])
                last_open = float(df['open'].iloc[-1])
                is_bullish = last_close > last_open
                is_bearish = last_close < last_open

                vwap = float(df['VWAP'].iloc[-1]) if 'VWAP' in df.columns else None
                rsi = float(df['RSI'].iloc[-1]) if 'RSI' in df.columns else None
                atr = float(df['ATR_14'].iloc[-1]) if 'ATR_14' in df.columns else None
                # 震荡平仓：价格回到BB中轨±0.5% 或 15m反转
                try:
                    bb_mid = float(df['BB_mid'].iloc[-1]) if 'BB_mid' in df.columns else None
                    # 15m反转确认
                    macd15_death = False
                    macd15_golden = False
                    try:
                        ohlcv_c = get_klines(symbol, TIMEFRAME_CONFIRM, limit=50)
                        if ohlcv_c:
                            df_c = pd.DataFrame(ohlcv_c, columns=['timestamp','open','high','low','close','volume'])
                            macd_c, sig_c, _ = calculate_macd(df_c['close'], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
                            macd15_death = (macd_c.iloc[-2] >= sig_c.iloc[-2] and macd_c.iloc[-1] < sig_c.iloc[-1])
                            macd15_golden = (macd_c.iloc[-2] <= sig_c.iloc[-2] and macd_c.iloc[-1] > sig_c.iloc[-1])
                    except Exception:
                        pass
                    bb_mid_hit = (bb_mid and (abs(last_close - bb_mid) / bb_mid < 0.005))
                    reversal15 = (pos['side'] == 'long' and macd15_death) or (pos['side'] == 'short' and macd15_golden)
                    if bb_mid_hit or reversal15:
                        exit_now = True
                        log_message("INFO", f"{symbol} 平仓触发：{'BB中轨' if bb_mid_hit else '15m反转'} 全平")
                except Exception:
                    pass

                # MACD背离检测（简化版）：价格创新高/低但DIFF未同步创新高/低，幅度>5%
                def has_bearish_divergence(series_price, series_diff):
                    try:
                        p1 = float(series_price.iloc[-3]); p2 = float(series_price.iloc[-1])
                        d1 = float(series_diff.iloc[-3]); d2 = float(series_diff.iloc[-1])
                        return (p2 > p1) and (d2 < d1) and ((p2 - p1) / p1 >= 0.05)
                    except:
                        return False
                def has_bullish_divergence(series_price, series_diff):
                    try:
                        p1 = float(series_price.iloc[-3]); p2 = float(series_price.iloc[-1])
                        d1 = float(series_diff.iloc[-3]); d2 = float(series_diff.iloc[-1])
                        return (p2 < p1) and (d2 > d1) and ((p1 - p2) / p1 >= 0.05)
                    except:
                        return False

                macd_diff = df['MACD'] if 'MACD' in df.columns else None
                price_series = df['close']

                side = pos['side']  # 'long' or 'short'
                entry_price = float(pos.get('entry_price', last_close))
                size = float(pos.get('size', 0))
                entry_time = pos.get('timestamp')

                # 计算未实现盈亏与追踪阈值
                unrealized_pnl = (last_close - entry_price) * size if side == 'long' else (entry_price - last_close) * size
                trailing_trigger = (atr if atr and atr > 0 else 0)  # >=1x ATR 启用追踪
                trailing_distance = 0.5 * atr if atr and atr > 0 else None

                # 退出条件集合
                exit_now = False
                exit_partial = False
                partial_ratio = 0.5

                # 1) VWAP反转：趋势失效立即全平
                if vwap is not None:
                    if side == 'long' and last_close < vwap:
                        exit_now = True
                        log_message("INFO", f"{symbol} 多仓VWAP反转: close<{vwap:.4f} 全平")
                    if side == 'short' and last_close > vwap:
                        exit_now = True
                        log_message("INFO", f"{symbol} 空仓VWAP反转: close>{vwap:.4f} 全平")

                # 2) RSI极值：分批/全平（长>80全平，>70半平；短<20全平，<30半平）
                if rsi is not None:
                    if side == 'long':
                        if rsi > 80:
                            exit_now = True
                            log_message("INFO", f"{symbol} 多仓RSI>80 全平")
                        elif rsi > 70:
                            exit_partial = True
                            partial_ratio = 0.5
                            log_message("INFO", f"{symbol} 多仓RSI>70 平50%")
                    else:
                        if rsi < 20:
                            exit_now = True
                            log_message("INFO", f"{symbol} 空仓RSI<20 全平")
                        elif rsi < 30:
                            exit_partial = True
                            partial_ratio = 0.5
                            log_message("INFO", f"{symbol} 空仓RSI<30 平50%")

                # 3) MACD背离：幅度>5% 全平
                if macd_diff is not None:
                    if side == 'long' and has_bearish_divergence(price_series, macd_diff):
                        exit_now = True
                        log_message("INFO", f"{symbol} 多仓熊背离 全平")
                    if side == 'short' and has_bullish_divergence(price_series, macd_diff):
                        exit_now = True
                        log_message("INFO", f"{symbol} 空仓牛背离 全平")

                # 4) 时间止损：>120分钟无新高/低退出（更稳健）
                try:
                    from datetime import datetime as dt
                    now = dt.now(timezone(timedelta(hours=8)))
                    if entry_time and (now - entry_time).total_seconds() > 120 * 60:
                        exit_now = True
                        log_message("INFO", f"{symbol} 持仓超过120分钟 时间止损全平")
                except Exception:
                    pass

                # 5) Breakeven保护与分批止盈
                # Breakeven: 盈利达到0.5%后，回落到入场价则全平
                try:
                    if side == 'long':
                        if (last_close - entry_price) / entry_price >= 0.005 and not pos.get('breakeven_active'):
                            pos['breakeven_active'] = True
                        if pos.get('breakeven_active') and last_close <= entry_price:
                            exit_now = True
                            log_message("INFO", f"{symbol} 多仓回落至保本位 全平")
                    else:
                        if (entry_price - last_close) / entry_price >= 0.005 and not pos.get('breakeven_active'):
                            pos['breakeven_active'] = True
                        if pos.get('breakeven_active') and last_close >= entry_price:
                            exit_now = True
                            log_message("INFO", f"{symbol} 空仓回升至保本位 全平")
                except:
                    pass

                # 分批TP：盈利达到1x ATR先平50%
                if atr and atr > 0:
                    if side == 'long' and (last_close - entry_price) >= atr and not pos.get('partial1_done'):
                        exit_partial = True; partial_ratio = 0.5
                        pos['partial1_done'] = True
                        log_message("INFO", f"{symbol} 多仓达到1x ATR 盈利 先平50%")
                    if side == 'short' and (entry_price - last_close) >= atr and not pos.get('partial1_done'):
                        exit_partial = True; partial_ratio = 0.5
                        pos['partial1_done'] = True
                        log_message("INFO", f"{symbol} 空仓达到1x ATR 盈利 先平50%")

                # MACD柱峰值回落：平出30%
                try:
                    hist = df['MACD_HIST'] if 'MACD_HIST' in df.columns else None
                    if hist is not None and len(hist) >= 3:
                        if side == 'long' and hist.iloc[-1] < hist.iloc[-2] and hist.iloc[-2] > hist.iloc[-3] and not pos.get('partial_hist_done'):
                            exit_partial = True; partial_ratio = 0.3
                            pos['partial_hist_done'] = True
                            log_message("INFO", f"{symbol} 多仓MACD柱峰回落 平30%")
                        if side == 'short' and hist.iloc[-1] > hist.iloc[-2] and hist.iloc[-2] < hist.iloc[-3] and not pos.get('partial_hist_done'):
                            exit_partial = True; partial_ratio = 0.3
                            pos['partial_hist_done'] = True
                            log_message("INFO", f"{symbol} 空仓MACD柱峰回升 平30%")
                except:
                    pass

                # DOGE高噪保护：多仓RSI<30立即全平
                if symbol.startswith('DOGE') and side == 'long' and rsi is not None and rsi < 30:
                    exit_now = True
                    log_message("INFO", f"{symbol} DOGE 多仓RSI<30 噪声保护 全平")

                # 执行退出
                if exit_now or exit_partial:
                    close_ratio = 1.0 if exit_now else partial_ratio
                    close_size = max(size * close_ratio, 0)
                    if close_size <= 0:
                        continue
                    side_out = 'sell' if side == 'long' else 'buy'
                    try:
                        order = exchange.create_order(
                            symbol,
                            'market',
                            side_out,
                            close_size,
                            None,
                            {'tdMode': TD_MODE, 'posSide': 'long' if side == 'long' else 'short'}
                        )
                        if order:
                            exit_price = last_close
                            pnl = (exit_price - entry_price) * close_size if side == 'long' else (entry_price - exit_price) * close_size
                            log_message("SUCCESS", f"{symbol} 平仓成功: {side_out} {close_size} @ {exit_price:.4f}, PnL={pnl:.4f}")
                            # 更新统计
                            update_trade_stats(symbol, side, pnl, entry_price, exit_price)
                            # 更新/移除持仓
                            remain_size = size - close_size
                            if remain_size <= 0:
                                del position_tracker['positions'][symbol]
                            else:
                                position_tracker['positions'][symbol]['size'] = remain_size
                                position_tracker['positions'][symbol]['entry_price'] = exit_price  # 重新计算基准
                    except Exception as e:
                        log_message("ERROR", f"{symbol} 平仓失败: {e}")

            except Exception as e:
                log_message("ERROR", f"{symbol} 检查持仓失败: {e}")
    except Exception as e:
        log_message("ERROR", f"check_positions运行失败: {e}")

# =================================
# 回测模块（统一为5m，使用相同入场/平仓规则）
# =================================
def backtest_strategy_5m(symbol, days=14):
    """5m回测：VWAP+MACD+RSI 入场与平仓规则，返回交易记录与统计"""
    try:
        # 历史K线数量（5m，每天288根）
        limit = max(300, days * 288)
        ohlcv = get_klines(symbol, '5m', limit=limit)
        if not ohlcv:
            return {'symbol': symbol, 'trades': [], 'stats': {}}

        df = process_klines(ohlcv)
        if df is None or len(df) < 100:
            return {'symbol': symbol, 'trades': [], 'stats': {}}

        trades = []
        position = None  # {'side','entry_price','size','entry_time'}
        equity = 10000.0
        size_per_trade = 1000.0  # 模拟每次名义资金
        for i in range(20, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i - 1]

            # 指标
            vwap = row['VWAP']
            rsi = row['RSI']
            macd = row['MACD']; macd_sig = row['MACD_SIGNAL']
            vol_ma20 = df['vol_ma20'].iloc[i] if 'vol_ma20' in df.columns else None
            volume_ok = (vol_ma20 is None) or (row['volume'] >= 0.7 * vol_ma20)
            close = row['close']; open_ = row['open']
            vwap_bias = abs(close - vwap) / vwap > 0.001
            is_bullish = close > open_; is_bearish = close < open_
            atr = row['ATR_14'] if 'ATR_14' in df.columns else None

            # 交叉
            golden = (prev['MACD'] <= prev['MACD_SIGNAL']) and (macd > macd_sig)
            death = (prev['MACD'] >= prev['MACD_SIGNAL']) and (macd < macd_sig)

            # 平仓逻辑（若有持仓）
            if position:
                side = position['side']; entry_price = position['entry_price']
                # VWAP反转
                vwap_exit = (side == 'long' and close < vwap) or (side == 'short' and close > vwap)
                # RSI极值
                rsi_exit_full = (side == 'long' and rsi > 80) or (side == 'short' and rsi < 20)

                # 简化MACD背离：比较最近3根
                def bearish_div(i):
                    try:
                        p2 = df['close'].iloc[i]; p1 = df['close'].iloc[i-2]
                        d2 = df['MACD'].iloc[i]; d1 = df['MACD'].iloc[i-2]
                        return (p2 > p1) and (d2 < d1) and ((p2 - p1) / p1 >= 0.05)
                    except: return False
                def bullish_div(i):
                    try:
                        p2 = df['close'].iloc[i]; p1 = df['close'].iloc[i-2]
                        d2 = df['MACD'].iloc[i]; d1 = df['MACD'].iloc[i-2]
                        return (p2 < p1) and (d2 > d1) and ((p1 - p2) / p1 >= 0.05)
                    except: return False

                div_exit = (side == 'long' and bearish_div(i)) or (side == 'short' and bullish_div(i))

                # ATR追踪：盈利>1x ATR后，回撤>0.5x ATR退出
                trailing_exit = False
                if atr and atr > 0:
                    if side == 'long' and (close - entry_price) >= atr:
                        recent_high = float(df['high'].iloc[max(0, i-12):i+1].max())
                        trailing_exit = (recent_high - close) >= (0.5 * atr)
                    if side == 'short' and (entry_price - close) >= atr:
                        recent_low = float(df['low'].iloc[max(0, i-12):i+1].min())
                        trailing_exit = (close - recent_low) >= (0.5 * atr)

                if vwap_exit or rsi_exit_full or div_exit or trailing_exit:
                    # 退出
                    exit_price = close
                    pnl = (exit_price - entry_price) if side == 'long' else (entry_price - exit_price)

                    # Funding费用模拟：0.01%/8h，按持仓时长线性扣减
                    try:
                        funding_rate = 0.0001  # 0.01%
                        # 计算持仓时长（小时）
                        et = entry_time
                        if not isinstance(et, (pd.Timestamp, datetime)):
                            et = pd.to_datetime(et) if et is not None else current['timestamp']
                        hold_hours = (current['timestamp'] - et).total_seconds() / 3600.0 if et else 0.0
                        size = 1.0  # 简化按单位仓位扣减，可根据实际 size/equity 替换
                        pnl -= funding_rate * (hold_hours / 8.0) * size
                    except Exception:
                        pass

                    pnl *= DEFAULT_LEVERAGE  # 杠杆模拟
                    fee_rate = 0.0005  # 每侧手续费（约0.05%）
                    slippage_rate = 0.0005  # 滑点成本（约0.05%）
                    net_ret = (pnl / entry_price) - (fee_rate * 2) - slippage_rate
                    equity += net_ret * size_per_trade
                    trades.append({'side': side, 'entry': entry_price, 'exit': exit_price, 'pnl': pnl})
                    position = None
                    continue

            # 入场逻辑（仅在无持仓）
            if not position and volume_ok:
                if golden and (close > vwap) and (rsi > 40) and vwap_bias and is_bullish:
                    position = {'side': 'long', 'entry_price': close, 'entry_time': df['timestamp'].iloc[i]}
                elif death and (close < vwap) and (rsi < 60) and vwap_bias and is_bearish:
                    position = {'side': 'short', 'entry_price': close, 'entry_time': df['timestamp'].iloc[i]}

        # 统计
        wins = sum(1 for t in trades if t['pnl'] > 0)
        total_pnl = sum(t['pnl'] for t in trades)
        win_rate = (wins / len(trades) * 100) if trades else 0.0
        stats = {'symbol': symbol, 'trades_count': len(trades), 'win_rate': win_rate, 'total_pnl': total_pnl, 'equity': equity}
        return {'symbol': symbol, 'trades': trades, 'stats': stats}
    except Exception as e:
        log_message("ERROR", f"{symbol} 回测失败: {e}")
        return {'symbol': symbol, 'trades': [], 'stats': {}}

# =================================
# 主程序入口 - 增强版
# =================================
if __name__ == "__main__":
    # 使用增强版交易循环替代原版本
    try:
        # 初始化交易所连接
        exchange = initialize_exchange()
        if not exchange:
            log_message("ERROR", "交易所初始化失败")
            exit(1)
        
        # 启动不强制进行API连接测试，避免因短暂网络问题退出
        # if not test_api_connection():
        #     log_message("ERROR", "API连接测试失败，请检查配置")
        #     exit(1)
        
        # 显示启动信息
        log_message("SUCCESS", "VWAP+MACD(12,26,9)+RSI(14)策略交易系统启动成功")
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