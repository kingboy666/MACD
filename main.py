import ccxt
import pandas as pd
# import pandas_ta as ta  # 暂时注释掉，使用自定义指标计算

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
import traceback
import numpy as np
from datetime import datetime, timedelta
import time
import json
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ============================================
# API配置 - 从环境变量获取
# ============================================
def get_okx_config():
    """从环境变量获取OKX API配置"""
    config = {
        'api_key': os.getenv('OKX_API_KEY'),
        'secret_key': os.getenv('OKX_SECRET_KEY'),
        'passphrase': os.getenv('OKX_PASSPHRASE')
    }
    
    # 调试信息：打印环境变量状态（不显示实际值）
    print(f"环境变量检查:")
    print(f"  OKX_API_KEY: {'已设置' if config['api_key'] else '未设置'}")
    print(f"  OKX_SECRET_KEY: {'已设置' if config['secret_key'] else '未设置'}")
    print(f"  OKX_PASSPHRASE: {'已设置' if config['passphrase'] else '未设置'}")
    
    return config

# ============================================
# 交易所初始化 (OKX)
# ============================================
def initialize_exchange():
    """初始化OKX交易所连接"""
    try:
        # 先检查 .env 文件是否存在
        env_file_path = '.env'
        if os.path.exists(env_file_path):
            print(f"✅ 找到 .env 文件: {env_file_path}")
        else:
            print(f"⚠️  未找到 .env 文件: {env_file_path}")
        
        # 打印所有环境变量（仅用于调试）
        print("所有环境变量列表:")
        for key in os.environ:
            if 'OKX' in key.upper():
                print(f"  {key}: {'已设置' if os.environ[key] else '空值'}")
        
        config = get_okx_config()
        
        if not all([config['api_key'], config['secret_key'], config['passphrase']]):
            print("❌ 未找到API配置!")
            print("请设置以下环境变量:")
            print("  - OKX_API_KEY")
            print("  - OKX_SECRET_KEY")
            print("  - OKX_PASSPHRASE")
            print("
在 Railway 中设置环境变量的步骤:")
            print("1. 进入你的 Railway 项目")
            print("2. 点击 'Variables' 标签")
            print("3. 添加以下三个变量:")
            print("   - Name: OKX_API_KEY, Value: 你的API密钥")
            print("   - Name: OKX_SECRET_KEY, Value: 你的密钥")
            print("   - Name: OKX_PASSPHRASE, Value: 你的密码短语")
            print("4. 点击 'Deploy' 重新部署")
            raise ValueError("请先设置OKX API配置")
        
        exchange = ccxt.okx({
            'apiKey': config['api_key'],
            'secret': config['secret_key'],
            'password': config['passphrase'],
            'sandbox': False,  # 实盘交易
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap',  # 永续合约
            }
        })
        
        print("✅ OKX交易所连接初始化成功")
        return exchange
        
    except Exception as e:
        print(f"❌ 交易所初始化失败: {str(e)}")
        raise

# 初始化缺失的全局变量
klines_cache = {}
cooldown_symbols = {}
timeframe_1h = '1h'
exchange = None  # 延迟初始化，在启动时设置

# ============================================
# 全局配置常量
# ============================================
MAX_LEVERAGE_BTC_ETH = 50
MAX_LEVERAGE_OTHERS = 30
DEFAULT_LEVERAGE = 20
RISK_PER_TRADE = 0.1
MIN_TRADE_AMOUNT_USD = 1
MAX_OPEN_POSITIONS = 5
COOLDOWN_PERIOD = 5 * 60
ATR_PERIOD = 14
MIN_ATR_PERCENTAGE = 0.005
MAX_ATR_PERCENTAGE = 0.10
MAX_DAILY_TRADES = 20
MAX_DAILY_LOSS = 0.05

# MACD指标配置
MACD_FAST = 6                             # MACD快线周期
MACD_SLOW = 32                            # MACD慢线周期
MACD_SIGNAL = 9                           # MACD信号线周期

# 止损止盈配置
FIXED_SL_PERCENTAGE = 0.02
FIXED_TP_PERCENTAGE = 0.04
MAX_SL_PERCENTAGE = 0.03

# 移动止盈止损配置
TRAILING_STOP_ACTIVATION_PERCENTAGE = 0.01  # 价格移动1%后激活移动止损
TRAILING_STOP_CALLBACK_PERCENTAGE = 0.005   # 回调0.5%触发止损
TRAILING_STOP_CHECK_INTERVAL = 60           # 每60秒检查一次移动止损条件

# 服务器状态检测配置
SERVER_CHECK_INTERVAL = 300                 # 每5分钟检查一次服务器状态
MAX_SERVER_CHECK_FAILURES = 3               # 连续失败3次判定为掉线

# ADX指标配置
ADX_PERIOD = 14                            # ADX计算周期
ADX_THRESHOLD_LOW = 20                     # ADX低于此值视为震荡市场
ADX_THRESHOLD_HIGH = 25                    # ADX高于此值视为趋势市场

# ============================================
# 全局变量
# ============================================
# 交易统计
trade_stats = {
    'initial_balance': 0,
    'current_balance': 0,
    'total_trades': 0,
    'winning_trades': 0,
    'losing_trades': 0,
    'total_profit_loss': 0,
    'daily_trades': 0,
    'daily_pnl': 0,
    'last_reset_date': datetime.now().strftime('%Y-%m-%d'),
    'daily_reset_time': datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
}

# 持仓跟踪器
position_tracker = {
    'positions': {},
    'last_update': datetime.now()
}

# 最新信号
latest_signals = {}

# 冷却期
cooldown_tracker = {}

# 服务器状态跟踪
server_status = {
    'is_online': True,
    'check_failures': 0,
    'last_check_time': time.time()
}

# 移动止损跟踪
trailing_stops = {}

# ============================================
# 日常重置函数
# ============================================
def check_daily_reset():
    """检查是否需要重置每日统计"""
    now = datetime.now()
    if now.date() > trade_stats['daily_reset_time'].date():
        log_message("INFO", f"每日统计重置 - 昨日交易: {trade_stats['daily_trades']}, 昨日盈亏: {trade_stats['daily_pnl']:.2f} USDT")
        
# ============================================
# 日志功能
# ============================================
def log_message(level, message):
    """增强的日志功能"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] [{level.upper()}] {message}")

# ============================================
# API连接测试
# ============================================
def test_api_connection():
    """测试交易所API连接"""
    try:
        exchange.fetch_balance()
        log_message("SUCCESS", "API连接测试成功")
        return True
    except Exception as e:
        log_message("ERROR", f"API连接测试失败: {str(e)}")
        return False

# ============================================
# 获取K线数据
# ============================================
def get_klines(symbol, timeframe, limit=100):
    """获取K线数据，带缓存机制和重试"""
    try:
        cache_key = f"{symbol}_{timeframe}"
        if cache_key in klines_cache:
            cached_data, fetch_time = klines_cache[cache_key]
            cache_duration = 60 if timeframe == '1m' else 3600 if timeframe == '1h' else 300
            if (time.time() - fetch_time) < cache_duration:
                return cached_data

        # 重试机制
        max_retries = 3
        for attempt in range(max_retries):
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                if ohlcv is not None and len(ohlcv) > 0:
                    klines_cache[cache_key] = (ohlcv, time.time())
                    return ohlcv
                else:
                    log_message("WARNING", f"{symbol} {timeframe} 获取到空的K线数据，尝试 {attempt + 1}/{max_retries}")
            except Exception as retry_e:
                log_message("WARNING", f"获取 {symbol} {timeframe} K线数据第{attempt + 1}次尝试失败: {str(retry_e)}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # 等待1秒后重试
                else:
                    raise retry_e
        
        return None
        
    except Exception as e:
        log_message("ERROR", f"获取 {symbol} {timeframe} K线数据最终失败: {str(e)}")
        return None

# ============================================
# 获取账户信息
# ============================================
def get_account_info():
    """获取账户信息"""
    try:
        balance = exchange.fetch_balance()
        usdt_balance = balance.get('USDT', {})
        free_balance = usdt_balance.get('free', 0)
        used_balance = usdt_balance.get('used', 0)
        total_balance = usdt_balance.get('total', 0)
        
        unrealized_pnl = 0
        try:
            positions = exchange.fetch_positions()
            for position in positions:
                if position['contracts'] > 0:
                    unrealized_pnl += float(position.get('unrealizedPnl', 0))
        except Exception as e:
            log_message("WARNING", f"获取未实现盈亏失败: {str(e)}")
        
        account_info = {
            'free_balance': free_balance,
            'used_balance': used_balance,
            'total_balance': total_balance,
            'unrealized_pnl': unrealized_pnl,
            'available_balance': free_balance,
            'equity': total_balance + unrealized_pnl
        }
        
        return account_info
        
    except Exception as e:
        log_message("ERROR", f"获取账户信息失败: {str(e)}")
        return None

# ============================================
# 计算仓位大小
# ============================================
def calculate_position_size(account_info, symbol, price, stop_loss, risk_ratio):
    """智能仓位计算 - 考虑多仓位资金分配"""
    try:
        # 获取账户信息
        total_balance = account_info.get('total_balance', 0)
        available_balance = account_info.get('available_balance', 0)
        
        if total_balance <= 0:
            log_message("ERROR", f"账户总额为0，无法计算仓位大小")
            return 0
        
        # 计算当前持仓占用的资金
        current_positions_value = 0
        active_positions_count = len(position_tracker['positions'])
        
        for pos_symbol, pos_data in position_tracker['positions'].items():
            if 'entry_price' in pos_data and 'size' in pos_data:
                pos_value = pos_data['entry_price'] * pos_data['size'] / DEFAULT_LEVERAGE
                current_positions_value += pos_value
        
        # 计算总可用交易资金（账户总额的80%）
        total_trading_fund = total_balance * 0.8
        
        # 计算剩余可用资金
        remaining_fund = total_trading_fund - current_positions_value
        
        log_message("INFO", f"账户总额: {total_balance:.2f} USDT")
        log_message("INFO", f"总交易资金(80%): {total_trading_fund:.2f} USDT")
        log_message("INFO", f"当前持仓占用: {current_positions_value:.2f} USDT")
        log_message("INFO", f"剩余可用资金: {remaining_fund:.2f} USDT")
        log_message("INFO", f"当前持仓数量: {active_positions_count}")
        
        # 检查是否还有足够资金开新仓
        if remaining_fund <= 0:
            log_message("WARNING", f"剩余资金不足，无法开新仓位")
            return 0
        
        # 智能资金分配策略 - 适应各种账户大小
        if active_positions_count == 0:
            # 第一个仓位：使用较大比例的资金
            if total_trading_fund >= 100:  # 大账户
                position_fund = min(remaining_fund * 0.6, total_trading_fund * 0.3)
            else:  # 小账户，使用更大比例
                position_fund = remaining_fund * 0.8
        elif active_positions_count < 3:
            # 前3个仓位：根据账户大小调整分配
            if total_trading_fund >= 100:
                max_new_positions = min(3 - active_positions_count, 2)
                position_fund = remaining_fund / (max_new_positions + 1)
            else:
                # 小账户，平均分配剩余资金
                position_fund = remaining_fund * 0.5
        else:
            # 超过3个仓位：使用较小资金
            if total_trading_fund >= 100:
                position_fund = min(remaining_fund * 0.2, total_trading_fund * 0.1)
            else:
                position_fund = remaining_fund * 0.3
        
        # 对于大账户，限制单个仓位不超过总资金的25%
        if total_trading_fund >= 100:
            max_single_position = total_trading_fund * 0.25
            position_fund = min(position_fund, max_single_position)
        
        # 确保不超过剩余资金
        position_fund = min(position_fund, remaining_fund)
        
        # 对于极小账户的特殊处理
        if position_fund < 1 and remaining_fund >= 1:
            position_fund = min(remaining_fund, 1)
        
        # 最终检查
        if position_fund <= 0:
            log_message("WARNING", f"计算的交易资金为0，无法开仓")
            return 0
        
        log_message("INFO", f"本次交易分配资金: {position_fund:.2f} USDT")
        
        # 计算仓位大小（考虑杠杆）
        position_value_with_leverage = position_fund * DEFAULT_LEVERAGE
        position_size = position_value_with_leverage / price
        
        log_message("INFO", f"杠杆后仓位价值: {position_value_with_leverage:.2f} USDT")
        log_message("INFO", f"计算仓位大小: {position_size:.6f}")
        
        # 检查交易所最小交易量要求
        try:
            markets = exchange.load_markets()
            market = markets.get(symbol)
            
            if market:
                min_amount = market.get('limits', {}).get('amount', {}).get('min', 0.001)
                log_message("INFO", f"{symbol} 最小交易量: {min_amount}")
                
                if position_size < min_amount:
                    log_message("WARNING", f"计算仓位{position_size:.6f}小于最小交易量{min_amount}")
                    
                    # 动态计算所需杠杆以满足最小交易量
                    min_position_value = min_amount * price
                    required_leverage = min_position_value / position_fund
                    
                    log_message("INFO", f"最小仓位价值: {min_position_value:.2f} USDT")
                    log_message("INFO", f"当前分配资金: {position_fund:.2f} USDT")
                    log_message("INFO", f"需要杠杆倍数: {required_leverage:.1f}x")
                    
                    # 检查是否在合理杠杆范围内（最大100倍）
                    if required_leverage <= 100:
                        # 使用最小交易量
                        position_size = min_amount
                        actual_leverage = required_leverage
                        actual_fund_used = min_position_value / actual_leverage
                        
                        log_message("INFO", f"调整为最小交易量: {position_size}")
                        log_message("INFO", f"实际使用杠杆: {actual_leverage:.1f}x")
                        log_message("INFO", f"实际使用资金: {actual_fund_used:.2f} USDT")
                    else:
                        # 即使100倍杠杆也不够，检查是否可以用更多资金
                        max_affordable_fund = min_position_value / 100  # 100倍杠杆下的最小资金
                        
                        if total_trading_fund >= max_affordable_fund:
                            # 使用更多资金来满足最小交易量
                            position_size = min_amount
                            actual_fund_used = max_affordable_fund
                            log_message("INFO", f"使用100倍杠杆，调整资金为: {actual_fund_used:.2f} USDT")
                            log_message("INFO", f"调整为最小交易量: {position_size}")
                        else:
                            # 真的资金不足
                            log_message("ERROR", f"即使100倍杠杆也需要{max_affordable_fund:.2f} USDT，但总资金只有{total_trading_fund:.2f} USDT")
                            return 0
        except Exception as e:
            log_message("WARNING", f"获取市场信息失败: {e}")
            # 使用保守的默认最小值
            if position_size < 0.01:  # 提高默认最小值到0.01
                if (0.01 * price) / DEFAULT_LEVERAGE <= total_trading_fund:
                    position_size = 0.01
                    log_message("INFO", f"使用默认最小交易量: {position_size}")
                else:
                    log_message("ERROR", f"资金不足以满足默认最小交易量要求")
                    return 0
        
        # 最终验证
        final_trade_value = (position_size * price) / DEFAULT_LEVERAGE
        log_message("INFO", f"最终交易价值: {final_trade_value:.2f} USDT")
        log_message("INFO", f"最终仓位大小: {position_size:.6f}")
        
        return position_size
        
    except Exception as e:
        log_message("ERROR", f"仓位计算失败: {e}")
        return 0

# ============================================
# 处理K线数据并计算指标
# ============================================
def process_klines(ohlcv):
    """处理K线数据并计算技术指标"""
    try:
        # 检查输入数据
        if ohlcv is None:
            log_message("ERROR", "K线数据为空(None)")
            return None
        
        if not isinstance(ohlcv, list) or len(ohlcv) == 0:
            log_message("ERROR", f"K线数据格式错误或为空: {type(ohlcv)}, 长度: {len(ohlcv) if hasattr(ohlcv, '__len__') else 'N/A'}")
            return None
        
        # MACD(6,32,9)需要至少32+9=41根K线，为了安全起见要求50根
        min_required = max(MACD_SLOW + MACD_SIGNAL, 50)
        if len(ohlcv) < min_required:
            log_message("ERROR", f"K线数据不足，只有{len(ohlcv)}根，MACD({MACD_FAST},{MACD_SLOW},{MACD_SIGNAL})至少需要{min_required}根")
            return None
        
        # 创建DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # 检查数据完整性
        if df.empty:
            log_message("ERROR", "创建的DataFrame为空")
            return None
        
        # 检查必要的列是否存在且有数据
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if df[col].isna().all():
                log_message("ERROR", f"列 {col} 全部为空值")
                return None
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # 计算MACD指标
        try:
            macd_line, signal_line, histogram = calculate_macd(df['close'], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
            df['MACD'] = macd_line
            df['MACD_SIGNAL'] = signal_line
            df['MACD_HIST'] = histogram
        except Exception as macd_e:
            log_message("ERROR", f"MACD计算失败: {str(macd_e)}")
            return None
        
        # 计算ATR（仅用于止盈止损）
        try:
            df['ATR_14'] = calculate_atr(df['high'], df['low'], df['close'], period=ATR_PERIOD)
        except Exception as atr_e:
            log_message("WARNING", f"ATR计算失败: {str(atr_e)}")
            df['ATR_14'] = None
        
        # 计算ADX（用于趋势/震荡判断）
        try:
            df['ADX'] = calculate_adx(df['high'], df['low'], df['close'], period=ADX_PERIOD)
        except Exception as adx_e:
            log_message("WARNING", f"ADX计算失败: {str(adx_e)}，使用默认值")
            df['ADX'] = 25  # 使用默认值
        
        return df
        
    except Exception as e:
        log_message("ERROR", f"处理K线数据失败: {str(e)}")
        import traceback
        log_message("ERROR", f"详细错误信息: {traceback.format_exc()}")
        return None

# ============================================
# 生成交易信号 - MACD金叉/死叉 + ADX过滤
# ============================================
def generate_signal(symbol):
    """基于MACD金叉/死叉生成交易信号，使用ADX过滤震荡市场"""
    try:
        # 使用30分钟时间周期
        # 确保有足够的K线数据计算MACD(6,32,9)
        min_required = max(MACD_SLOW + MACD_SIGNAL, 50)
        ohlcv = get_klines(symbol, '30m', limit=max(100, min_required + 10))
        if ohlcv is None or len(ohlcv) < min_required:
            log_message("WARNING", f"{symbol} 获取K线数据失败或数据不足，需要{min_required}根，实际{len(ohlcv) if ohlcv else 0}根")
            return None, 0
        
        df = process_klines(ohlcv)
        if df is None:
            return None, 0
        
        # 检查是否成功计算了指标
        if df['MACD'].isna().all() or df['MACD_SIGNAL'].isna().all():
            log_message("WARNING", f"{symbol} MACD指标计算失败")
            return None, 0
        
        current_macd = df['MACD'].iloc[-1]
        current_signal = df['MACD_SIGNAL'].iloc[-1]
        prev_macd = df['MACD'].iloc[-2]
        prev_signal = df['MACD_SIGNAL'].iloc[-2]
        
        # === ADX趋势/震荡判断 ===
        adx_value = df['ADX'].iloc[-1] if 'ADX' in df.columns and not df['ADX'].isna().all() else 0
        is_trending = adx_value > ADX_THRESHOLD_HIGH
        is_ranging = adx_value < ADX_THRESHOLD_LOW
        
        # 如果ADX低于阈值，市场处于震荡状态，不产生信号
        if is_ranging:
            log_message("DEBUG", f"{symbol} ADX值为 {adx_value:.2f}，低于{ADX_THRESHOLD_LOW}，震荡市场，不产生信号")
            return None, 0
        
        # ATR波动率过滤（仅用于避免过度波动的市场）
        if not df['ATR_14'].isna().all():
            current_close = df['close'].iloc[-1]
            atr_value = df['ATR_14'].iloc[-1]
            atr_percentage = atr_value / current_close
            
            if atr_percentage > MAX_ATR_PERCENTAGE:
                log_message("DEBUG", f"{symbol} ATR波动率过高 ({atr_percentage:.4f})，市场过于激烈")
                return None, 0
        
        # === 核心信号：MACD金叉/死叉 + K线确认 ===
        signal = None
        strength = 0
        
        # 30分钟K线方向判断
        current_open = df['open'].iloc[-1]
        current_close = df['close'].iloc[-1]
        is_bullish_candle = current_close > current_open  # 阳线
        is_bearish_candle = current_close < current_open  # 阴线
        
        # MACD金叉 - 做多信号（仅需K线确认）
        if prev_macd <= prev_signal and current_macd > current_signal:
            # K线确认：必须是阳线
            if is_bullish_candle:
                signal = "做多"
                # 计算信号强度
                macd_diff = current_macd - current_signal
                adx_bonus = 10 if is_trending else 0
                strength = min(100, int(60 + abs(macd_diff) * 1000 + adx_bonus))
                log_message("SIGNAL", f"{symbol} MACD金叉+阳线确认，ADX={adx_value:.2f}，生成做多信号，强度: {strength}")
            else:
                log_message("DEBUG", f"{symbol} MACD金叉但当前K线收阴线，等待阳线确认")
        
        # MACD死叉 - 做空信号（仅需K线确认）
        elif prev_macd >= prev_signal and current_macd < current_signal:
            # K线确认：必须是阴线
            if is_bearish_candle:
                signal = "做空"
                # 计算信号强度
                macd_diff = current_signal - current_macd
                adx_bonus = 10 if is_trending else 0
                strength = min(100, int(60 + abs(macd_diff) * 1000 + adx_bonus))
                log_message("SIGNAL", f"{symbol} MACD死叉+阴线确认，ADX={adx_value:.2f}，生成做空信号，强度: {strength}")
            else:
                log_message("DEBUG", f"{symbol} MACD死叉但当前K线收阳线，等待阴线确认")
        
        return signal, strength
        
    except Exception as e:
        log_message("ERROR", f"{symbol} 生成信号失败: {str(e)}")
        traceback.print_exc()
        return None, 0

# ============================================
# 计算止损止盈
# ============================================
def calculate_stop_loss_take_profit(symbol, price, signal, atr_value):
    """计算止损止盈价格"""
    try:
        if signal == "做多":
            # 多头止损：入场价 - 固定百分比
            stop_loss = price * (1 - FIXED_SL_PERCENTAGE)
            take_profit = price * (1 + FIXED_TP_PERCENTAGE)
        else:  # 做空
            # 空头止损：入场价 + 固定百分比  
            stop_loss = price * (1 + FIXED_SL_PERCENTAGE)
            take_profit = price * (1 - FIXED_TP_PERCENTAGE)
        
        return stop_loss, take_profit
        
    except Exception as e:
        log_message("ERROR", f"计算止损止盈失败: {str(e)}")
        return None, None

# ============================================
# 风险管理检查
# ============================================
def check_and_execute_risk_management(symbol, signal, signal_strength):
    """执行风险管理检查"""
    try:
        # 检查每日交易次数限制
        if trade_stats['daily_trades'] >= MAX_DAILY_TRADES:
            log_message("WARNING", f"已达到每日最大交易次数 ({MAX_DAILY_TRADES})")
            return None
        
        # 检查每日亏损限制
        if trade_stats['daily_pnl'] < -trade_stats['initial_balance'] * MAX_DAILY_LOSS:
            log_message("WARNING", f"已达到每日最大亏损限制")
            return None
        
        # 根据信号强度调整风险
        if signal_strength >= 80:
            adjusted_risk = RISK_PER_TRADE * 1.5  # 高强度信号增加风险
        elif signal_strength >= 60:
            adjusted_risk = RISK_PER_TRADE
        else:
            adjusted_risk = RISK_PER_TRADE * 0.5  # 低强度信号降低风险
        
        return min(adjusted_risk, 0.15)  # 最大风险不超过15%
        
    except Exception as e:
        log_message("ERROR", f"风险管理检查失败: {str(e)}")
        return None

# ============================================
# 执行交易
# ============================================
def execute_trade(symbol, signal, signal_strength):
    """执行交易"""
    try:
        # 检查冷却期
        if symbol in cooldown_symbols and cooldown_symbols[symbol] > time.time():
            remaining_time = int(cooldown_symbols[symbol] - time.time())
            log_message("DEBUG", f"{symbol} 在冷却期内，还剩 {remaining_time} 秒")
            return False
        
        # 检查是否已有相同方向的持仓
        if symbol in position_tracker['positions']:
            existing_position = position_tracker['positions'][symbol]
            if (existing_position['side'] == 'long' and signal == "做多") or \
               (existing_position['side'] == 'short' and signal == "做空"):
                log_message("DEBUG", f"{symbol} 已有{signal}持仓，不重复开仓")
                return False
        
        # 检查持仓数量限制
        open_positions = len([pos for pos in position_tracker['positions'].values() if pos['size'] > 0])
        if open_positions >= MAX_OPEN_POSITIONS:
            log_message("WARNING", f"已达到最大持仓数量 ({MAX_OPEN_POSITIONS})")
            return False
        
        # 执行风险管理检查
        adjusted_risk = check_and_execute_risk_management(symbol, signal, signal_strength)
        if adjusted_risk is None:
            log_message("WARNING", f"{symbol} 风险管理检查未通过")
            return False
        
        # 获取当前价格
        ticker = exchange.fetch_ticker(symbol)
        price = ticker['last']
        
        # 获取K线数据用于计算止损止盈
        ohlcv = get_klines(symbol, timeframe_1h)
        if ohlcv is None:
            log_message("ERROR", f"{symbol} 获取K线数据失败")
            return False
        
        df = process_klines(ohlcv)
        if df is None or df['ATR_14'].isna().all():
            log_message("ERROR", f"{symbol} ATR指标计算失败")
            return False
        
        atr_value = df['ATR_14'].iloc[-1]
        
        # 计算止损止盈
        sl, tp = calculate_stop_loss_take_profit(symbol, price, signal, atr_value)
        
        # 获取账户信息
        account_info = get_account_info()
        if not account_info:
            log_message("ERROR", f"{symbol} 获取账户信息失败")
            return False
        
        # 计算仓位大小
        position_size = calculate_position_size(account_info, symbol, price, sl, adjusted_risk)
        
        if position_size <= 0:
            log_message("ERROR", f"{symbol} 计算仓位大小失败")
            return False
        
        # 执行下单
        side = 'buy' if signal == "做多" else 'sell'
        pos_side = 'long' if signal == "做多" else 'short'
        
        try:
            log_message("TRADE", f"{symbol} 准备下单: {side} {position_size} @ {price}")
            
            # 市价下单
            order = exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=position_size,
                params={'posSide': pos_side}
            )
            
            log_message("SUCCESS", f"{symbol} 下单成功，订单ID: {order['id']}")
            
            # 等待订单执行
            time.sleep(2)
            
            # 验证订单状态
            try:
                order_status = exchange.fetch_order(order['id'], symbol)
                if order_status['status'] != 'closed':
                    log_message("WARNING", f"{symbol} 订单未完全成交: {order_status['status']}")
                
                actual_price = float(order_status.get('average', price))
                actual_size = float(order_status.get('filled', position_size))
                
            except Exception as e:
                log_message("WARNING", f"{symbol} 获取订单状态失败: {str(e)}")
                actual_price = price
                actual_size = position_size
            
            log_message("SUCCESS", f"{symbol} 成交确认: {side} {actual_size} @ {actual_price}")
            
            # 设置止损止盈
            sl_side = 'sell' if signal == "做多" else 'buy'
            tp_side = 'sell' if signal == "做多" else 'buy'
            
            sl_order_id = None
            tp_order_id = None
            
            try:
                # 设置止损
                sl_order = exchange.create_order(
                    symbol=symbol,
                    type='stop',
                    side=sl_side,
                    amount=actual_size,
                    price=sl,
                    params={'stopLossPrice': sl, 'posSide': pos_side}
                )
                sl_order_id = sl_order['id']
                log_message("SUCCESS", f"{symbol} 设置止损成功: {sl}")
                
                # 设置止盈
                tp_order = exchange.create_order(
                    symbol=symbol,
                    type='take_profit',
                    side=tp_side,
                    amount=actual_size,
                    price=tp,
                    params={'takeProfitPrice': tp, 'posSide': pos_side}
                )
                tp_order_id = tp_order['id']
                log_message("SUCCESS", f"{symbol} 设置止盈成功: {tp}")
                
            except Exception as e:
                log_message("ERROR", f"{symbol} 设置止损止盈失败: {str(e)}")
            
            # 更新持仓跟踪器
            position_tracker['positions'][symbol] = {
                'entry_price': actual_price,
                'size': actual_size,
                'side': 'long' if signal == "做多" else 'short',
                'pnl': 0.0,
                'sl': sl,
                'tp': tp,
                'entry_time': datetime.now(),
                'leverage': DEFAULT_LEVERAGE,
                'order_id': order['id'],
                'sl_order_id': sl_order_id,
                'tp_order_id': tp_order_id
            }
            
            # 更新交易统计
            trade_stats['total_trades'] += 1
            trade_stats['daily_trades'] += 1
            
            # 设置冷却期
            cooldown_symbols[symbol] = time.time() + COOLDOWN_PERIOD
            
            return True
            
        except Exception as e:
            log_message("ERROR", f"{symbol} 下单失败: {str(e)}")
            return False
            
    except Exception as e:
        log_message("ERROR", f"{symbol} 执行交易失败: {str(e)}")
        return False

# ============================================
# 更新持仓状态
# ============================================
def update_positions():
    """更新所有持仓的状态"""
    try:
        from datetime import datetime
        now = datetime.now()
        positions = position_tracker['positions'].copy()
        
        for symbol, position in positions.items():
            try:
                # 获取当前价格
                ticker = exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                
                # 计算未实现盈亏
                if position['side'] == 'long':
                    pnl_percentage = (current_price - position['entry_price']) / position['entry_price']
                else:
                    pnl_percentage = (position['entry_price'] - current_price) / position['entry_price']
                
                pnl = position['size'] * position['entry_price'] * pnl_percentage * position['leverage']
                position_tracker['positions'][symbol]['pnl'] = pnl
                
                # 检查MACD金叉/死叉平仓条件
                ohlcv = get_klines(symbol, '30m', limit=100)
                if ohlcv:
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    # 计算MACD指标
                    macd_line, signal_line, histogram = calculate_macd(df['close'], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
                    df['MACD'] = macd_line
                    df['MACD_SIGNAL'] = signal_line
                    
                    # 获取当前和前一个MACD值
                    current_macd = df['MACD'].iloc[-1]
                    current_signal = df['MACD_SIGNAL'].iloc[-1]
                    prev_macd = df['MACD'].iloc[-2]
                    prev_signal = df['MACD_SIGNAL'].iloc[-2]
                    
                    # 做多持仓，检查死叉平仓条件
                    if position['side'] == 'long' and prev_macd >= prev_signal and current_macd < current_signal:
                        log_message("SIGNAL", f"{symbol} MACD死叉，平仓做多持仓")
                        close_position(symbol, reason="MACD死叉平仓")
                    
                    # 做空持仓，检查金叉平仓条件
                    elif position['side'] == 'short' and prev_macd <= prev_signal and current_macd > current_signal:
                        log_message("SIGNAL", f"{symbol} MACD金叉，平仓做空持仓")
                        close_position(symbol, reason="MACD金叉平仓")
                
            except Exception as e:
                log_message("ERROR", f"{symbol} 更新持仓状态失败: {str(e)}")
        
        # 检查已平仓的持仓
        check_closed_positions()
                
    except Exception as e:
        log_message("ERROR", f"更新持仓状态时出错: {str(e)}")

# ============================================
# 平仓函数
# ============================================
def close_position(symbol, reason="手动平仓"):
    """平仓指定持仓"""
    try:
        if symbol not in position_tracker['positions']:
            log_message("WARNING", f"{symbol} 没有持仓")
            return False
        
        position = position_tracker['positions'][symbol]
        
        # 准备平仓订单
        side = 'sell' if position['side'] == 'long' else 'buy'
        pos_side = position['side']
        
        log_message("TRADE", f"{symbol} 准备平仓: {reason}")
        
        # 市价平仓
        order = exchange.create_order(
            symbol=symbol,
            type='market',
            side=side,
            amount=position['size'],
            params={'posSide': pos_side, 'reduceOnly': True}
        )
        
        log_message("SUCCESS", f"{symbol} 平仓订单提交成功，订单ID: {order['id']}")
        
        # 取消止损止盈订单
        try:
            if position.get('sl_order_id'):
                exchange.cancel_order(position['sl_order_id'], symbol)
                log_message("INFO", f"{symbol} 止损订单已取消")
        except:
            pass
        
        try:
            if position.get('tp_order_id'):
                exchange.cancel_order(position['tp_order_id'], symbol)
                log_message("INFO", f"{symbol} 止盈订单已取消")
        except:
            pass
        
        # 记录盈亏
        final_pnl = position.get('pnl', 0)
        trade_stats['total_profit_loss'] += final_pnl
        trade_stats['daily_pnl'] += final_pnl
        
        if final_pnl > 0:
            trade_stats['winning_trades'] += 1
            log_message("SUCCESS", f"{symbol} 盈利平仓: +{final_pnl:.2f} USDT ({reason})")
        else:
            trade_stats['losing_trades'] += 1
            log_message("WARNING", f"{symbol} 亏损平仓: {final_pnl:.2f} USDT ({reason})")
        
        # 从跟踪器移除
        del position_tracker['positions'][symbol]
        
        return True
        
    except Exception as e:
        log_message("ERROR", f"{symbol} 平仓失败: {str(e)}")
        return False

# ============================================
# 检查已平仓的持仓
# ============================================
def check_closed_positions():
    """检查是否有已经被交易所平仓的持仓"""
    try:
        # 获取当前持仓
        exchange_positions = {}
        try:
            positions = exchange.fetch_positions()
            for position in positions:
                if float(position['contracts']) > 0:
                    symbol = position['symbol']
                    exchange_positions[symbol] = position
        except Exception as e:
            log_message("ERROR", f"获取交易所持仓失败: {str(e)}")
            return
        
        # 检查本地跟踪的持仓是否在交易所中已经平仓
        for symbol in list(position_tracker['positions'].keys()):
            if symbol not in exchange_positions:
                log_message("INFO", f"{symbol} 在交易所已平仓，同步本地状态")
                position = position_tracker['positions'][symbol]
                
                # 记录盈亏
                final_pnl = position.get('pnl', 0)
                trade_stats['total_profit_loss'] += final_pnl
                trade_stats['daily_pnl'] += final_pnl
                
                if final_pnl > 0:
                    trade_stats['winning_trades'] += 1
                    log_message("SUCCESS", f"{symbol} 盈利平仓: +{final_pnl:.2f} USDT (交易所平仓)")
                else:
                    trade_stats['losing_trades'] += 1
                    log_message("WARNING", f"{symbol} 亏损平仓: {final_pnl:.2f} USDT (交易所平仓)")
                
                # 从跟踪器移除
                del position_tracker['positions'][symbol]
    
    except Exception as e:
        log_message("ERROR", f"检查已平仓持仓失败: {str(e)}")

# ============================================
# 显示交易统计
# ============================================
def display_trading_stats():
    """显示交易统计信息"""
    try:
        account_info = get_account_info()
        if account_info:
            trade_stats['current_balance'] = account_info['total_balance']
        
        print("\n" + "="*60)
        print("📊 交易统计")
        print("="*60)
        print(f"初始余额: {trade_stats['initial_balance']:.2f} USDT")
        print(f"当前余额: {trade_stats['current_balance']:.2f} USDT")
        print(f"总盈亏: {trade_stats['total_profit_loss']:.2f} USDT")
        print(f"总交易次数: {trade_stats['total_trades']}")
        print(f"盈利交易: {trade_stats['winning_trades']}")
        print(f"亏损交易: {trade_stats['losing_trades']}")
        if trade_stats['total_trades'] > 0:
            win_rate = trade_stats['winning_trades'] / trade_stats['total_trades'] * 100
            print(f"胜率: {win_rate:.1f}%")
        print(f"今日交易: {trade_stats['daily_trades']}")
        print(f"今日盈亏: {trade_stats['daily_pnl']:.2f} USDT")
        print(f"当前持仓: {len(position_tracker['positions'])}")
        print("="*60)
        
        # 显示持仓详情
        if position_tracker['positions']:
            print("📈 当前持仓:")
            for symbol, pos in position_tracker['positions'].items():
                print(f"  {symbol}: {pos['side']} {pos['size']:.4f} @ {pos['entry_price']:.4f} | 盈亏: {pos['pnl']:.2f} USDT")
        
    except Exception as e:
        log_message("ERROR", f"显示统计信息失败: {str(e)}")

# ============================================
# 交易循环函数
# ============================================
def trading_loop():
    """主交易循环"""
    try:
        log_message("INFO", "启动交易循环...")
        
        # 设置交易对列表 - 热度前10 + 指定4个合约
        # 热度排名前10的合约
        top_10_pairs = [
            'BTC-USDT-SWAP', 'ETH-USDT-SWAP', 'SOL-USDT-SWAP', 'BNB-USDT-SWAP',
            'XRP-USDT-SWAP', 'DOGE-USDT-SWAP', 'ADA-USDT-SWAP', 'AVAX-USDT-SWAP',
            'SHIB-USDT-SWAP', 'DOT-USDT-SWAP'
        ]
        # 指定的4个合约
        specified_pairs = ['FIL-USDT-SWAP', 'ZRO-USDT-SWAP', 'WIF-USDT-SWAP', 'WLD-USDT-SWAP']
        trading_pairs = top_10_pairs + specified_pairs
        
        # 初始化交易统计
        account_info = get_account_info()
        if account_info:
            trade_stats['initial_balance'] = account_info['total_balance']
            trade_stats['current_balance'] = account_info['total_balance']
            log_message("SUCCESS", f"初始余额: {trade_stats['initial_balance']:.2f} USDT")
        
        loop_count = 0
        
        # 主循环
        while True:
            try:
                loop_count += 1
                log_message("INFO", f"\n{'='*60}")
                log_message("INFO", f"循环 #{loop_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                log_message("INFO", f"{'='*60}")
                
                # 检查每日重置
                check_daily_reset()
                
                # 更新持仓状态
                update_positions()
                
                # 检查每个交易对的信号
                for symbol in trading_pairs:
                    try:
                        # 生成信号
                        signal, strength = generate_signal(symbol)
                        
                        # 更新最新信号
                        latest_signals[symbol] = (signal, strength, datetime.now())
                        
                        # 如果有信号，执行交易
                        if signal and strength > 40:  # 只交易强度大于40的信号
                            log_message("SIGNAL", f"{symbol} 生成{signal}信号，强度: {strength}")
                            execute_trade(symbol, signal, strength)
                        
                        # 避免请求过快
                        time.sleep(1)
                        
                    except Exception as e:
                        log_message("ERROR", f"{symbol} 处理信号时出错: {str(e)}")
                
                # 显示交易统计（每10个循环显示一次）
                if loop_count % 10 == 0:
                    display_trading_stats()
                
                # 等待下一个循环
                log_message("INFO", "等待60秒后继续下一个循环...")
                time.sleep(60)
                
            except Exception as e:
                log_message("ERROR", f"交易循环中出错: {str(e)}")
                traceback.print_exc()
                time.sleep(60)
                
    except KeyboardInterrupt:
        log_message("INFO", "交易循环被手动中断")
        display_trading_stats()
    except Exception as e:
        log_message("ERROR", f"交易循环启动失败: {str(e)}")
        traceback.print_exc()

# ============================================
# 启动交易系统
# ============================================
def start_trading_system():
    """启动交易系统函数"""
    global exchange
    try:
        # 初始化交易所连接
        exchange = initialize_exchange()
        
        # 测试API连接
        if not test_api_connection():
            log_message("ERROR", "API连接测试失败，请检查配置")
            return
        
        # 显示启动信息
        log_message("SUCCESS", "=" * 60)
        log_message("SUCCESS", "MACD(6,32,9)策略实盘交易系统 - OKX版")
        log_message("SUCCESS", "=" * 60)
        log_message("INFO", f"交易所: OKX")
        log_message("INFO", f"杠杆: {DEFAULT_LEVERAGE}x")
        log_message("INFO", f"单次风险: {RISK_PER_TRADE*100}%")
        log_message("INFO", f"最大持仓: {MAX_OPEN_POSITIONS}")
        log_message("INFO", f"冷却期: {COOLDOWN_PERIOD//60}分钟")
        log_message("INFO", f"每日最大交易: {MAX_DAILY_TRADES}")
        log_message("INFO", f"每日最大亏损: {MAX_DAILY_LOSS*100}%")
        log_message("INFO", "使用30分钟K线图")
        log_message("INFO", "入场信号: MACD快线上穿/下穿慢线(金叉/死叉)")
        log_message("INFO", "震荡过滤: ADX < 20")
        log_message("INFO", "趋势确认: ADX > 25")
        log_message("INFO", "平仓条件: MACD反向交叉")
        log_message("INFO", "MACD平仓规则: 做多时MACD死叉平仓，做空时MACD金叉平仓")
        log_message("INFO", f"交易对: 热度前10 + FIL, ZRO, WIF, WLD (共14个)")
        log_message("SUCCESS", "=" * 60)
        
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