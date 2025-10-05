import ccxt
import pandas as pd
# import pandas_ta as ta  # 暂时注释掉，使用自定义指标计算

# === 环境变量诊断（开头添加） ===
print("=== 启动时环境变量诊断 ===")
import os
okx_vars = ['OKX_API_KEY', 'OKX_SECRET_KEY', 'OKX_PASSPHRASE']
for var in okx_vars:
    value = os.getenv(var)
    if value:
        print(f"✅ {var}: 已设置 (长度: {len(value)})")
    else:
        print(f"❌ {var}: 未设置或为空")

# 检查所有环境变量中是否有 OKX 相关的
all_vars = list(os.environ.keys())
okx_related = [var for var in all_vars if 'OKX' in var.upper()]
if okx_related:
    print(f"找到 OKX 相关变量: {okx_related}")
else:
    print("未找到任何 OKX 相关环境变量")

print(f"总环境变量数量: {len(all_vars)}")
print("=== 诊断完成，继续启动程序 ===")

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
            print("在 Railway 中设置环境变量的步骤:")
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
MAX_LEVERAGE_BTC = 100        # BTC最大杠杆100倍
MAX_LEVERAGE_ETH = 50         # ETH最大杠杆50倍
MAX_LEVERAGE_OTHERS = 30      # 其他币种最大杠杆30倍
DEFAULT_LEVERAGE_BTC = 50     # BTC默认杠杆50倍
DEFAULT_LEVERAGE_ETH = 30     # ETH默认杠杆30倍
DEFAULT_LEVERAGE_OTHERS = 20  # 其他币种默认杠杆20倍
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

# 移动止盈止损配置 - 基于ATR
TRAILING_STOP_ACTIVATION_ATR_MULTIPLIER = 1.5   # ATR的1.5倍后激活移动止损
TRAILING_STOP_CALLBACK_ATR_MULTIPLIER = 0.8     # ATR的0.8倍作为回调触发
TRAILING_TP_ACTIVATION_ATR_MULTIPLIER = 1.0     # ATR的1倍后激活移动止盈
TRAILING_TP_STEP_ATR_MULTIPLIER = 0.5           # ATR的0.5倍作为止盈步长
TRAILING_CHECK_INTERVAL = 30                    # 每30秒检查一次移动止盈止损条件

# ATR动态止盈止损配置
ATR_STOP_LOSS_MULTIPLIER = 2.0                  # 止损距离 = ATR * 2.0
ATR_TAKE_PROFIT_MULTIPLIER = 3.0                # 初始止盈距离 = ATR * 3.0
ATR_MIN_MULTIPLIER = 1.0                        # ATR最小倍数限制
ATR_MAX_MULTIPLIER = 5.0                        # ATR最大倍数限制

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
    formatted_message = f"[{timestamp}] [{level.upper()}] {message}"
    print(formatted_message)
    # 强制刷新输出缓冲区，确保在Railway等云平台上能看到实时日志
    import sys
    sys.stdout.flush()

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
# 动态杠杆选择
# ============================================
def get_leverage_for_symbol(symbol):
    """根据交易对选择合适的杠杆倍数"""
    try:
        # 提取基础货币名称
        base_currency = symbol.split('-')[0].upper()
        
        if base_currency == 'BTC':
            return DEFAULT_LEVERAGE_BTC
        elif base_currency == 'ETH':
            return DEFAULT_LEVERAGE_ETH
        else:
            return DEFAULT_LEVERAGE_OTHERS
            
    except Exception as e:
        log_message("WARNING", f"获取{symbol}杠杆失败，使用默认值: {str(e)}")
        return DEFAULT_LEVERAGE_OTHERS

def get_max_leverage_for_symbol(symbol):
    """根据交易对获取最大杠杆倍数"""
    try:
        # 提取基础货币名称
        base_currency = symbol.split('-')[0].upper()
        
        if base_currency == 'BTC':
            return MAX_LEVERAGE_BTC
        elif base_currency == 'ETH':
            return MAX_LEVERAGE_ETH
        else:
            return MAX_LEVERAGE_OTHERS
            
    except Exception as e:
        log_message("WARNING", f"获取{symbol}最大杠杆失败，使用默认值: {str(e)}")
        return MAX_LEVERAGE_OTHERS

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
                # 使用该币种的动态杠杆
                pos_leverage = get_leverage_for_symbol(pos_symbol)
                pos_value = pos_data['entry_price'] * pos_data['size'] / pos_leverage
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
        
        # 获取该币种的动态杠杆
        leverage = get_leverage_for_symbol(symbol)
        max_leverage = get_max_leverage_for_symbol(symbol)
        
        # 计算仓位大小（考虑动态杠杆）
        position_value_with_leverage = position_fund * leverage
        position_size = position_value_with_leverage / price
        
        log_message("INFO", f"使用杠杆: {leverage}x (最大{max_leverage}x)")
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
                if (0.01 * price) / leverage <= total_trading_fund:
                    position_size = 0.01
                    log_message("INFO", f"使用默认最小交易量: {position_size}")
                else:
                    log_message("ERROR", f"资金不足以满足默认最小交易量要求")
                    return 0
        
        # 最终验证
        final_trade_value = (position_size * price) / leverage
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
        
        # 显示MACD数值
        log_message("INFO", f"{symbol} MACD: {current_macd:.6f}, 信号线: {current_signal:.6f}")
        
        # === ADX趋势/震荡判断 ===
        adx_value = df['ADX'].iloc[-1] if 'ADX' in df.columns and not df['ADX'].isna().all() else 0
        is_trending = adx_value > ADX_THRESHOLD_HIGH
        is_ranging = adx_value < ADX_THRESHOLD_LOW
        
        # 显示ADX分析结果（无论是否震荡都显示）
        log_message("INFO", f"{symbol} ADX值: {adx_value:.2f} (震荡<{ADX_THRESHOLD_LOW}, 趋势>{ADX_THRESHOLD_HIGH})")
        
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
        
        # === 核心信号：MACD金叉/死叉 + K线收盘确认 ===
        signal = None
        strength = 0
        
        # 当前K线（金叉/死叉发生的这根K线）的开盘和收盘价
        current_open = df['open'].iloc[-1]
        current_close = df['close'].iloc[-1]
        is_current_bullish = current_close > current_open  # 当前K线是阳线
        is_current_bearish = current_close < current_open  # 当前K线是阴线
        
        log_message("DEBUG", f"{symbol} 当前K线: 开盘{current_open:.6f}, 收盘{current_close:.6f}, {'阳线' if is_current_bullish else '阴线' if is_current_bearish else '十字星'}")
        
        # MACD金叉 - 做多信号（需要金叉这根K线收阳线确认）
        if prev_macd <= prev_signal and current_macd > current_signal:
            # K线确认：金叉这根K线必须收阳线
            if is_current_bullish:
                signal = "做多"
                # 计算信号强度
                macd_diff = current_macd - current_signal
                adx_bonus = 10 if is_trending else 0
                strength = min(100, int(60 + abs(macd_diff) * 1000 + adx_bonus))
                log_message("SIGNAL", f"{symbol} MACD金叉+当前K线收阳确认，ADX={adx_value:.2f}，生成做多信号，强度: {strength}")
            else:
                log_message("DEBUG", f"{symbol} MACD金叉但当前K线收阴线，等待阳线收盘确认")
        
        # MACD死叉 - 做空信号（需要死叉这根K线收阴线确认）
        elif prev_macd >= prev_signal and current_macd < current_signal:
            # K线确认：死叉这根K线必须收阴线
            if is_current_bearish:
                signal = "做空"
                # 计算信号强度
                macd_diff = current_signal - current_macd
                adx_bonus = 10 if is_trending else 0
                strength = min(100, int(60 + abs(macd_diff) * 1000 + adx_bonus))
                log_message("SIGNAL", f"{symbol} MACD死叉+当前K线收阴确认，ADX={adx_value:.2f}，生成做空信号，强度: {strength}")
            else:
                log_message("DEBUG", f"{symbol} MACD死叉但当前K线收阳线，等待阴线收盘确认")
        
        return signal, strength
        
    except Exception as e:
        log_message("ERROR", f"{symbol} 生成信号失败: {str(e)}")
        traceback.print_exc()
        return None, 0

# ============================================
# 计算止损止盈 - 基于ATR动态计算
# ============================================
def calculate_stop_loss_take_profit(symbol, price, signal, atr_value):
    """基于ATR动态计算止损止盈价格"""
    try:
        # 如果ATR无效，使用固定百分比作为备选
        if atr_value is None or atr_value <= 0:
            log_message("WARNING", f"{symbol} ATR值无效，使用固定百分比")
            if signal == "做多":
                stop_loss = price * (1 - FIXED_SL_PERCENTAGE)
                take_profit = price * (1 + FIXED_TP_PERCENTAGE)
            else:  # 做空
                stop_loss = price * (1 + FIXED_SL_PERCENTAGE)
                take_profit = price * (1 - FIXED_TP_PERCENTAGE)
            return stop_loss, take_profit
        
        # 计算ATR距离
        atr_stop_distance = atr_value * ATR_STOP_LOSS_MULTIPLIER
        atr_tp_distance = atr_value * ATR_TAKE_PROFIT_MULTIPLIER
        
        # 限制ATR距离在合理范围内
        max_stop_distance = price * MAX_SL_PERCENTAGE
        min_stop_distance = price * 0.005  # 最小0.5%
        
        atr_stop_distance = max(min_stop_distance, min(atr_stop_distance, max_stop_distance))
        
        if signal == "做多":
            # 多头：止损在下方，止盈在上方
            stop_loss = price - atr_stop_distance
            take_profit = price + atr_tp_distance
        else:  # 做空
            # 空头：止损在上方，止盈在下方
            stop_loss = price + atr_stop_distance
            take_profit = price - atr_tp_distance
        
        # 计算实际百分比用于日志
        if signal == "做多":
            sl_percentage = (price - stop_loss) / price * 100
            tp_percentage = (take_profit - price) / price * 100
        else:
            sl_percentage = (stop_loss - price) / price * 100
            tp_percentage = (price - take_profit) / price * 100
        
        log_message("INFO", f"{symbol} ATR动态计算 - ATR:{atr_value:.6f}, 止损距离:{atr_stop_distance:.6f}({sl_percentage:.2f}%), 止盈距离:{atr_tp_distance:.6f}({tp_percentage:.2f}%)")
        
        return stop_loss, take_profit
        
    except Exception as e:
        log_message("ERROR", f"计算ATR止损止盈失败: {str(e)}")
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
                # 设置止损条件单 - 使用正确的OKX格式
                log_message("INFO", f"{symbol} 准备设置止损条件单: {sl:.6f}")
                sl_order = exchange.create_order(
                    symbol=symbol,
                    type='stop',
                    side=sl_side,
                    amount=actual_size,
                    price=sl,
                    params={'stopLossPrice': sl, 'posSide': pos_side}
                )
                sl_order_id = sl_order['id']
                log_message("SUCCESS", f"{symbol} 止损条件单设置成功: {sl:.6f}, ID: {sl_order_id}")
                
                # 设置止盈条件单 - 使用正确的OKX格式
                log_message("INFO", f"{symbol} 准备设置止盈条件单: {tp:.6f}")
                tp_order = exchange.create_order(
                    symbol=symbol,
                    type='take_profit',
                    side=tp_side,
                    amount=actual_size,
                    price=tp,
                    params={'takeProfitPrice': tp, 'posSide': pos_side}
                )
                tp_order_id = tp_order['id']
                log_message("SUCCESS", f"{symbol} 止盈条件单设置成功: {tp:.6f}, ID: {tp_order_id}")
                
            except Exception as e:
                log_message("ERROR", f"{symbol} 设置条件单失败: {str(e)}")
                sl_order_id = None
                tp_order_id = None
                log_message("WARNING", f"{symbol} 条件单设置失败，将完全依赖程序监控")
            
            # 更新持仓跟踪器
            position_tracker['positions'][symbol] = {
                'entry_price': actual_price,
                'size': actual_size,
                'side': 'long' if signal == "做多" else 'short',
                'pnl': 0.0,
                'sl': sl,
                'tp': tp,
                'entry_time': datetime.now(),
                'leverage': leverage,
                'order_id': order['id'],
                'sl_order_id': sl_order_id,
                'tp_order_id': tp_order_id
            }
            
            # 设置移动止盈止损跟踪 - 基于ATR
            trailing_stops[symbol] = {
                'active': False,  # 保持兼容性
                'trailing_stop_active': False,  # 移动止损激活状态
                'trailing_tp_active': False,    # 移动止盈激活状态
                'trailing_stop_trigger': None,  # 移动止损触发价
                'trailing_tp_trigger': None,    # 移动止盈触发价
                'side': 'long' if signal == "做多" else 'short',
                'size': actual_size,
                'entry_price': actual_price,
                'entry_atr': atr_value,  # 记录开仓时的ATR值
                'last_check': time.time()
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
# 移动止盈止损功能 - 基于ATR动态调整
# ============================================
def check_trailing_stop(symbol, current_price):
    """基于ATR检查并更新移动止盈止损"""
    if symbol not in trailing_stops or symbol not in position_tracker['positions']:
        return
    
    ts = trailing_stops[symbol]
    position = position_tracker['positions'][symbol]
    
    # 检查是否需要更新
    if time.time() - ts.get('last_check', 0) < TRAILING_CHECK_INTERVAL:
        return
    
    ts['last_check'] = time.time()
    
    try:
        # 获取当前ATR值
        ohlcv = get_klines(symbol, '1h', limit=50)
        if ohlcv:
            df = process_klines(ohlcv)
            if df is not None and not df['ATR_14'].isna().all():
                current_atr = df['ATR_14'].iloc[-1]
            else:
                current_atr = None
        else:
            current_atr = None
        
        # 如果无法获取ATR，使用固定百分比
        if current_atr is None or current_atr <= 0:
            current_atr = current_price * 0.02  # 使用2%作为默认ATR
            log_message("WARNING", f"{symbol} 无法获取ATR，使用默认值: {current_atr:.6f}")
        
        # 确保键名一致性
        if 'activated' in ts and 'active' not in ts:
            ts['active'] = ts['activated']
        elif 'active' not in ts:
            ts['active'] = False
        
        # 初始化移动止损和移动止盈状态
        if 'trailing_stop_active' not in ts:
            ts['trailing_stop_active'] = False
        if 'trailing_tp_active' not in ts:
            ts['trailing_tp_active'] = False
            
        entry_price = position['entry_price']
        
        # 多头持仓
        if ts['side'] == 'long':
            # === 移动止损逻辑 ===
            # 检查是否达到移动止损激活条件
            activation_distance = current_atr * TRAILING_STOP_ACTIVATION_ATR_MULTIPLIER
            if not ts['trailing_stop_active'] and current_price >= entry_price + activation_distance:
                ts['trailing_stop_active'] = True
                callback_distance = current_atr * TRAILING_STOP_CALLBACK_ATR_MULTIPLIER
                ts['trailing_stop_trigger'] = current_price - callback_distance
                log_message("INFO", f"{symbol} 多头移动止损已激活，触发价: {ts['trailing_stop_trigger']:.6f} (ATR:{current_atr:.6f})")
            
            # 如果移动止损已激活，检查是否需要更新
            elif ts['trailing_stop_active']:
                callback_distance = current_atr * TRAILING_STOP_CALLBACK_ATR_MULTIPLIER
                new_stop_trigger = current_price - callback_distance
                
                # 只有当新的止损价格更高时才更新（保护更多利润）
                if new_stop_trigger > ts.get('trailing_stop_trigger', 0):
                    old_trigger = ts.get('trailing_stop_trigger', 0)
                    ts['trailing_stop_trigger'] = new_stop_trigger
                    log_message("INFO", f"{symbol} 多头移动止损更新: {old_trigger:.6f} -> {new_stop_trigger:.6f}")
                    
                    # 更新交易所止损订单
                    update_stop_loss_order(symbol, position, new_stop_trigger)
                
                # 检查是否触发移动止损
                if current_price <= ts['trailing_stop_trigger']:
                    log_message("TRADE", f"{symbol} 触发多头移动止损，当前价: {current_price:.6f}, 触发价: {ts['trailing_stop_trigger']:.6f}")
                    close_position(symbol, "移动止损触发")
                    return
            
            # === 移动止盈逻辑 ===
            # 检查是否达到移动止盈激活条件
            tp_activation_distance = current_atr * TRAILING_TP_ACTIVATION_ATR_MULTIPLIER
            if not ts['trailing_tp_active'] and current_price >= entry_price + tp_activation_distance:
                ts['trailing_tp_active'] = True
                tp_step = current_atr * TRAILING_TP_STEP_ATR_MULTIPLIER
                ts['trailing_tp_trigger'] = current_price + tp_step
                log_message("INFO", f"{symbol} 多头移动止盈已激活，目标价: {ts['trailing_tp_trigger']:.6f}")
                
                # 更新交易所止盈订单
                update_take_profit_order(symbol, position, ts['trailing_tp_trigger'])
            
            # 如果移动止盈已激活，检查是否需要更新
            elif ts['trailing_tp_active']:
                tp_step = current_atr * TRAILING_TP_STEP_ATR_MULTIPLIER
                new_tp_trigger = current_price + tp_step
                
                # 只有当新的止盈价格更高时才更新（追求更大利润）
                if new_tp_trigger > ts.get('trailing_tp_trigger', 0):
                    old_tp = ts.get('trailing_tp_trigger', 0)
                    ts['trailing_tp_trigger'] = new_tp_trigger
                    log_message("INFO", f"{symbol} 多头移动止盈更新: {old_tp:.6f} -> {new_tp_trigger:.6f}")
                    
                    # 更新交易所止盈订单
                    update_take_profit_order(symbol, position, new_tp_trigger)
        
        # 空头持仓
        else:
            # === 移动止损逻辑 ===
            # 检查是否达到移动止损激活条件
            activation_distance = current_atr * TRAILING_STOP_ACTIVATION_ATR_MULTIPLIER
            if not ts['trailing_stop_active'] and current_price <= entry_price - activation_distance:
                ts['trailing_stop_active'] = True
                callback_distance = current_atr * TRAILING_STOP_CALLBACK_ATR_MULTIPLIER
                ts['trailing_stop_trigger'] = current_price + callback_distance
                log_message("INFO", f"{symbol} 空头移动止损已激活，触发价: {ts['trailing_stop_trigger']:.6f} (ATR:{current_atr:.6f})")
            
            # 如果移动止损已激活，检查是否需要更新
            elif ts['trailing_stop_active']:
                callback_distance = current_atr * TRAILING_STOP_CALLBACK_ATR_MULTIPLIER
                new_stop_trigger = current_price + callback_distance
                
                # 只有当新的止损价格更低时才更新（保护更多利润）
                if new_stop_trigger < ts.get('trailing_stop_trigger', float('inf')):
                    old_trigger = ts.get('trailing_stop_trigger', 0)
                    ts['trailing_stop_trigger'] = new_stop_trigger
                    log_message("INFO", f"{symbol} 空头移动止损更新: {old_trigger:.6f} -> {new_stop_trigger:.6f}")
                    
                    # 更新交易所止损订单
                    update_stop_loss_order(symbol, position, new_stop_trigger)
                
                # 检查是否触发移动止损
                if current_price >= ts['trailing_stop_trigger']:
                    log_message("TRADE", f"{symbol} 触发空头移动止损，当前价: {current_price:.6f}, 触发价: {ts['trailing_stop_trigger']:.6f}")
                    close_position(symbol, "移动止损触发")
                    return
            
            # === 移动止盈逻辑 ===
            # 检查是否达到移动止盈激活条件
            tp_activation_distance = current_atr * TRAILING_TP_ACTIVATION_ATR_MULTIPLIER
            if not ts['trailing_tp_active'] and current_price <= entry_price - tp_activation_distance:
                ts['trailing_tp_active'] = True
                tp_step = current_atr * TRAILING_TP_STEP_ATR_MULTIPLIER
                ts['trailing_tp_trigger'] = current_price - tp_step
                log_message("INFO", f"{symbol} 空头移动止盈已激活，目标价: {ts['trailing_tp_trigger']:.6f}")
                
                # 更新交易所止盈订单
                update_take_profit_order(symbol, position, ts['trailing_tp_trigger'])
            
            # 如果移动止盈已激活，检查是否需要更新
            elif ts['trailing_tp_active']:
                tp_step = current_atr * TRAILING_TP_STEP_ATR_MULTIPLIER
                new_tp_trigger = current_price - tp_step
                
                # 只有当新的止盈价格更低时才更新（追求更大利润）
                if new_tp_trigger < ts.get('trailing_tp_trigger', float('inf')):
                    old_tp = ts.get('trailing_tp_trigger', 0)
                    ts['trailing_tp_trigger'] = new_tp_trigger
                    log_message("INFO", f"{symbol} 空头移动止盈更新: {old_tp:.6f} -> {new_tp_trigger:.6f}")
                    
                    # 更新交易所止盈订单
                    update_take_profit_order(symbol, position, new_tp_trigger)
    
    except Exception as e:
        log_message("ERROR", f"{symbol} 检查ATR移动止盈止损失败: {str(e)}")

def update_take_profit_order(symbol, position, new_tp_price):
    """更新交易所止盈订单"""
    try:
        # 取消旧的止盈订单
        if position.get('tp_order_id'):
            try:
                exchange.cancel_order(position['tp_order_id'], symbol)
                log_message("INFO", f"{symbol} 旧止盈订单已取消")
            except Exception as e:
                log_message("WARNING", f"{symbol} 取消旧止盈订单失败: {e}")
        
        # 创建新的移动止盈订单
        side = 'sell' if position['side'] == 'long' else 'buy'
        pos_side = position['side']
        
        tp_order = exchange.create_order(
            symbol=symbol,
            type='take_profit',
            side=side,
            amount=position['size'],
            price=new_tp_price,
            params={
                'takeProfitPrice': new_tp_price,
                'posSide': pos_side,
                'reduceOnly': True
            }
        )
        
        # 更新止盈订单ID
        position_tracker['positions'][symbol]['tp_order_id'] = tp_order['id']
        log_message("SUCCESS", f"{symbol} 移动止盈订单已更新，新价格: {new_tp_price:.6f}，订单ID: {tp_order['id']}")
        
    except Exception as e:
        log_message("ERROR", f"{symbol} 更新移动止盈订单失败: {str(e)}")

def update_stop_loss_order(symbol, position, new_sl_price):
    """更新交易所止损订单"""
    try:
        # 取消旧的止损订单
        if position.get('sl_order_id'):
            try:
                exchange.cancel_order(position['sl_order_id'], symbol)
                log_message("INFO", f"{symbol} 旧止损订单已取消")
            except Exception as e:
                log_message("WARNING", f"{symbol} 取消旧止损订单失败: {e}")
        
        # 创建新的移动止损订单
        side = 'sell' if position['side'] == 'long' else 'buy'
        pos_side = position['side']
        
        sl_order = exchange.create_order(
            symbol=symbol,
            type='stop',
            side=side,
            amount=position['size'],
            price=new_sl_price,
            params={
                'stopLossPrice': new_sl_price,
                'posSide': pos_side,
                'reduceOnly': True
            }
        )
        
        # 更新止损订单ID
        position_tracker['positions'][symbol]['sl_order_id'] = sl_order['id']
        log_message("SUCCESS", f"{symbol} 移动止损订单已更新，新价格: {new_sl_price:.6f}，订单ID: {sl_order['id']}")
        
    except Exception as e:
        log_message("ERROR", f"{symbol} 更新移动止损订单失败: {str(e)}")

# ============================================
# 为已存在的持仓设置止损止盈条件单
# ============================================
def setup_missing_stop_orders():
    """为没有止损止盈订单的持仓补充设置条件单"""
    try:
        for symbol, position in position_tracker['positions'].items():
            # 检查是否缺少止损止盈订单
            if not position.get('sl_order_id') or not position.get('tp_order_id'):
                log_message("INFO", f"{symbol} 检测到缺少止损止盈订单，正在补充设置...")
                
                entry_price = position['entry_price']
                side = position['side']
                size = position['size']
                
                # 重新计算止损止盈价格
                sl_price = position.get('sl')
                tp_price = position.get('tp')
                
                if not sl_price or not tp_price:
                    # 如果没有止损止盈价格，重新计算
                    if side == 'long':
                        sl_price = entry_price * (1 - FIXED_SL_PERCENTAGE)
                        tp_price = entry_price * (1 + FIXED_TP_PERCENTAGE)
                    else:
                        sl_price = entry_price * (1 + FIXED_SL_PERCENTAGE)
                        tp_price = entry_price * (1 - FIXED_TP_PERCENTAGE)
                    
                    position_tracker['positions'][symbol]['sl'] = sl_price
                    position_tracker['positions'][symbol]['tp'] = tp_price
                
                # 设置订单方向和仓位方向
                sl_side = 'sell' if side == 'long' else 'buy'
                tp_side = 'sell' if side == 'long' else 'buy'
                pos_side = 'long' if side == 'long' else 'short'
                
                # 设置止损条件单
                if not position.get('sl_order_id'):
                    try:
                        sl_order = exchange.create_order(
                            symbol=symbol,
                            type='market',
                            side=sl_side,
                            amount=size,
                            params={
                                'stopLossPrice': sl_price,
                                'posSide': pos_side,
                                'reduceOnly': True
                            }
                        )
                        position_tracker['positions'][symbol]['sl_order_id'] = sl_order['id']
                        log_message("SUCCESS", f"{symbol} 补充止损条件单成功: {sl_price:.6f}")
                    except Exception as e:
                        log_message("ERROR", f"{symbol} 补充止损条件单失败: {str(e)}")
                
                # 设置止盈条件单
                if not position.get('tp_order_id'):
                    try:
                        tp_order = exchange.create_order(
                            symbol=symbol,
                            type='market',
                            side=tp_side,
                            amount=size,
                            params={
                                'takeProfitPrice': tp_price,
                                'posSide': pos_side,
                                'reduceOnly': True
                            }
                        )
                        position_tracker['positions'][symbol]['tp_order_id'] = tp_order['id']
                        log_message("SUCCESS", f"{symbol} 补充止盈条件单成功: {tp_price:.6f}")
                    except Exception as e:
                        log_message("ERROR", f"{symbol} 补充止盈条件单失败: {str(e)}")
                        
    except Exception as e:
        log_message("ERROR", f"补充设置止损止盈条件单时出错: {str(e)}")

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
                
                # 1. 检查固定止损止盈（程序监控）
                entry_price = position['entry_price']
                sl_price = position['sl']
                tp_price = position['tp']
                
                # 显示当前监控状态（包括移动止盈止损状态）
                ts_info = ""
                if symbol in trailing_stops:
                    ts = trailing_stops[symbol]
                    if ts.get('trailing_stop_active'):
                        ts_info += f", 移动止损:{ts.get('trailing_stop_trigger', 0):.6f}"
                    if ts.get('trailing_tp_active'):
                        ts_info += f", 移动止盈:{ts.get('trailing_tp_trigger', 0):.6f}"
                
                log_message("DEBUG", f"{symbol} 监控中 - 当前价格:{current_price:.6f}, 固定止损:{sl_price:.6f}, 固定止盈:{tp_price:.6f}, 盈亏:{pnl:.2f}{ts_info}")
                
                if position['side'] == 'long':
                    # 做多仓位检查
                    if current_price <= sl_price:
                        log_message("SIGNAL", f"{symbol} 触发固定止损: 当前价格 {current_price:.6f} <= 止损价格 {sl_price:.6f}")
                        close_position(symbol, reason="固定止损")
                        continue
                    elif current_price >= tp_price:
                        log_message("SIGNAL", f"{symbol} 触发固定止盈: 当前价格 {current_price:.6f} >= 止盈价格 {tp_price:.6f}")
                        close_position(symbol, reason="固定止盈")
                        continue
                else:
                    # 做空仓位检查
                    if current_price >= sl_price:
                        log_message("SIGNAL", f"{symbol} 触发固定止损: 当前价格 {current_price:.6f} >= 止损价格 {sl_price:.6f}")
                        close_position(symbol, reason="固定止损")
                        continue
                    elif current_price <= tp_price:
                        log_message("SIGNAL", f"{symbol} 触发固定止盈: 当前价格 {current_price:.6f} <= 止盈价格 {tp_price:.6f}")
                        close_position(symbol, reason="固定止盈")
                        continue
                
                # 2. 检查移动止盈
                check_trailing_stop(symbol, current_price)
                
                # 3. 检查MACD金叉/死叉平仓条件（K线确认）
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
                    
                    # K线方向判断（MACD反转的这根K线收盘确认）
                    current_open = df['open'].iloc[-1]
                    current_close = df['close'].iloc[-1]
                    is_current_bullish = current_close > current_open  # 当前K线收阳线
                    is_current_bearish = current_close < current_open  # 当前K线收阴线
                    
                    log_message("DEBUG", f"{symbol} 平仓检查 - 当前K线: 开盘{current_open:.6f}, 收盘{current_close:.6f}, {'阳线' if is_current_bullish else '阴线' if is_current_bearish else '十字星'}")
                    
                    # 做多持仓，检查死叉+当前K线收阴线平仓条件
                    if (position['side'] == 'long' and 
                        prev_macd >= prev_signal and current_macd < current_signal):
                        
                        if is_current_bearish:
                            log_message("SIGNAL", f"{symbol} MACD死叉+当前K线收阴确认，平仓做多持仓 (盈亏:{pnl:.2f})")
                            close_position(symbol, reason="MACD死叉+阴线平仓")
                        else:
                            log_message("DEBUG", f"{symbol} MACD死叉但当前K线收阳线，等待阴线收盘确认")
                    
                    # 做空持仓，检查金叉+当前K线收阳线平仓条件
                    elif (position['side'] == 'short' and 
                          prev_macd <= prev_signal and current_macd > current_signal):
                        
                        if is_current_bullish:
                            log_message("SIGNAL", f"{symbol} MACD金叉+当前K线收阳确认，平仓做空持仓 (盈亏:{pnl:.2f})")
                            close_position(symbol, reason="MACD金叉+阳线平仓")
                        else:
                            log_message("DEBUG", f"{symbol} MACD金叉但当前K线收阴线，等待阳线收盘确认")
                
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
            if 'total_profit' not in trade_stats:
                trade_stats['total_profit'] = 0
            trade_stats['total_profit'] += final_pnl
            log_message("SUCCESS", f"{symbol} 盈利平仓: +{final_pnl:.2f} USDT ({reason})")
        else:
            trade_stats['losing_trades'] += 1
            if 'total_loss' not in trade_stats:
                trade_stats['total_loss'] = 0
            trade_stats['total_loss'] += abs(final_pnl)
            log_message("WARNING", f"{symbol} 亏损平仓: {final_pnl:.2f} USDT ({reason})")
        
        # 清理移动止损数据
        if symbol in trailing_stops:
            del trailing_stops[symbol]
        
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
# 保存数据到文件供Web界面使用
# ============================================
def save_dashboard_data():
    """保存交易数据到JSON文件供Web界面读取"""
    try:
        # 获取账户信息
        account_info = get_account_info()
        if account_info:
            trade_stats['current_balance'] = account_info['total_balance']
        
        # 计算胜率
        total_trades = trade_stats.get('total_trades', 0)
        win_trades = trade_stats.get('winning_trades', 0)
        lose_trades = trade_stats.get('losing_trades', 0)
        win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
        
        # 准备持仓数据
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
                    'leverage': position.get('leverage', get_leverage_for_symbol(symbol)),
                    'stopLoss': position.get('sl', 0),
                    'takeProfit': position.get('tp', 0),
                    'pnl': position.get('pnl', 0)
                })
            except Exception as e:
                log_message("WARNING", f"获取{symbol}价格失败: {e}")
        
        # 准备信号数据
        trading_pairs = [
            'BTC-USDT-SWAP', 'ETH-USDT-SWAP', 'SOL-USDT-SWAP', 'BNB-USDT-SWAP',
            'XRP-USDT-SWAP', 'DOGE-USDT-SWAP', 'ADA-USDT-SWAP', 'AVAX-USDT-SWAP',
            'SHIB-USDT-SWAP', 'DOT-USDT-SWAP', 'FIL-USDT-SWAP', 'ZRO-USDT-SWAP',
            'WIF-USDT-SWAP', 'WLD-USDT-SWAP'
        ]
        
        signals = []
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
                log_message("WARNING", f"获取{symbol}信号失败: {e}")
        
        # 组装完整数据
        dashboard_data = {
            'account': {
                'totalBalance': trade_stats.get('current_balance', 0),
                'freeBalance': account_info.get('free_balance', 0) if account_info else 0,
                'dailyPnl': trade_stats.get('daily_pnl', 0),
                'totalPnl': trade_stats.get('total_profit_loss', 0)
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
        
        # 保存到文件
        with open('dashboard_data.json', 'w', encoding='utf-8') as f:
            json.dump(dashboard_data, f, ensure_ascii=False, indent=2)
        
        log_message("DEBUG", f"仪表板数据已更新: 余额{dashboard_data['account']['totalBalance']:.2f}, 持仓{len(positions)}, 胜率{win_rate:.1f}%")
        
    except Exception as e:
        log_message("ERROR", f"保存仪表板数据失败: {str(e)}")

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
        
        # 保存数据供Web界面使用
        save_dashboard_data()
        
    except Exception as e:
        log_message("ERROR", f"显示统计信息失败: {str(e)}")

# 在主循环中也定期保存数据
def periodic_save_data():
    """定期保存数据供Web界面使用"""
    try:
        save_dashboard_data()
    except Exception as e:
        log_message("ERROR", f"定期保存数据失败: {str(e)}")

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
                
                # 检查并补充缺失的止损止盈条件单
                setup_missing_stop_orders()
                
                # 检查每个交易对的信号
                for symbol in trading_pairs:
                    try:
                        log_message("INFO", f"正在分析 {symbol}...")
                        
                        # 生成信号
                        signal, strength = generate_signal(symbol)
                        
                        # 更新最新信号
                        latest_signals[symbol] = (signal, strength, datetime.now())
                        
                        # 显示信号结果（包括无信号的情况）
                        if signal and strength > 40:
                            log_message("SIGNAL", f"{symbol} 生成{signal}信号，强度: {strength}")
                            execute_trade(symbol, signal, strength)
                        elif signal:
                            log_message("INFO", f"{symbol} 信号强度不足: {signal} 强度:{strength}")
                        else:
                            log_message("INFO", f"{symbol} 无交易信号")
                        
                        # 避免请求过快
                        time.sleep(1)
                        
                    except Exception as e:
                        log_message("ERROR", f"{symbol} 处理信号时出错: {str(e)}")
                
                # 显示交易统计（每10个循环显示一次）
                if loop_count % 10 == 0:
                    display_trading_stats()
                
                # 等待下一个循环 - 缩短到30秒以便更及时的止损止盈监控
                log_message("INFO", "等待30秒后继续下一个循环...")
                time.sleep(30)
                
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
# 启动Web仪表板
# ============================================
def start_web_dashboard():
    """启动Web仪表板服务器"""
    try:
        from web_server import start_web_server
        import threading
        
        # 在单独线程中启动Web服务器
        web_thread = threading.Thread(target=start_web_server, kwargs={'port': 8080, 'debug': False}, daemon=True)
        web_thread.start()
        
        log_message("SUCCESS", "🌐 Web仪表板已启动")
        log_message("INFO", "📱 访问地址: http://localhost:8080")
        
        return True
    except Exception as e:
        log_message("ERROR", f"启动Web仪表板失败: {str(e)}")
        return False

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
        
        # 启动Web仪表板
        start_web_dashboard()
        
        # 显示启动信息
        log_message("SUCCESS", "=" * 60)
        log_message("SUCCESS", "MACD(6,32,9)策略实盘交易系统 - OKX版")
        log_message("SUCCESS", "=" * 60)
        log_message("INFO", f"交易所: OKX")
        log_message("INFO", f"杠杆: BTC {DEFAULT_LEVERAGE_BTC}x, ETH {DEFAULT_LEVERAGE_ETH}x, 其他 {DEFAULT_LEVERAGE_OTHERS}x")
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
        log_message("INFO", "🌐 Web仪表板: http://localhost:8080")
        log_message("SUCCESS", "=" * 60)
        
        # 启动交易循环
        trading_loop()
        
    except Exception as e:
        log_message("ERROR", f"启动交易系统失败: {str(e)}")
        traceback.print_exc()

# ============================================
# 为已存在持仓补充止盈止损条件单
# ============================================
def setup_missing_stop_orders():
    """为已存在但缺少止损止盈订单的持仓补充设置条件单"""
    try:
        # 获取交易所当前持仓
        positions = exchange.fetch_positions()
        
        for position in positions:
            if float(position['contracts']) > 0:  # 有持仓
                symbol = position['symbol']
                size = float(position['contracts'])
                side = position['side']  # 'long' 或 'short'
                entry_price = float(position['entryPrice'])
                
                log_message("INFO", f"检查 {symbol} 持仓: {side} {size} @ {entry_price}")
                
                # 检查本地跟踪器中是否有这个持仓的条件单记录
                if symbol not in position_tracker['positions']:
                    # 本地没有记录，需要补充
                    log_message("WARNING", f"{symbol} 本地无记录，补充止盈止损条件单")
                    
                    # 计算止损止盈价格
                    if side == 'long':
                        sl = entry_price * (1 - FIXED_SL_PERCENTAGE)
                        tp = entry_price * (1 + FIXED_TP_PERCENTAGE)
                        sl_side = 'sell'
                        tp_side = 'sell'
                    else:  # short
                        sl = entry_price * (1 + FIXED_SL_PERCENTAGE)
                        tp = entry_price * (1 - FIXED_TP_PERCENTAGE)
                        sl_side = 'buy'
                        tp_side = 'buy'
                    
                    sl_order_id = None
                    tp_order_id = None
                    
                    try:
                        # 设置止损条件单
                        log_message("INFO", f"{symbol} 补充设置止损条件单: {sl:.6f}")
                        sl_order = exchange.create_order(
                            symbol=symbol,
                            type='stop',
                            side=sl_side,
                            amount=size,
                            price=sl,
                            params={'stopLossPrice': sl, 'posSide': side}
                        )
                        sl_order_id = sl_order['id']
                        log_message("SUCCESS", f"{symbol} 补充止损条件单成功: {sl:.6f}, ID: {sl_order_id}")
                        
                        # 设置止盈条件单
                        log_message("INFO", f"{symbol} 补充设置止盈条件单: {tp:.6f}")
                        tp_order = exchange.create_order(
                            symbol=symbol,
                            type='take_profit',
                            side=tp_side,
                            amount=size,
                            price=tp,
                            params={'takeProfitPrice': tp, 'posSide': side}
                        )
                        tp_order_id = tp_order['id']
                        log_message("SUCCESS", f"{symbol} 补充止盈条件单成功: {tp:.6f}, ID: {tp_order_id}")
                        
                        # 添加到本地跟踪器
                        position_tracker['positions'][symbol] = {
                            'entry_price': entry_price,
                            'size': size,
                            'side': side,
                            'pnl': 0.0,
                            'sl': sl,
                            'tp': tp,
                            'entry_time': datetime.now(),
                            'leverage': get_leverage_for_symbol(symbol),
                            'order_id': None,  # 原始开仓订单ID未知
                            'sl_order_id': sl_order_id,
                            'tp_order_id': tp_order_id
                        }
                        
                        log_message("SUCCESS", f"{symbol} 持仓补充完成，已添加到本地跟踪器")
                        
                    except Exception as e:
                        log_message("ERROR", f"{symbol} 补充条件单失败: {str(e)}")
                
                elif position_tracker['positions'][symbol].get('sl_order_id') is None or position_tracker['positions'][symbol].get('tp_order_id') is None:
                    # 本地有记录但缺少条件单ID，需要补充
                    log_message("WARNING", f"{symbol} 本地记录缺少条件单ID，尝试补充")
                    
                    pos_data = position_tracker['positions'][symbol]
                    sl = pos_data.get('sl')
                    tp = pos_data.get('tp')
                    
                    if sl and tp:
                        sl_side = 'sell' if side == 'long' else 'buy'
                        tp_side = 'sell' if side == 'long' else 'buy'
                        
                        try:
                            if pos_data.get('sl_order_id') is None:
                                # 补充止损条件单
                                sl_order = exchange.create_order(
                                    symbol=symbol,
                                    type='stop',
                                    side=sl_side,
                                    amount=size,
                                    price=sl,
                                    params={'stopLossPrice': sl, 'posSide': side}
                                )
                                position_tracker['positions'][symbol]['sl_order_id'] = sl_order['id']
                                log_message("SUCCESS", f"{symbol} 补充止损条件单: {sl:.6f}, ID: {sl_order['id']}")
                            
                            if pos_data.get('tp_order_id') is None:
                                # 补充止盈条件单
                                tp_order = exchange.create_order(
                                    symbol=symbol,
                                    type='take_profit',
                                    side=tp_side,
                                    amount=size,
                                    price=tp,
                                    params={'takeProfitPrice': tp, 'posSide': side}
                                )
                                position_tracker['positions'][symbol]['tp_order_id'] = tp_order['id']
                                log_message("SUCCESS", f"{symbol} 补充止盈条件单: {tp:.6f}, ID: {tp_order['id']}")
                                
                        except Exception as e:
                            log_message("ERROR", f"{symbol} 补充缺失条件单失败: {str(e)}")
                else:
                    log_message("DEBUG", f"{symbol} 条件单完整，无需补充")
                    
    except Exception as e:
        log_message("ERROR", f"补充止盈止损条件单失败: {str(e)}")

# ============================================
# 主程序入口
# ============================================
if __name__ == "__main__":
    start_trading_system()