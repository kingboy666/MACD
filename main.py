import ccxt
import pandas as pd
import traceback
import numpy as np
from datetime import datetime, timedelta
import time
import json
from dotenv import load_dotenv
# import pandas_ta as ta  # 暂时注释掉，使用自定义指标计算

# 加载环境变量
load_dotenv()

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

            break
        except Exception as e:
            log_message("ERROR", f"交易循环异常: {str(e)}")
            traceback.print_exc()
            time.sleep(60)  # 异常后等待1分钟再继续
=======
            # 主循环延迟
            log_message("INFO", f"交易循环完成，等待{MAIN_LOOP_DELAY}秒...")
            time.sleep(MAIN_LOOP_DELAY)
            
        except KeyboardInterrupt:
=======
def main_loop():
    """主循环入口（兼容性函数）"""
    trading_loop()

# ============================================
# 主程序入口
# ============================================
if __name__ == "__main__":
    start_trading_system()
=======
def main_loop():
    """主循环入口（兼容性函数）"""
    trading_loop()

# ============================================
# 主程序入口
# ============================================
if __name__ == "__main__":
    start_trading_system()
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
timeframe_30m = '30m'
exchange = None  # 延迟初始化，在启动时设置

# ============================================
# 全局配置常量
# ============================================
# 智能杠杆配置
MAX_LEVERAGE_BTC = 100                       # BTC最大杠杆
MAX_LEVERAGE_ETH = 50                        # ETH最大杠杆
MAX_LEVERAGE_MAJOR = 30                      # 主流币最大杠杆
MAX_LEVERAGE_OTHERS = 20                     # 其他币种最大杠杆
DEFAULT_LEVERAGE = 20                        # 默认杠杆（备用）

# 主流币种列表（享受较高杠杆）
MAJOR_COINS = ['SOL', 'BNB', 'XRP', 'ADA', 'AVAX', 'DOT', 'MATIC', 'LINK', 'UNI', 'LTC']
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

# 移动止盈止损配置 - 基于ATR的动态调整
TRAILING_STOP_ACTIVATION_PERCENTAGE = 0.015  # 价格移动1.5%后激活移动止损
TRAILING_STOP_CALLBACK_PERCENTAGE = 0.008   # 回调0.8%触发止损
TRAILING_TAKE_PROFIT_ACTIVATION_PERCENTAGE = 0.01  # 价格移动1%后激活移动止盈
TRAILING_TAKE_PROFIT_STEP_PERCENTAGE = 0.005  # 每次移动止盈步长0.5%
TRAILING_CHECK_INTERVAL = 30                # 每30秒检查一次移动止盈止损条件

# ATR动态止盈止损配置
ATR_STOP_LOSS_MULTIPLIER = 2.0              # ATR止损倍数
ATR_TAKE_PROFIT_MULTIPLIER = 3.0            # ATR止盈倍数
ATR_TRAILING_ACTIVATION_MULTIPLIER = 1.5    # ATR移动止盈激活倍数
ATR_TRAILING_CALLBACK_MULTIPLIER = 1.0      # ATR移动止盈回调倍数
USE_ATR_DYNAMIC_STOPS = True                 # 启用ATR动态止盈止损
ATR_MIN_MULTIPLIER = 1.0                    # ATR最小倍数
ATR_MAX_MULTIPLIER = 5.0                    # ATR最大倍数

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
            cache_duration = 60 if timeframe == '1m' else 1800 if timeframe == '30m' else 3600 if timeframe == '1h' else 300
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
# 智能杠杆计算
# ============================================
def get_smart_leverage(symbol, account_balance, atr_percentage=None):
    """根据币种和账户大小智能计算杠杆倍数"""
    try:
        # 提取币种名称
        base_symbol = symbol.split('-')[0].upper()
        
        # 根据币种确定基础杠杆
        if base_symbol == 'BTC':
            max_leverage = MAX_LEVERAGE_BTC
            base_leverage = 60  # BTC基础杠杆（确保最低20倍）
        elif base_symbol == 'ETH':
            max_leverage = MAX_LEVERAGE_ETH
            base_leverage = 30  # ETH基础杠杆
        elif base_symbol in MAJOR_COINS:
            max_leverage = MAX_LEVERAGE_MAJOR
            base_leverage = 25  # 主流币基础杠杆（确保最低20倍）
        else:
            max_leverage = MAX_LEVERAGE_OTHERS
            base_leverage = 25  # 其他币种基础杠杆（确保最低20倍）
        
        # 根据账户大小调整杠杆
        if account_balance >= 10000:  # 大账户（1万USDT以上）
            leverage_multiplier = 1.0  # 可以使用较高杠杆
        elif account_balance >= 1000:  # 中等账户（1千-1万USDT）
            leverage_multiplier = 0.8  # 适中杠杆
        elif account_balance >= 100:   # 小账户（100-1000USDT）
            leverage_multiplier = 0.6  # 较低杠杆
        else:  # 微型账户（100USDT以下）
            leverage_multiplier = 0.4  # 最低杠杆
        
        # 根据ATR波动性调整杠杆（如果提供）
        volatility_multiplier = 1.0
        if atr_percentage:
            if atr_percentage > 0.05:      # 高波动性（>5%）
                volatility_multiplier = 0.6
            elif atr_percentage > 0.03:    # 中等波动性（3-5%）
                volatility_multiplier = 0.8
            elif atr_percentage > 0.015:   # 低波动性（1.5-3%）
                volatility_multiplier = 1.0
            else:                          # 极低波动性（<1.5%）
                volatility_multiplier = 1.2
        
        # 计算最终杠杆
        calculated_leverage = int(base_leverage * leverage_multiplier * volatility_multiplier)
        
        # 确保不超过最大杠杆限制
        final_leverage = min(calculated_leverage, max_leverage)
        
        # 确保最小杠杆（所有币种最低20倍）
        final_leverage = max(final_leverage, 20)  # 所有币种最低20倍杠杆
        
        log_message("INFO", f"{symbol} 智能杠杆计算:")
        log_message("INFO", f"  币种: {base_symbol}, 最大杠杆: {max_leverage}x")
        log_message("INFO", f"  账户余额: {account_balance:.2f} USDT")
        log_message("INFO", f"  基础杠杆: {base_leverage}x")
        log_message("INFO", f"  账户调整: {leverage_multiplier:.1f}x")
        if atr_percentage:
            log_message("INFO", f"  波动性调整: {volatility_multiplier:.1f}x (ATR: {atr_percentage:.3f})")
        log_message("INFO", f"  最终杠杆: {final_leverage}x")
        
        return final_leverage
        
    except Exception as e:
        log_message("ERROR", f"智能杠杆计算失败: {str(e)}")
        return DEFAULT_LEVERAGE

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
def calculate_position_size(symbol, price, total_balance):
    """计算仓位大小"""
    try:
        # 使用智能杠杆计算
        smart_leverage = get_smart_leverage(symbol, total_balance)
        
        # 计算本次交易分配的资金
        total_trading_fund = total_balance * 0.8
        
        # 智能分配仓位资金
        open_positions = len(position_tracker['positions'])
        if open_positions == 0:
            position_fund = total_trading_fund * 0.5  # 第一个仓位使用50%
        elif open_positions == 1:
            position_fund = total_trading_fund * 0.3  # 第二个仓位使用30%
        else:
            remaining_fund = total_trading_fund * 0.2  # 剩余20%平分给其他仓位
            max_additional_positions = MAX_OPEN_POSITIONS - 2
            position_fund = remaining_fund / max_additional_positions if max_additional_positions > 0 else remaining_fund
        
        log_message("INFO", f"本次交易分配资金: {position_fund:.2f} USDT")
        
        # 计算仓位大小（考虑智能杠杆）
        position_value_with_leverage = position_fund * smart_leverage
        position_size = position_value_with_leverage / price
        
        log_message("INFO", f"智能杠杆: {smart_leverage}x")
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
                    log_message("INFO", f"计算仓位{position_size:.6f}小于交易所最小量{min_amount}")
                    log_message("INFO", f"系统允许小额交易，继续使用计算仓位")
                    # 移除最小交易量强制要求，允许任何金额下单
        except Exception as e:
            log_message("WARNING", f"获取市场信息失败: {e}")
            log_message("INFO", f"使用计算仓位: {position_size:.6f}")
        
        # 最终验证
        final_trade_value = (position_size * price) / smart_leverage
        log_message("INFO", f"最终交易价值: {final_trade_value:.2f} USDT")
        log_message("INFO", f"最终仓位大小: {position_size:.6f}")
        
        return position_size
        
    except Exception as e:
        log_message("ERROR", f"仓位计算失败: {e}")
        return 0

def get_smart_leverage(symbol, account_balance, atr_percentage=None):
    """根据币种和账户大小智能计算杠杆倍数"""
    try:
        # 提取币种名称
        base_symbol = symbol.split('-')[0].upper()
        
        # 根据币种确定基础杠杆
        if base_symbol == 'BTC':
            max_leverage = MAX_LEVERAGE_BTC
            base_leverage = 60  # BTC基础杠杆
        elif base_symbol == 'ETH':
            max_leverage = MAX_LEVERAGE_ETH
            base_leverage = 30  # ETH基础杠杆
        elif base_symbol in MAJOR_COINS:
            max_leverage = MAX_LEVERAGE_MAJOR
            base_leverage = 25  # 主流币基础杠杆
        else:
            max_leverage = MAX_LEVERAGE_OTHERS
            base_leverage = 25  # 其他币种基础杠杆
        
        # 根据账户大小调整杠杆
        if account_balance >= 10000:  # 大账户（1万USDT以上）
            leverage_multiplier = 1.0  # 可以使用较高杠杆
        elif account_balance >= 1000:  # 中等账户（1千-1万USDT）
            leverage_multiplier = 0.8  # 适中杠杆
        elif account_balance >= 100:   # 小账户（100-1000USDT）
            leverage_multiplier = 0.6  # 较低杠杆
        else:  # 微型账户（100USDT以下）
            leverage_multiplier = 0.4  # 最低杠杆
        
        # 根据ATR波动性调整杠杆（如果提供）
        volatility_multiplier = 1.0
        if atr_percentage:
            if atr_percentage > 0.05:      # 高波动性（>5%）
                volatility_multiplier = 0.6
            elif atr_percentage > 0.03:    # 中等波动性（3-5%）
                volatility_multiplier = 0.8
            elif atr_percentage > 0.015:   # 低波动性（1.5-3%）
                volatility_multiplier = 1.0
            else:                          # 极低波动性（<1.5%）
                volatility_multiplier = 1.2
        
        # 计算最终杠杆
        calculated_leverage = int(base_leverage * leverage_multiplier * volatility_multiplier)
        
        # 确保不超过最大杠杆限制
        final_leverage = min(calculated_leverage, max_leverage)
        
        # 确保最小杠杆为5倍
        final_leverage = max(final_leverage, 5)
        
        log_message("INFO", f"{symbol} 智能杠杆计算:")
        log_message("INFO", f"  币种: {base_symbol}, 最大杠杆: {max_leverage}x")
        log_message("INFO", f"  账户余额: {account_balance:.2f} USDT")
        log_message("INFO", f"  基础杠杆: {base_leverage}x")
        log_message("INFO", f"  账户调整: {leverage_multiplier:.1f}x")
        if atr_percentage:
            log_message("INFO", f"  波动性调整: {volatility_multiplier:.1f}x (ATR: {atr_percentage:.3f})")
        log_message("INFO", f"  最终杠杆: {final_leverage}x")
        
        return final_leverage
        
    except Exception as e:
        log_message("ERROR", f"智能杠杆计算失败: {str(e)}")
        return DEFAULT_LEVERAGE



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
# 计算止损止盈
# ============================================
def calculate_stop_loss_take_profit(symbol, price, signal, atr_value):
    """计算止损止盈价格 - 支持ATR动态调整"""
    try:
        if USE_ATR_DYNAMIC_STOPS and atr_value and atr_value > 0:
            # 使用ATR动态计算止损止盈
            log_message("INFO", f"{symbol} 使用ATR动态止损止盈，ATR值: {atr_value:.6f}")
            
            # 计算ATR相对于价格的百分比
            atr_percentage = atr_value / price
            
            # 限制ATR倍数在合理范围内
            atr_sl_multiplier = max(ATR_MIN_MULTIPLIER, min(ATR_MAX_MULTIPLIER, ATR_STOP_LOSS_MULTIPLIER))
            atr_tp_multiplier = max(ATR_MIN_MULTIPLIER, min(ATR_MAX_MULTIPLIER, ATR_TAKE_PROFIT_MULTIPLIER))
            
            # 计算ATR止损止盈距离
            atr_sl_distance = atr_value * atr_sl_multiplier
            atr_tp_distance = atr_value * atr_tp_multiplier
            
            if signal == "做多":
                # 多头ATR止损止盈
                stop_loss = price - atr_sl_distance
                take_profit = price + atr_tp_distance
                
                # 确保止损不超过最大止损百分比
                max_sl_price = price * (1 - MAX_SL_PERCENTAGE)
                if stop_loss < max_sl_price:
                    stop_loss = max_sl_price
                    log_message("WARNING", f"{symbol} ATR止损过大，调整为最大止损: {stop_loss:.6f}")
                    
            else:  # 做空
                # 空头ATR止损止盈
                stop_loss = price + atr_sl_distance
                take_profit = price - atr_tp_distance
                
                # 确保止损不超过最大止损百分比
                max_sl_price = price * (1 + MAX_SL_PERCENTAGE)
                if stop_loss > max_sl_price:
                    stop_loss = max_sl_price
                    log_message("WARNING", f"{symbol} ATR止损过大，调整为最大止损: {stop_loss:.6f}")
            
            log_message("INFO", f"{symbol} ATR止损止盈 - 止损: {stop_loss:.6f}, 止盈: {take_profit:.6f}")
            log_message("INFO", f"{symbol} ATR倍数 - 止损: {atr_sl_multiplier}x, 止盈: {atr_tp_multiplier}x")
            
        else:
            # 使用固定百分比计算止损止盈
            log_message("INFO", f"{symbol} 使用固定百分比止损止盈")
            
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
        
        # 亏损限制已移除，可无限制下单
        
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
                'leverage': smart_leverage,  # 使用智能杠杆
                'order_id': order['id'],
                'sl_order_id': sl_order_id,
                'tp_order_id': tp_order_id
            }
            
            # 设置移动止盈跟踪
            trailing_stops[symbol] = {
                'active': False,
                'activation_price': tp if signal == "做多" else sl,
                'current_trigger': None,
                'side': 'long' if signal == "做多" else 'short',
                'size': actual_size,
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
# 移动止盈止损功能
# ============================================
def check_trailing_stop(symbol, current_price):
    """检查并更新移动止盈"""
    if symbol not in trailing_stops or symbol not in position_tracker['positions']:
        return
    
    ts = trailing_stops[symbol]
    position = position_tracker['positions'][symbol]
    
    # 检查是否需要更新
    if time.time() - ts.get('last_check', 0) < TRAILING_STOP_CHECK_INTERVAL:
        return
    
    ts['last_check'] = time.time()
    
    try:
        # 确保键名一致性
        if 'activated' in ts and 'active' not in ts:
            ts['active'] = ts['activated']
        elif 'active' not in ts:
            ts['active'] = False
        
        # 获取当前ATR值用于动态调整
        atr_value = None
        if USE_ATR_DYNAMIC_STOPS:
            try:
                ohlcv = get_klines(symbol, '30m', limit=50)
                if ohlcv:
                    df = process_klines(ohlcv)
                    if df is not None and 'ATR_14' in df.columns and not df['ATR_14'].isna().all():
                        atr_value = df['ATR_14'].iloc[-1]
                        log_message("DEBUG", f"{symbol} 移动止盈ATR值: {atr_value:.6f}")
            except Exception as e:
                log_message("WARNING", f"{symbol} 获取移动止盈ATR值失败: {str(e)}")
        
        # 计算激活和回调距离
        if USE_ATR_DYNAMIC_STOPS and atr_value and atr_value > 0:
            # 使用ATR动态计算
            activation_distance = atr_value * ATR_TRAILING_ACTIVATION_MULTIPLIER
            callback_distance = atr_value * ATR_TRAILING_CALLBACK_MULTIPLIER
            
            # 转换为百分比
            activation_percentage = activation_distance / position['entry_price']
            callback_percentage = callback_distance / current_price
            
            log_message("DEBUG", f"{symbol} ATR移动止盈 - 激活: {activation_percentage:.4f}%, 回调: {callback_percentage:.4f}%")
        else:
            # 使用固定百分比
            activation_percentage = TRAILING_STOP_ACTIVATION_PERCENTAGE
            callback_percentage = TRAILING_STOP_CALLBACK_PERCENTAGE
            log_message("DEBUG", f"{symbol} 固定移动止盈 - 激活: {activation_percentage:.4f}%, 回调: {callback_percentage:.4f}%")
            
        # 多头持仓
        if ts['side'] == 'long':
            # 计算当前盈利百分比
            profit_percentage = (current_price - position['entry_price']) / position['entry_price']
            
            # 检查是否达到激活条件（确保至少有足够盈利）
            if not ts['active']:
                # 确保至少有2%盈利或ATR激活距离，取较大者
                min_profit_required = 0.02  # 最少2%盈利
                required_activation = max(activation_percentage, min_profit_required)
                
                if current_price >= position['entry_price'] * (1 + required_activation):
                    ts['active'] = True
                    ts['atr_based'] = USE_ATR_DYNAMIC_STOPS and atr_value and atr_value > 0
                    
                    # 计算初始触发价，确保保护至少50%的当前利润
                    current_profit = current_price - position['entry_price']
                    min_protection_price = position['entry_price'] + current_profit * 0.5
                    
                    # 使用回调百分比计算触发价
                    callback_trigger_price = current_price * (1 - callback_percentage)
                    
                    # 取两者中较高的价格，确保利润保护
                    ts['current_trigger'] = max(callback_trigger_price, min_protection_price)
                    
                    protected_profit = (ts['current_trigger'] - position['entry_price']) / position['entry_price']
                    log_message("SUCCESS", f"{symbol} 多头移动止盈已激活!")
                    log_message("SUCCESS", f"  当前盈利: {profit_percentage:.2%}, 保护利润: {protected_profit:.2%}")
                    log_message("SUCCESS", f"  触发价: {ts['current_trigger']:.6f} ({'ATR' if ts.get('atr_based') else '固定'}模式)")
                    
                    # 更新交易所止盈订单
                    update_take_profit_order(symbol, position, ts['current_trigger'])
            
            # 如果已激活，检查是否需要更新触发价
            elif ts['active']:
                # 重新计算回调百分比（如果是ATR模式）
                if ts.get('atr_based') and atr_value and atr_value > 0:
                    callback_percentage = (atr_value * ATR_TRAILING_CALLBACK_MULTIPLIER) / current_price
                
                # 如果价格创新高，更新触发价
                expected_trigger_base = ts['current_trigger'] / (1 - callback_percentage)
                if current_price > expected_trigger_base:
                    # 计算新的触发价
                    new_trigger_by_callback = current_price * (1 - callback_percentage)
                    
                    # 确保新触发价至少保护30%的总利润
                    total_profit = current_price - position['entry_price']
                    min_protection_price = position['entry_price'] + total_profit * 0.3
                    new_trigger = max(new_trigger_by_callback, min_protection_price)
                    
                    # 只有当新触发价更高时才更新（保护利润原则）
                    if new_trigger > ts['current_trigger']:
                        old_trigger = ts['current_trigger']
                        ts['current_trigger'] = new_trigger
                        
                        old_protected_profit = (old_trigger - position['entry_price']) / position['entry_price']
                        new_protected_profit = (new_trigger - position['entry_price']) / position['entry_price']
                        
                        log_message("INFO", f"{symbol} 多头移动止盈更新: {old_trigger:.6f} -> {new_trigger:.6f}")
                        log_message("INFO", f"  保护利润提升: {old_protected_profit:.2%} -> {new_protected_profit:.2%}")
                        
                        # 更新交易所止盈订单
                        update_take_profit_order(symbol, position, new_trigger)
                
                # 检查是否触发平仓
                if ts['active'] and current_price <= ts['current_trigger']:
                    final_profit_pct = (ts['current_trigger'] - position['entry_price']) / position['entry_price'] * 100
                    log_message("TRADE", f"{symbol} 触发多头移动止盈，保护利润: {final_profit_pct:.2f}%")
                    log_message("TRADE", f"  当前价: {current_price:.6f}, 触发价: {ts['current_trigger']:.6f}")
                    close_position(symbol, "移动止盈触发")
        
        # 空头持仓
        else:
            # 计算当前盈利百分比
            profit_percentage = (position['entry_price'] - current_price) / position['entry_price']
            
            # 检查是否达到激活条件（确保至少有足够盈利）
            if not ts['active']:
                # 确保至少有2%盈利或ATR激活距离，取较大者
                min_profit_required = 0.02  # 最少2%盈利
                required_activation = max(activation_percentage, min_profit_required)
                
                if current_price <= position['entry_price'] * (1 - required_activation):
                    ts['active'] = True
                    ts['atr_based'] = USE_ATR_DYNAMIC_STOPS and atr_value and atr_value > 0
                    
                    # 计算初始触发价，确保保护至少50%的当前利润
                    current_profit = position['entry_price'] - current_price
                    max_protection_price = position['entry_price'] - current_profit * 0.5
                    
                    # 使用回调百分比计算触发价
                    callback_trigger_price = current_price * (1 + callback_percentage)
                    
                    # 取两者中较低的价格，确保利润保护
                    ts['current_trigger'] = min(callback_trigger_price, max_protection_price)
                    
                    protected_profit = (position['entry_price'] - ts['current_trigger']) / position['entry_price']
                    log_message("SUCCESS", f"{symbol} 空头移动止盈已激活!")
                    log_message("SUCCESS", f"  当前盈利: {profit_percentage:.2%}, 保护利润: {protected_profit:.2%}")
                    log_message("SUCCESS", f"  触发价: {ts['current_trigger']:.6f} ({'ATR' if ts.get('atr_based') else '固定'}模式)")
                    
                    # 更新交易所止盈订单
                    update_take_profit_order(symbol, position, ts['current_trigger'])
            
            # 如果已激活，检查是否需要更新触发价
            elif ts['active']:
                # 重新计算回调百分比（如果是ATR模式）
                if ts.get('atr_based') and atr_value and atr_value > 0:
                    callback_percentage = (atr_value * ATR_TRAILING_CALLBACK_MULTIPLIER) / current_price
                
                # 如果价格创新低，更新触发价
                expected_trigger_base = ts['current_trigger'] / (1 + callback_percentage)
                if current_price < expected_trigger_base:
                    # 计算新的触发价
                    new_trigger_by_callback = current_price * (1 + callback_percentage)
                    
                    # 确保新触发价至少保护30%的总利润
                    total_profit = position['entry_price'] - current_price
                    max_protection_price = position['entry_price'] - total_profit * 0.3
                    new_trigger = min(new_trigger_by_callback, max_protection_price)
                    
                    # 只有当新触发价更低时才更新（保护利润原则）
                    if new_trigger < ts['current_trigger']:
                        old_trigger = ts['current_trigger']
                        ts['current_trigger'] = new_trigger
                        
                        old_protected_profit = (position['entry_price'] - old_trigger) / position['entry_price']
                        new_protected_profit = (position['entry_price'] - new_trigger) / position['entry_price']
                        
                        log_message("INFO", f"{symbol} 空头移动止盈更新: {old_trigger:.6f} -> {new_trigger:.6f}")
                        log_message("INFO", f"  保护利润提升: {old_protected_profit:.2%} -> {new_protected_profit:.2%}")
                        
                        # 更新交易所止盈订单
                        update_take_profit_order(symbol, position, new_trigger)
                
                # 检查是否触发平仓
                if ts['active'] and current_price >= ts['current_trigger']:
                    final_profit_pct = (position['entry_price'] - ts['current_trigger']) / position['entry_price'] * 100
                    log_message("TRADE", f"{symbol} 触发空头移动止盈，保护利润: {final_profit_pct:.2f}%")
                    log_message("TRADE", f"  当前价: {current_price:.6f}, 触发价: {ts['current_trigger']:.6f}")
                    close_position(symbol, "移动止盈触发")
    
    except Exception as e:
        log_message("ERROR", f"{symbol} 检查移动止盈失败: {str(e)}")

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
        log_message("SUCCESS", f"{symbol} 移动止盈订单已更新，新价格: {new_tp_price:.4f}，订单ID: {tp_order['id']}")
        
    except Exception as e:
        log_message("ERROR", f"{symbol} 更新移动止盈订单失败: {str(e)}")

# ============================================
# 为已存在的持仓设置止损止盈条件单
# ============================================
def sync_exchange_positions():
    """同步交易所持仓，统一按MACD策略管理"""
    try:
        log_message("INFO", "正在同步交易所持仓，统一纳入MACD策略管理...")
        
        # 获取交易所当前持仓
        positions = exchange.fetch_positions()
        synced_count = 0
        
        for position in positions:
            if float(position['contracts']) > 0:  # 有持仓
                symbol = position['symbol']
                size = float(position['contracts'])
                side = position['side']  # 'long' 或 'short'
                entry_price = float(position['entryPrice'])
                unrealized_pnl = float(position.get('unrealizedPnl', 0))
                
                log_message("INFO", f"发现交易所持仓 {symbol}: {side} {size} @ {entry_price}")
                log_message("INFO", f"  当前未实现盈亏: {unrealized_pnl:.2f} USDT")
                
                # 检查本地跟踪器中是否有这个持仓的记录
                if symbol not in position_tracker['positions']:
                    # 本地没有记录，统一纳入MACD策略管理
                    log_message("WARNING", f"{symbol} 本地无记录，统一纳入MACD策略管理")
                    
                    # 使用统一策略计算止损止盈价格
                    sl_price, tp_price = calculate_stop_loss_take_profit(symbol, entry_price, side)
                    
                    # 获取智能杠杆（估算）
                    account_info = get_account_info()
                    if account_info:
                        smart_leverage = get_smart_leverage(symbol, account_info['total_balance'])
                    else:
                        smart_leverage = DEFAULT_LEVERAGE
                    
                    # 添加到本地跟踪器，统一按MACD策略管理
                    position_tracker['positions'][symbol] = {
                        'entry_price': entry_price,
                        'size': size,
                        'side': side,
                        'pnl': unrealized_pnl,
                        'sl': sl_price,
                        'tp': tp_price,
                        'entry_time': datetime.now(),  # 使用当前时间作为管理开始时间
                        'leverage': smart_leverage,
                        'order_id': None,  # 原始开仓订单ID未知
                        'sl_order_id': None,  # 将在后续补充
                        'tp_order_id': None,  # 将在后续补充
                        'strategy_managed': True,  # 标记为策略统一管理
                        'original_position': True  # 标记为程序启动前的原有持仓
                    }
                    
                    # 初始化移动止盈跟踪
                    if symbol not in trailing_stops:
                        trailing_stops[symbol] = {
                            'side': side,
                            'active': False,
                            'current_trigger': None,
                            'last_check': 0,
                            'atr_based': False
                        }
                    
                    synced_count += 1
                    log_message("SUCCESS", f"{symbol} 原有持仓已统一纳入MACD策略管理")
                    log_message("INFO", f"  策略止损: {sl_price:.6f}")
                    log_message("INFO", f"  策略止盈: {tp_price:.6f}")
                    log_message("INFO", f"  智能杠杆: {smart_leverage}x")
                    log_message("INFO", f"  移动止盈: 已初始化，等待激活条件")
                    
                else:
                    # 本地有记录，确保标记为策略管理
                    pos_data = position_tracker['positions'][symbol]
                    if not pos_data.get('strategy_managed', False):
                        pos_data['strategy_managed'] = True
                        pos_data['original_position'] = True
                        log_message("INFO", f"{symbol} 已有持仓纳入统一MACD策略管理")
                    
                    # 确保移动止盈跟踪已初始化
                    if symbol not in trailing_stops:
                        trailing_stops[symbol] = {
                            'side': side,
                            'active': False,
                            'current_trigger': None,
                            'last_check': 0,
                            'atr_based': False
                        }
                        log_message("INFO", f"{symbol} 移动止盈跟踪已初始化")
        
        if synced_count > 0:
            log_message("SUCCESS", f"成功同步 {synced_count} 个交易所持仓到MACD策略管理")
            log_message("INFO", "统一策略特性:")
            log_message("INFO", "  - 30分钟K线图")
            log_message("INFO", "  - MACD(6,32,9)金叉死叉信号")
            log_message("INFO", "  - ADX趋势过滤")
            log_message("INFO", "  - ATR动态止盈止损")
            log_message("INFO", "  - 智能移动止盈（利润保护）")
            log_message("INFO", "  - 智能杠杆分配")
        else:
            log_message("INFO", "未发现需要同步的交易所持仓")
                    
    except Exception as e:
        log_message("ERROR", f"同步交易所持仓时出错: {str(e)}")

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
                
                # 显示当前监控状态
                log_message("DEBUG", f"{symbol} 监控中 - 当前价格:{current_price:.6f}, 止损:{sl_price:.6f}, 止盈:{tp_price:.6f}, 盈亏:{pnl:.2f}")
                
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
                leverage_info = f"{pos.get('leverage', DEFAULT_LEVERAGE)}x"
                print(f"  {symbol}: {pos['side']} {pos['size']:.4f} @ {pos['entry_price']:.4f} | 杠杆: {leverage_info} | 盈亏: {pos['pnl']:.2f} USDT")
        
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
                
                # 同步交易所持仓（确保统一管理）
                sync_exchange_positions()
                
                # 更新持仓状态
                update_positions()
                
                # 检查并补充缺失的止损止盈条件单
                setup_missing_stop_orders()
                
                # API调用间隙
                time.sleep(1)
                
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
                        
                        # 避免请求过快，增加API调用间隙
                        time.sleep(2)  # 增加到2秒间隙
                        
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
        log_message("INFO", f"智能杠杆系统: 启用")
        log_message("INFO", f"  BTC最大杠杆: {MAX_LEVERAGE_BTC}x (最低20x)")
        log_message("INFO", f"  ETH最大杠杆: {MAX_LEVERAGE_ETH}x (最低20x)")
        log_message("INFO", f"  主流币最大杠杆: {MAX_LEVERAGE_MAJOR}x (最低20x)")
        log_message("INFO", f"  其他币种最大杠杆: {MAX_LEVERAGE_OTHERS}x (最低20x)")
        log_message("INFO", f"  全局最低杠杆: 20x")
        log_message("INFO", f"单次风险: {RISK_PER_TRADE*100}%")
        log_message("INFO", f"最大持仓: {MAX_OPEN_POSITIONS}")
        log_message("INFO", f"冷却期: {COOLDOWN_PERIOD//60}分钟")
        log_message("INFO", f"每日最大交易: {MAX_DAILY_TRADES}")
        log_message("INFO", "亏损限制: 已移除，可无限制下单")
        log_message("INFO", "使用30分钟K线图")
        log_message("INFO", "入场信号: MACD快线上穿/下穿慢线(金叉/死叉)")
        log_message("INFO", "震荡过滤: ADX < 20")
        log_message("INFO", "趋势确认: ADX > 25")
        log_message("INFO", "平仓条件: MACD反向交叉")
        log_message("INFO", "MACD平仓规则: 做多时MACD死叉平仓，做空时MACD金叉平仓")
        log_message("INFO", f"ATR动态止盈止损: {'启用' if USE_ATR_DYNAMIC_STOPS else '禁用'}")
        if USE_ATR_DYNAMIC_STOPS:
            log_message("INFO", f"ATR止损倍数: {ATR_STOP_LOSS_MULTIPLIER}x")
            log_message("INFO", f"ATR止盈倍数: {ATR_TAKE_PROFIT_MULTIPLIER}x")
            log_message("INFO", f"ATR移动止盈激活倍数: {ATR_TRAILING_ACTIVATION_MULTIPLIER}x")
            log_message("INFO", f"ATR移动止盈回调倍数: {ATR_TRAILING_CALLBACK_MULTIPLIER}x")
        log_message("INFO", f"交易对: 热度前10 + FIL, ZRO, WIF, WLD (共14个)")
        log_message("SUCCESS", "=" * 60)
        
        # 同步交易所持仓，统一按MACD策略管理
        sync_exchange_positions()
        
        # 启动交易循环
        trading_loop()
        
    except Exception as e:
        log_message("ERROR", f"启动交易系统失败: {str(e)}")
        traceback.print_exc()

# ============================================
# 胜率统计和回测模块
# ============================================

def update_trade_stats(symbol, side, pnl, entry_price, exit_price):
    """更新交易统计数据"""
    global trade_stats
    
    try:
        trade_stats['total_trades'] += 1
        trade_stats['total_pnl'] += pnl
        
        if pnl > 0:
            trade_stats['winning_trades'] += 1
            trade_stats['total_profit'] += pnl
        else:
            trade_stats['losing_trades'] += 1
            trade_stats['total_loss'] += abs(pnl)
        
        # 计算胜率
        if trade_stats['total_trades'] > 0:
            trade_stats['win_rate'] = (trade_stats['winning_trades'] / trade_stats['total_trades']) * 100
        
        # 记录交易历史
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
        
        # 保持最近100笔交易记录
        if len(trade_stats['trade_history']) > 100:
            trade_stats['trade_history'] = trade_stats['trade_history'][-100:]
        
        log_message("INFO", f"交易统计更新:")
        log_message("INFO", f"  总交易次数: {trade_stats['total_trades']}")
        log_message("INFO", f"  胜率: {trade_stats['win_rate']:.2f}%")
        log_message("INFO", f"  总盈亏: {trade_stats['total_pnl']:.2f} USDT")
        
    except Exception as e:
        log_message("ERROR", f"更新交易统计失败: {str(e)}")

def get_performance_metrics():
    """获取详细的性能指标"""
    try:
        if trade_stats['total_trades'] == 0:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'average_win': 0,
                'average_loss': 0,
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0
            }
        
        # 计算盈利因子
        profit_factor = 0
        if trade_stats['total_loss'] > 0:
            profit_factor = trade_stats['total_profit'] / trade_stats['total_loss']
        
        # 计算平均盈利和亏损
        avg_win = trade_stats['total_profit'] / trade_stats['winning_trades'] if trade_stats['winning_trades'] > 0 else 0
        avg_loss = trade_stats['total_loss'] / trade_stats['losing_trades'] if trade_stats['losing_trades'] > 0 else 0
        
        # 计算最大连续盈利和亏损
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in trade_stats['trade_history']:
            if trade['pnl'] > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
        
        return {
            'total_trades': trade_stats['total_trades'],
            'win_rate': trade_stats['win_rate'],
            'profit_factor': profit_factor,
            'average_win': avg_win,
            'average_loss': avg_loss,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'total_pnl': trade_stats['total_pnl']
        }
        
    except Exception as e:
        log_message("ERROR", f"获取性能指标失败: {str(e)}")
        return {}

def historical_backtest(symbol, days=30):
    """简单的历史回测功能"""
    try:
        log_message("INFO", f"开始{symbol}历史回测 (过去{days}天)")
        
        # 获取历史数据
        since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        ohlcv = exchange.fetch_ohlcv(symbol, '30m', since=since, limit=1000)
        
        if not ohlcv or len(ohlcv) < 50:
            log_message("WARNING", f"{symbol} 历史数据不足，跳过回测")
            return None
        
        # 处理数据并计算指标
        df = process_klines(ohlcv)
        if df is None:
            log_message("WARNING", f"{symbol} 数据处理失败，跳过回测")
            return None
        
        # 模拟交易
        backtest_results = {
            'symbol': symbol,
            'total_signals': 0,
            'profitable_signals': 0,
            'total_return': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'trades': []
        }
        
        position = None
        equity_curve = [10000]  # 假设起始资金10000 USDT
        peak_equity = 10000
        
        for i in range(50, len(df)):  # 从第50根K线开始，确保指标计算完整
            current_price = df.iloc[i]['close']
            current_macd = df.iloc[i]['MACD']
            current_signal = df.iloc[i]['MACD_SIGNAL']
            prev_macd = df.iloc[i-1]['MACD']
            prev_signal = df.iloc[i-1]['MACD_SIGNAL']
            current_adx = df.iloc[i]['ADX']
            
            # 检查信号
            golden_cross = prev_macd <= prev_signal and current_macd > current_signal
            death_cross = prev_macd >= prev_signal and current_macd < current_signal
            
            # 开仓逻辑
            if position is None and current_adx > 25:
                if golden_cross:
                    position = {
                        'side': 'long',
                        'entry_price': current_price,
                        'entry_time': i
                    }
                    backtest_results['total_signals'] += 1
                elif death_cross:
                    position = {
                        'side': 'short', 
                        'entry_price': current_price,
                        'entry_time': i
                    }
                    backtest_results['total_signals'] += 1
            
            # 平仓逻辑
            elif position is not None:
                should_close = False
                
                if position['side'] == 'long' and death_cross:
                    should_close = True
                elif position['side'] == 'short' and golden_cross:
                    should_close = True
                
                if should_close:
                    # 计算收益
                    if position['side'] == 'long':
                        pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                    else:
                        pnl_pct = (position['entry_price'] - current_price) / position['entry_price']
                    
                    pnl_amount = equity_curve[-1] * pnl_pct * 0.8  # 假设使用80%资金
                    new_equity = equity_curve[-1] + pnl_amount
                    
                    equity_curve.append(new_equity)
                    peak_equity = max(peak_equity, new_equity)
                    
                    # 记录交易
                    trade_record = {
                        'side': position['side'],
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'pnl_pct': pnl_pct * 100,
                        'pnl_amount': pnl_amount,
                        'duration': i - position['entry_time']
                    }
                    
                    backtest_results['trades'].append(trade_record)
                    backtest_results['total_return'] += pnl_pct * 100
                    
                    if pnl_pct > 0:
                        backtest_results['profitable_signals'] += 1
                    
                    position = None
        
        # 计算最终指标
        if backtest_results['total_signals'] > 0:
            backtest_results['win_rate'] = (backtest_results['profitable_signals'] / backtest_results['total_signals']) * 100
        
        # 计算最大回撤
        for equity in equity_curve:
            drawdown = (peak_equity - equity) / peak_equity * 100
            backtest_results['max_drawdown'] = max(backtest_results['max_drawdown'], drawdown)
            if equity > peak_equity:
                peak_equity = equity
        
        log_message("INFO", f"{symbol} 回测结果:")
        log_message("INFO", f"  信号总数: {backtest_results['total_signals']}")
        log_message("INFO", f"  胜率: {backtest_results['win_rate']:.2f}%")
        log_message("INFO", f"  总收益率: {backtest_results['total_return']:.2f}%")
        log_message("INFO", f"  最大回撤: {backtest_results['max_drawdown']:.2f}%")
        
        return backtest_results
        
    except Exception as e:
        log_message("ERROR", f"历史回测失败: {str(e)}")
        return None

# ============================================
# 更新持仓信息
# ============================================

def update_positions():
    """更新所有持仓信息"""
    try:
        positions = exchange.fetch_positions()
        active_positions = [pos for pos in positions if float(pos['contracts']) != 0]
        
        for position in active_positions:
            symbol = position['symbol']
            size = float(position['contracts'])
            side = position['side']
            entry_price = float(position['entryPrice']) if position['entryPrice'] else 0
            mark_price = float(position['markPrice']) if position['markPrice'] else 0
            pnl = float(position['unrealizedPnl']) if position['unrealizedPnl'] else 0
            
            # 更新持仓跟踪器
            position_tracker['positions'][symbol] = {
                'size': size,
                'side': side,
                'entry_price': entry_price,
                'current_price': mark_price,
                'unrealized_pnl': pnl,
                'last_update': datetime.now()
            }
            
            # 检查移动止盈
            check_trailing_stop(symbol, mark_price)
        
        # 清理已平仓的持仓记录
        current_symbols = {pos['symbol'] for pos in active_positions}
        symbols_to_remove = []
        
        for symbol in position_tracker['positions']:
            if symbol not in current_symbols:
                symbols_to_remove.append(symbol)
        
        for symbol in symbols_to_remove:
            log_message("INFO", f"清理已平仓持仓记录: {symbol}")
            if symbol in position_tracker['positions']:
                del position_tracker['positions'][symbol]
            if symbol in position_tracker['trailing_stops']:
                del position_tracker['trailing_stops'][symbol]
        
        log_message("INFO", f"持仓更新完成，当前活跃持仓: {len(active_positions)}")
        
    except Exception as e:
        log_message("ERROR", f"更新持仓信息失败: {str(e)}")

# ============================================
# 平仓函数
# ============================================

def close_position(symbol, reason="手动平仓"):
    """平仓指定交易对"""
    try:
        if symbol not in position_tracker['positions']:
            log_message("WARNING", f"{symbol} 无持仓记录")
            return False
        
        position_info = position_tracker['positions'][symbol]
        side = position_info['side']
        size = abs(position_info['size'])
        entry_price = position_info.get('entry_price', 0)
        
        # 获取当前价格
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        
        # 执行平仓
        close_side = 'sell' if side == 'long' else 'buy'
        
        log_message("INFO", f"执行平仓: {symbol} {close_side} {size} @ {current_price}")
        
        order = exchange.create_market_order(symbol, close_side, size, None, None, {
            'reduceOnly': True
        })
        
        if order:
            log_message("SUCCESS", f"平仓成功: {symbol} {reason}")
            
            # 计算盈亏
            if side == 'long':
                pnl = (current_price - entry_price) * size
            else:
                pnl = (entry_price - current_price) * size
            
            # 更新交易统计
            update_trade_stats(symbol, side, pnl, entry_price, current_price)
            
            # 清理持仓记录
            if symbol in position_tracker['positions']:
                del position_tracker['positions'][symbol]
            if symbol in position_tracker['trailing_stops']:
                del position_tracker['trailing_stops'][symbol]
            
            return True
        else:
            log_message("ERROR", f"平仓失败: {symbol}")
            return False
            
    except Exception as e:
        log_message("ERROR", f"平仓 {symbol} 失败: {str(e)}")
        return False

# ============================================
                    
=======
            break
        except Exception as e:
            log_message("ERROR", f"交易循环异常: {str(e)}")
            traceback.print_exc()
            time.sleep(60)  # 异常后等待1分钟再继续
=======
            # 主循环延迟
            log_message("INFO", f"交易循环完成，等待{MAIN_LOOP_DELAY}秒...")
            time.sleep(MAIN_LOOP_DELAY)
            
        except KeyboardInterrupt:
=======
def main_loop():
    """主循环入口（兼容性函数）"""
    trading_loop()

# ============================================
# 主程序入口
# ============================================
if __name__ == "__main__":
    start_trading_system()
=======
=======
            break
        except Exception as e:
            log_message("ERROR", f"交易循环异常: {str(e)}")
            traceback.print_exc()
            time.sleep(60)  # 异常后等待1分钟再继续
=======
            # 主循环延迟
            log_message("INFO", f"交易循环完成，等待{MAIN_LOOP_DELAY}秒...")
            time.sleep(MAIN_LOOP_DELAY)
            
        except KeyboardInterrupt:
            log_message("INFO", "收到退出信号，正在安全关闭...")
            break
        except Exception as e:
            log_message("ERROR", f"交易循环异常: {str(e)}")
            traceback.print_exc()
            time.sleep(60)  # 异常后等待1分钟再继续

def main_loop():
    """主循环入口（兼容性函数）"""
    trading_loop()

# ============================================
# 主程序入口
# ============================================
if __name__ == "__main__":
    start_trading_system()
=======
# ============================================
# 主程序入口
# ============================================
if __name__ == "__main__":
    start_trading_system()
=======
=======
            break
        except Exception as e:
            log_message("ERROR", f"交易循环异常: {str(e)}")
            traceback.print_exc()
            time.sleep(60)  # 异常后等待1分钟再继续

def main_loop():
    """主循环入口（兼容性函数）"""
    trading_loop()

# ============================================
# 主程序入口
# ============================================
if __name__ == "__main__":
    start_trading_system()
=======
=======
def main_loop():
    """主循环入口（兼容性函数）"""
    trading_loop()

# ============================================
# 主程序入口
# ============================================
if __name__ == "__main__":
    start_trading_system()
=======
# ============================================
# 主程序入口
# ============================================
if __name__ == "__main__":
    start_trading_system()
=======
=======
        except KeyboardInterrupt:
            log_message("INFO", "收到退出信号，正在安全关闭...")
            break
        except Exception as e:
            log_message("ERROR", f"交易循环异常: {str(e)}")
            traceback.print_exc()
            time.sleep(60)  # 异常后等待1分钟再继续

def main_loop():
    """主循环入口（兼容性函数）"""
    trading_loop()

# ============================================
# 主程序入口
# ============================================
if __name__ == "__main__":
    start_trading_system()
=======
def main_loop():
    """主循环入口（兼容性函数）"""
    trading_loop()

# ============================================
# 主程序入口
# ============================================
if __name__ == "__main__":
    start_trading_system()
=======
# ============================================
# 主程序入口
# ============================================
if __name__ == "__main__":
    start_trading_system()
=======
=======
            break
        except Exception as e:
            log_message("ERROR", f"交易循环异常: {str(e)}")
            traceback.print_exc()
            time.sleep(60)  # 异常后等待1分钟再继续

def main_loop():
    """主循环入口（兼容性函数）"""
    trading_loop()

# ============================================
# 主程序入口
# ============================================
if __name__ == "__main__":
    start_trading_system()
=======
=======
def main_loop():
    """主循环入口（兼容性函数）"""
    trading_loop()

# ============================================
# 主程序入口
# ============================================
if __name__ == "__main__":
    start_trading_system()
=======
# ============================================
# 主程序入口
# ============================================
if __name__ == "__main__":
    start_trading_system()
=======