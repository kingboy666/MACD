#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bollinger Bands Strategy (OKX USDT-SWAP, 15m) - Enhanced Version
- Indicator: Bollinger Bands(18,2) + ADX + ATR
- Multi-layer risk management with dynamic stop loss
- Enhanced range-bound trading with multiple confirmations
"""
import os
import time
import math
import logging
from typing import Dict, Any
import json
import urllib.request

import ccxt
import pandas as pd

LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').strip().upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('bollinger-enhanced')

# 通知配置
NOTIFY_WEBHOOK = os.environ.get('NOTIFY_WEBHOOK', '').strip()
NOTIFY_TYPE = os.environ.get('NOTIFY_TYPE', '').strip().lower()
NOTIFY_MENTION_MOBILES = [m.strip() for m in os.environ.get('NOTIFY_MENTION_MOBILES', '').split(',') if m.strip()]
PUSHPLUS_TOKEN = os.environ.get('PUSHPLUS_TOKEN', '').strip()
WXPUSHER_APP_TOKEN = os.environ.get('WXPUSHER_APP_TOKEN', '').strip()
WXPUSHER_UID = os.environ.get('WXPUSHER_UID', '').strip()

def _post_json(url: str, payload: dict):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    urllib.request.urlopen(req, timeout=5).read()

def notify_event(title: str, message: str, level: str = 'info'):
    if NOTIFY_TYPE in ('wecom', 'feishu', 'ding', 'generic') and not NOTIFY_WEBHOOK:
        return
    try:
        if NOTIFY_TYPE == 'wecom':
            content = f"【{title}】\n{message}"
            payload = {
                'msgtype': 'text',
                'text': {
                    'content': content,
                    'mentioned_mobile_list': NOTIFY_MENTION_MOBILES or []
                }
            }
            _post_json(NOTIFY_WEBHOOK, payload)
        elif NOTIFY_TYPE == 'feishu':
            payload = {
                'msg_type': 'text',
                'content': { 'text': f"【{title}】\n{message}" }
            }
            _post_json(NOTIFY_WEBHOOK, payload)
        elif NOTIFY_TYPE == 'ding':
            payload = {
                'msgtype': 'text',
                'text': { 'content': f"【{title}】\n{message}" }
            }
            _post_json(NOTIFY_WEBHOOK, payload)
        elif NOTIFY_TYPE == 'pushplus':
            if PUSHPLUS_TOKEN:
                payload = {
                    'token': PUSHPLUS_TOKEN,
                    'title': title,
                    'content': message,
                    'template': 'txt'
                }
                _post_json('https://www.pushplus.plus/send', payload)
        elif NOTIFY_TYPE == 'wxpusher':
            if WXPUSHER_APP_TOKEN and WXPUSHER_UID:
                payload = {
                    'appToken': WXPUSHER_APP_TOKEN,
                    'content': f"【{title}】\n{message}",
                    'summary': title,
                    'contentType': 1,
                    'uids': [WXPUSHER_UID]
                }
                _post_json('https://wxpusher.zjiecode.com/api/send/message', payload)
        else:
            payload = {
                'title': title,
                'message': message,
                'level': level,
                'ts': int(time.time())
            }
            _post_json(NOTIFY_WEBHOOK, payload)
    except Exception as e:
        log.warning(f'notify_event failed: {e}')

API_KEY = os.environ.get('OKX_API_KEY', '').strip()
API_SECRET = os.environ.get('OKX_SECRET_KEY', '').strip()
API_PASS = os.environ.get('OKX_PASSPHRASE', '').strip()
DRY_RUN = os.environ.get('DRY_RUN', 'false').strip().lower() in ('1', 'true', 'yes')
if not API_KEY or not API_SECRET or not API_PASS:
    if not DRY_RUN:
        raise SystemExit('Missing OKX credentials: set OKX_API_KEY, OKX_SECRET_KEY, OKX_PASSPHRASE')
    else:
        log.warning('Running in DRY_RUN mode without OKX credentials; no orders will be placed.')

BUDGET_USDT = float(os.environ.get('BUDGET_USDT', '5').strip() or 5)
DEFAULT_LEVERAGE = int(float(os.environ.get('DEFAULT_LEVERAGE', '20').strip() or 20))
# 动态止损使用ATR倍数，固定止盈作为兜底
TP_PCT = float(os.environ.get('TP_PCT', '0.035').strip() or 0.035)  # 3.5% 兜底止盈
SL_ATR_MULTIPLIER = float(os.environ.get('SL_ATR_MULTIPLIER', '2.0').strip() or 2.0)  # ATR止损倍数
SCAN_INTERVAL = int(float(os.environ.get('SCAN_INTERVAL', '10').strip() or 10))
USE_BALANCE_AS_MARGIN = os.environ.get('USE_BALANCE_AS_MARGIN', 'true').strip().lower() in ('1', 'true', 'yes')
MARGIN_UTILIZATION = float(os.environ.get('MARGIN_UTILIZATION', '0.95').strip() or 0.95)
# 峰值追踪止盈配置
TRAIL_ENABLE = os.environ.get('TRAIL_ENABLE', 'true').strip().lower() in ('1', 'true', 'yes')
TRAIL_DD_PCT = float(os.environ.get('TRAIL_DD_PCT', '0.05').strip() or 0.05)  # 从峰值回撤阈值（5%）
TRAIL_REQUIRE_PROFIT = os.environ.get('TRAIL_REQUIRE_PROFIT', 'true').strip().lower() in ('1', 'true', 'yes')  # 仅在持仓为正收益时触发

# 布林带参数
BB_PERIOD = int(os.environ.get('BB_PERIOD', '18').strip() or 18)
BB_STD = float(os.environ.get('BB_STD', '2.0').strip() or 2.0)
BB_SLOPE_PERIOD = int(os.environ.get('BB_SLOPE_PERIOD', '5').strip() or 5)
# ADX 过滤参数
ADX_PERIOD = int(os.environ.get('ADX_PERIOD', '14').strip() or 14)
ADX_MIN_TREND = float(os.environ.get('ADX_MIN_TREND', '20').strip() or 20)
# ATR 参数
ATR_PERIOD = int(os.environ.get('ATR_PERIOD', '14').strip() or 14)

# 趋势判断阈值
SLOPE_UP_THRESH = float(os.environ.get('SLOPE_UP_THRESH', '0.0015').strip() or 0.0015)
SLOPE_DOWN_THRESH = float(os.environ.get('SLOPE_DOWN_THRESH', '-0.0015').strip() or -0.0015)
SLOPE_FLAT_RANGE = float(os.environ.get('SLOPE_FLAT_RANGE', '0.0008').strip() or 0.0008)

# 带宽变化阈值
BANDWIDTH_EXPAND_THRESH = float(os.environ.get('BANDWIDTH_EXPAND_THRESH', '0.12').strip() or 0.12)
BANDWIDTH_SQUEEZE_THRESH = float(os.environ.get('BANDWIDTH_SQUEEZE_THRESH', '-0.12').strip() or -0.12)

# 价格位置容差
PRICE_TOLERANCE = float(os.environ.get('PRICE_TOLERANCE', '0.002').strip() or 0.002)

# 震荡市增强确认参数
MIN_RISK_REWARD = float(os.environ.get('MIN_RISK_REWARD', '2.0').strip() or 2.0)  # 最小盈亏比
HAMMER_SHADOW_RATIO = float(os.environ.get('HAMMER_SHADOW_RATIO', '2.0').strip() or 2.0)  # 锤子线影线/实体比

# 下降趋势抢反弹开关
ENABLE_DOWNTREND_BOUNCE = os.environ.get('ENABLE_DOWNTREND_BOUNCE', 'false').strip().lower() in ('1', 'true', 'yes')
DOWNTREND_POSITION_RATIO = float(os.environ.get('DOWNTREND_POSITION_RATIO', '0.3').strip() or 0.3)

TIMEFRAME = '15m'
SYMBOLS = [
    'FIL/USDT:USDT', 'ZRO/USDT:USDT', 'WIF/USDT:USDT', 'WLD/USDT:USDT',
    'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'XRP/USDT:USDT',
    'ARB/USDT:USDT'
]

SYMBOL_LEVERAGE: Dict[str, int] = {
    'FIL/USDT:USDT': 50,
    'ZRO/USDT:USDT': 20,
    'WIF/USDT:USDT': 50,
    'WLD/USDT:USDT': 50,
    'BTC/USDT:USDT': 100,
    'ETH/USDT:USDT': 100,
    'SOL/USDT:USDT': 50,
    'XRP/USDT:USDT': 50,
    'ARB/USDT:USDT': 50,
}

exchange = ccxt.okx({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'password': API_PASS,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'swap',
        'types': ['swap'],
    }
})

POS_MODE = os.environ.get('POS_MODE', 'net').strip().lower()

def ensure_position_mode():
    try:
        mode = 'long_short_mode' if POS_MODE == 'hedge' else 'net_mode'
        exchange.privatePostAccountSetPositionMode({'posMode': mode})
        log.info(f'已设置持仓模式 -> {"双向对冲" if POS_MODE == "hedge" else "单向净持仓"}')
    except Exception as e:
        log.warning(f'设置持仓模式失败: {e}')

def symbol_to_inst_id(sym: str) -> str:
    base = sym.split('/')[0]
    return f'{base}-USDT-SWAP'

markets_info: Dict[str, Dict[str, Any]] = {}

def load_market_info(symbol: str) -> Dict[str, Any]:
    if symbol in markets_info:
        return markets_info[symbol]
    inst_id = symbol_to_inst_id(symbol)
    resp = exchange.publicGetPublicInstruments({'instType': 'SWAP', 'instId': inst_id})
    data = (resp.get('data') or [])[0]
    info = {
        'instId': inst_id,
        'ctVal': float(data.get('ctVal', 0) or 0),
        'ctType': data.get('ctType'),
        'lotSz': float(data.get('lotSz', 0) or 0),
        'minSz': float(data.get('minSz', 0) or 0),
        'tickSz': float(data.get('tickSz', 0) or 0),
        'pxTick': float(data.get('tickSz', 0) or 0),
    }
    markets_info[symbol] = info
    return info

def ensure_leverage(symbol: str):
    lev = int(SYMBOL_LEVERAGE.get(symbol, DEFAULT_LEVERAGE) or DEFAULT_LEVERAGE)
    inst_id = symbol_to_inst_id(symbol)
    try:
        exchange.privatePostAccountSetLeverage({'instId': inst_id, 'mgnMode': 'cross', 'lever': str(lev)})
        log.info(f'已设置杠杆 {symbol} -> {lev}倍')
    except Exception as e:
        log.warning(f'设置杠杆失败 {symbol}: {e}')

def get_position(symbol: str) -> Dict[str, Any]:
    inst_id = symbol_to_inst_id(symbol)
    try:
        resp = exchange.privateGetAccountPositions({'instType': 'SWAP', 'instId': inst_id})
        for p in resp.get('data', []):
            if p.get('instId') == inst_id and float(p.get('pos', 0) or 0) != 0:
                size = abs(float(p.get('pos', 0) or 0))
                side = 'long' if p.get('posSide', 'net') in ('long', 'net') and float(p.get('pos', 0)) > 0 else 'short'
                entry = float(p.get('avgPx') or p.get('lastAvgPrice') or p.get('avgPrice') or 0)
                return {'size': size, 'side': side, 'entry': entry}
    except Exception as e:
        log.warning(f'get_position failed: {e}')
    return {'size': 0.0, 'side': None, 'entry': 0.0}

def get_positions_both(symbol: str) -> Dict[str, Dict[str, float]]:
    """返回该合约的多空独立持仓信息"""
    res = {'long': {'size': 0.0, 'entry': 0.0}, 'short': {'size': 0.0, 'entry': 0.0}}
    inst_id = symbol_to_inst_id(symbol)
    try:
        resp = exchange.privateGetAccountPositions({'instType': 'SWAP', 'instId': inst_id})
        for p in resp.get('data', []):
            if p.get('instId') != inst_id:
                continue
            pos = float(p.get('pos', 0) or 0)
            if pos == 0:
                continue
            pos_side = p.get('posSide', 'net')
            entry = float(p.get('avgPx') or p.get('lastAvgPrice') or p.get('avgPrice') or 0)
            if pos_side == 'long' or (pos_side == 'net' and pos > 0):
                res['long'] = {'size': abs(pos), 'entry': entry}
            elif pos_side == 'short' or (pos_side == 'net' and pos < 0):
                res['short'] = {'size': abs(pos), 'entry': entry}
    except Exception as e:
        log.debug(f'get_positions_both failed {symbol}: {e}')
    return res

def place_market_order(symbol: str, side: str, budget_usdt: float, position_ratio: float = 1.0) -> bool:
    """市价下单"""
    if DRY_RUN:
        log.info(f'[DRY_RUN] 模拟开仓 {symbol} {side} 仓位比例={position_ratio*100:.0f}%')
        return True
    try:
        balance = exchange.fetch_balance()
        avail = float(balance.get('USDT', {}).get('free') or balance.get('USDT', {}).get('available') or 0)
    except Exception:
        avail = 0.0
    equity_usdt = max(0.0, avail) * position_ratio
    if equity_usdt <= 0:
        log.warning('No available USDT balance to open position')
        return False

    info = load_market_info(symbol)
    inst_id = info['instId']
    ticker = exchange.fetch_ticker(symbol)
    price = float(ticker.get('last') or ticker.get('close') or 0)
    if price <= 0:
        raise Exception('invalid price')
    ct_val = float(info.get('ctVal') or 0)
    if ct_val <= 0:
        ct_val = 0.01

    if USE_BALANCE_AS_MARGIN:
        leverage = SYMBOL_LEVERAGE.get(symbol, DEFAULT_LEVERAGE)
        target_notional = equity_usdt * max(1, leverage) * MARGIN_UTILIZATION
        contracts = (target_notional / price) / ct_val
    else:
        contracts = (equity_usdt / price) / ct_val

    lot = float(info.get('lotSz') or 0)
    minsz = float(info.get('minSz') or 0)
    if lot > 0:
        contracts = math.floor(contracts / lot) * lot
    if contracts <= 0 or (minsz > 0 and contracts < minsz):
        log.warning(f'计算得到的合约张数过小: {contracts}, 最小下单={minsz}, 步长={lot}')
        return False
    
    side_okx = 'buy' if side == 'buy' else 'sell'
    params = {
        'instId': inst_id,
        'tdMode': 'cross',
        'side': side_okx,
        'ordType': 'market',
        'sz': str(contracts),
    }
    if POS_MODE == 'hedge':
        params['posSide'] = 'long' if side_okx == 'buy' else 'short'
    
    try:
        exchange.privatePostTradeOrder(params)
        log.info(f'下单成功 {symbol}: 方向={side} 数量={contracts}, 预算={equity_usdt:.2f}USDT (仓位比例={position_ratio*100:.0f}%)')
        notify_event('开仓成功', f'{symbol} {side} 数量={contracts} 预算={equity_usdt:.2f}U 比例={position_ratio*100:.0f}%')
        return True
    except Exception as e:
        log.warning(f'下单失败 {symbol}: {e}')
        return False

def close_position_market(symbol: str, side_to_close: str, qty: float) -> bool:
    """市价平仓"""
    info = load_market_info(symbol)
    inst_id = info['instId']
    side_okx = 'sell' if side_to_close == 'long' else 'buy'
    lot = float(info.get('lotSz') or 0)
    minsz = float(info.get('minSz') or 0)
    sz = qty
    if lot > 0:
        sz = math.floor(sz / lot) * lot
    sz = min(sz, qty)
    if sz <= 0 or (minsz > 0 and sz < minsz):
        log.warning(f'平仓数量过小: {sz}')
        return False
    
    params = {
        'instId': inst_id,
        'tdMode': 'cross',
        'side': side_okx,
        'ordType': 'market',
        'sz': str(sz),
        'reduceOnly': True,
    }
    if POS_MODE == 'hedge':
        params['posSide'] = 'long' if side_to_close == 'long' else 'short'
    
    if DRY_RUN:
        log.info(f'[DRY_RUN] 模拟平仓 {symbol} 方向={side_to_close} 数量={sz}')
        return True
    try:
        exchange.privatePostTradeOrder(params)
        log.info(f'已市价平仓 {symbol} 方向={side_to_close} 数量={sz}')
        notify_event('已市价平仓', f'{symbol} 方向={side_to_close} 数量={sz}')
        return True
    except Exception as e:
        log.warning(f'平仓失败 {symbol}: {e}')
        return False

# ========== 技术指标计算 ==========

def calculate_bollinger_bands(closes: pd.Series, period: int = 20, std_multiplier: float = 2.0):
    """计算布林带"""
    middle = closes.rolling(window=period).mean()
    std = closes.rolling(window=period).std()
    upper = middle + std_multiplier * std
    lower = middle - std_multiplier * std
    bandwidth = (upper - lower) / middle
    return upper, middle, lower, bandwidth

def calc_atr(highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 14) -> pd.Series:
    """计算ATR（平均真实波幅）"""
    tr_components = pd.concat([
        (highs - lows).abs(),
        (highs - closes.shift()).abs(),
        (lows - closes.shift()).abs()
    ], axis=1)
    tr = tr_components.max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

def calc_adx(highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 14) -> pd.Series:
    """计算ADX（平均趋向指标）"""
    up_move = highs.diff()
    down_move = lows.diff().abs()
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move.fillna(0)
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move.fillna(0)
    tr_components = pd.concat([
        (highs - lows).abs(),
        (highs - closes.shift()).abs(),
        (lows - closes.shift()).abs()
    ], axis=1)
    tr = tr_components.max(axis=1)
    atr = tr.rolling(period).mean()
    atr_safe = atr.replace(0, pd.NA)
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr_safe)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr_safe)
    denom = (plus_di + minus_di).replace(0, pd.NA)
    dx = (abs(plus_di - minus_di) / denom) * 100
    adx = dx.rolling(period).mean().fillna(0)
    return adx

def detect_bb_trend(middle: pd.Series, lookback: int = 5):
    """判断中线方向：up/down/flat"""
    if len(middle) < lookback + 1:
        return 'flat'
    slope = (middle.iloc[-1] - middle.iloc[-lookback-1]) / middle.iloc[-lookback-1]
    if slope > SLOPE_UP_THRESH:
        return 'up'
    elif slope < SLOPE_DOWN_THRESH:
        return 'down'
    elif abs(slope) <= SLOPE_FLAT_RANGE:
        return 'flat'
    else:
        return 'flat'

def detect_bandwidth_change(bandwidth: pd.Series, lookback: int = 5):
    """判断开口/收口：expanding/squeezing/stable"""
    if len(bandwidth) < lookback + 1:
        return 'stable'
    change = (bandwidth.iloc[-1] - bandwidth.iloc[-lookback-1]) / bandwidth.iloc[-lookback-1]
    if change > BANDWIDTH_EXPAND_THRESH:
        return 'expanding'
    elif change < BANDWIDTH_SQUEEZE_THRESH:
        return 'squeezing'
    else:
        return 'stable'

def check_hammer_pattern(ohlcv_data, index: int = -1) -> bool:
    """检查是否有锤子线形态（看涨反转）"""
    if len(ohlcv_data) < abs(index) + 1:
        return False
    bar = ohlcv_data[index]
    open_price = bar[1]
    high = bar[2]
    low = bar[3]
    close = bar[4]
    
    # 实体大小
    body = abs(close - open_price)
    if body == 0:
        return False
    
    # 下影线
    lower_shadow = min(open_price, close) - low
    # 上影线
    upper_shadow = high - max(open_price, close)
    
    # 锤子线特征：下影线 >= 实体的N倍，且上影线很小
    is_hammer = (lower_shadow >= body * HAMMER_SHADOW_RATIO and upper_shadow < body * 0.5)
    return is_hammer

def check_shooting_star_pattern(ohlcv_data, index: int = -1) -> bool:
    """检查是否有流星线形态（看跌反转）"""
    if len(ohlcv_data) < abs(index) + 1:
        return False
    bar = ohlcv_data[index]
    open_price = bar[1]
    high = bar[2]
    low = bar[3]
    close = bar[4]
    
    body = abs(close - open_price)
    if body == 0:
        return False
    
    lower_shadow = min(open_price, close) - low
    upper_shadow = high - max(open_price, close)
    
    # 流星线特征：上影线 >= 实体的N倍，且下影线很小
    is_star = (upper_shadow >= body * HAMMER_SHADOW_RATIO and lower_shadow < body * 0.5)
    return is_star

def calculate_risk_reward(entry_price: float, target_price: float, stop_price: float) -> float:
    """计算盈亏比"""
    if entry_price == 0 or stop_price == 0:
        return 0.0
    potential_profit = abs(target_price - entry_price)
    potential_loss = abs(entry_price - stop_price)
    if potential_loss == 0:
        return 0.0
    return potential_profit / potential_loss

# ========== 主策略逻辑 ==========

last_bar_ts: Dict[str, int] = {}
symbol_state: Dict[str, Dict[str, Any]] = {}

log.info('=' * 70)
log.info(f'Start Enhanced Bollinger Bands Strategy - {TIMEFRAME} timeframe')
log.info(f'布林带参数: 周期={BB_PERIOD}, 标准差={BB_STD}倍')
log.info(f'动态止损: ATR倍数={SL_ATR_MULTIPLIER}')
log.info(f'震荡市确认: 盈亏比>={MIN_RISK_REWARD}, K线形态验证')
log.info('=' * 70)

if not DRY_RUN:
    ensure_position_mode()
    for sym in SYMBOLS:
        ensure_leverage(sym)
else:
    log.warning('DRY_RUN 开启：跳过设置持仓模式与杠杆')

for sym in SYMBOLS:
    symbol_state[sym] = {'trend': 'unknown', 'bandwidth_status': 'unknown', 'peak_long': None, 'peak_short': None}

stats = {'trades': 0, 'wins': 0, 'losses': 0, 'realized_pnl': 0.0}
cycle_count = 0

def get_last_closed_bar_ts(ohlcv_row):
    return int(ohlcv_row[0])

while True:
    try:
        cycle_count += 1
        if DRY_RUN:
            free, total = 0.0, 0.0
        else:
            try:
                balance = exchange.fetch_balance()
                usdt = balance.get('USDT', {})
                free = float(usdt.get('free') or usdt.get('available') or 0)
                used = float(usdt.get('used') or 0)
                total = float(usdt.get('total') or (free + used))
            except Exception:
                free, total = 0.0, 0.0
        
        winrate = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0.0
        log.info(f'周期 {cycle_count}: 扫描 {len(SYMBOLS)} 个交易对, 间隔={SCAN_INTERVAL}s | USDT 可用={free:.2f} 总额={total:.2f} | 累计已实现盈亏={stats["realized_pnl"]:.2f} | 胜率={winrate:.1f}% ({stats["wins"]}/{stats["trades"]})')
        
        for symbol in SYMBOLS:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=60)
                if not ohlcv or len(ohlcv) < BB_PERIOD + 10:
                    log.debug(f'{symbol} insufficient OHLCV: {0 if not ohlcv else len(ohlcv)}')
                    continue
                
                closes = pd.Series([c[4] for c in ohlcv])
                highs = pd.Series([c[2] for c in ohlcv])
                lows = pd.Series([c[3] for c in ohlcv])
                
                # 计算技术指标
                upper, middle, lower, bandwidth = calculate_bollinger_bands(closes, BB_PERIOD, BB_STD)
                atr = calc_atr(highs, lows, closes, ATR_PERIOD)
                adx = calc_adx(highs, lows, closes, ADX_PERIOD)
                
                # 判断趋势和带宽状态
                trend = detect_bb_trend(middle, BB_SLOPE_PERIOD)
                bandwidth_status = detect_bandwidth_change(bandwidth, BB_SLOPE_PERIOD)
                
                # 获取当前价格和指标值
                price = float(closes.iloc[-1])
                curr_upper = float(upper.iloc[-1])
                curr_middle = float(middle.iloc[-1])
                curr_lower = float(lower.iloc[-1])
                curr_bandwidth = float(bandwidth.iloc[-1])
                curr_atr = float(atr.iloc[-1])
                prev_close = float(closes.iloc[-2]) if len(closes) >= 2 else price
                adx_last = float(adx.iloc[-1])
                
                # 更新状态
                prev_state = symbol_state[symbol]
                upper_run = (prev_state.get('upper_run', 0) + 1) if price >= curr_upper * (1 - PRICE_TOLERANCE) else 0
                symbol_state[symbol] = {
                    'trend': trend,
                    'bandwidth_status': bandwidth_status,
                    'upper': curr_upper,
                    'middle': curr_middle,
                    'lower': curr_lower,
                    'bandwidth': curr_bandwidth,
                    'atr': curr_atr,
                    'upper_run': upper_run,
                    'lower_run': (prev_state.get('lower_run', 0) + 1) if price <= curr_lower * (1 + PRICE_TOLERANCE) else 0,
                    'adx': adx_last
                }
                
                # 状态变化通知
                if prev_state.get('trend') != trend or prev_state.get('bandwidth_status') != bandwidth_status:
                    log.info(f'{symbol} 状态变化: 趋势={trend}, 带宽={bandwidth_status}, 价格={price:.6f}, ATR={curr_atr:.6f}')
                
                # 获取当前持仓
                both = get_positions_both(symbol)
                long_size = float(both['long']['size'])
                long_entry = float(both['long']['entry'])
                short_size = float(both['short']['size'])
                short_entry = float(both['short']['entry'])
                
                # 峰值追踪：若无持仓则重置峰值/谷值
                if TRAIL_ENABLE:
                    if long_size <= 0:
                        symbol_state[symbol]['peak_long'] = None
                    if short_size <= 0:
                        symbol_state[symbol]['peak_short'] = None
                
                # 确保只在K线收盘后操作一次
                cur_bar_ts = get_last_closed_bar_ts(ohlcv[-1])
                acted_key = last_bar_ts.get(symbol)
                
                # ========== 动态止盈止损监控（多头） ==========
                if long_size > 0 and long_entry > 0 and price > 0:
                    pnl_pct = (price - long_entry) / long_entry
                    try:
                        ct_val = float(load_market_info(symbol).get('ctVal') or 0)
                    except Exception:
                        ct_val = 0.0
                    unreal = long_size * ct_val * (price - long_entry)
                    
                    # 动态止损价 = 开仓价 - ATR倍数
                    dynamic_sl_price = long_entry - (SL_ATR_MULTIPLIER * curr_atr)
                    
                    # 峰值追踪：更新峰值并计算从峰值的回撤
                    if TRAIL_ENABLE:
                        prev_peak = symbol_state.get(symbol, {}).get('peak_long')
                        new_peak = max(prev_peak or price, price)
                        symbol_state[symbol]['peak_long'] = new_peak
                        drawdown_from_peak = (new_peak - price) / new_peak if new_peak > 0 else 0.0
                    else:
                        drawdown_from_peak = 0.0
                    
                    log.debug(f'持仓 {symbol} 多头: 数量={long_size} 开仓={long_entry:.6f} 现价={price:.6f} 浮动={unreal:.2f} ({pnl_pct*100:.2f}%) 动态止损={dynamic_sl_price:.6f} 峰值={symbol_state.get(symbol, {}).get("peak_long")} 回撤={drawdown_from_peak*100:.2f}%')
                    
                    # 1. 峰值追踪止盈（优先级最高，先于 ATR 动态止损）
                    if TRAIL_ENABLE and pnl_pct > 0 and drawdown_from_peak >= TRAIL_DD_PCT if TRAIL_REQUIRE_PROFIT else TRAIL_ENABLE and drawdown_from_peak >= TRAIL_DD_PCT:
                        close_price = price
                        realized = long_size * ct_val * (close_price - long_entry)
                        ok = close_position_market(symbol, 'long', long_size)
                        if ok:
                            stats['trades'] += 1
                            if realized > 0:
                                stats['wins'] += 1
                            else:
                                stats['losses'] += 1
                            stats['realized_pnl'] += realized
                            log.info(f'{symbol} 多头峰值回撤止盈: 回撤={drawdown_from_peak*100:.2f}% 已实现={realized:.2f} | 累计={stats["realized_pnl"]:.2f}')
                            notify_event('多头峰值回撤止盈', f'{symbol} 回撤={drawdown_from_peak*100:.2f}% 已实现={realized:.2f} 累计={stats["realized_pnl"]:.2f}')
                            last_bar_ts[symbol] = cur_bar_ts
                            continue
                    
                    # 2. 动态止损检查
                    if price <= dynamic_sl_price:
                        close_price = price
                        realized = long_size * ct_val * (close_price - long_entry)
                        ok = close_position_market(symbol, 'long', long_size)
                        if ok:
                            stats['trades'] += 1
                            if realized > 0:
                                stats['wins'] += 1
                            else:
                                stats['losses'] += 1
                            stats['realized_pnl'] += realized
                            log.info(f'{symbol} 多头动态止损: ATR={curr_atr:.6f} 已实现={realized:.2f} | 累计={stats["realized_pnl"]:.2f}')
                            notify_event('多头动态止损', f'{symbol} ATR止损 已实现={realized:.2f} 累计={stats["realized_pnl"]:.2f}')
                            last_bar_ts[symbol] = cur_bar_ts
                            continue
                    
                    # 2. 固定止盈检查（兜底）
                    if pnl_pct >= TP_PCT:
                        close_price = price
                        realized = long_size * ct_val * (close_price - long_entry)
                        ok = close_position_market(symbol, 'long', long_size)
                        if ok:
                            stats['trades'] += 1
                            if realized > 0:
                                stats['wins'] += 1
                            else:
                                stats['losses'] += 1
                            stats['realized_pnl'] += realized
                            log.info(f'{symbol} 多头固定止盈: {pnl_pct*100:.1f}% 已实现={realized:.2f} | 累计={stats["realized_pnl"]:.2f}')
                            notify_event('多头固定止盈', f'{symbol} 已实现={realized:.2f} 累计={stats["realized_pnl"]:.2f}')
                            last_bar_ts[symbol] = cur_bar_ts
                            continue
                    
                    # 3. 动态止盈：布林带追踪
                    upper_run = symbol_state.get(symbol, {}).get('upper_run', 0)
                    if upper_run >= 3:
                        # 价格连续在上轨运行>=3根K，跌破中轨止盈
                        if price <= curr_middle * (1 - PRICE_TOLERANCE):
                            close_price = price
                            realized = long_size * ct_val * (close_price - long_entry)
                            ok = close_position_market(symbol, 'long', long_size)
                            if ok:
                                stats['trades'] += 1
                                if realized > 0:
                                    stats['wins'] += 1
                                else:
                                    stats['losses'] += 1
                                stats['realized_pnl'] += realized
                                log.info(f'{symbol} 中轨追踪止盈: 已实现={realized:.2f} | 累计={stats["realized_pnl"]:.2f}')
                                notify_event('中轨追踪止盈', f'{symbol} 已实现={realized:.2f} 累计={stats["realized_pnl"]:.2f}')
                                last_bar_ts[symbol] = cur_bar_ts
                                continue
                    else:
                        # 常规情况：触及上轨平仓
                        if price >= curr_upper * (1 - PRICE_TOLERANCE):
                            close_price = price
                            realized = long_size * ct_val * (close_price - long_entry)
                            ok = close_position_market(symbol, 'long', long_size)
                            if ok:
                                stats['trades'] += 1
                                if realized > 0:
                                    stats['wins'] += 1
                                else:
                                    stats['losses'] += 1
                                stats['realized_pnl'] += realized
                                log.info(f'{symbol} 触及上轨平仓: 已实现={realized:.2f} | 累计={stats["realized_pnl"]:.2f}')
                                notify_event('触及上轨平仓', f'{symbol} 已实现={realized:.2f} 累计={stats["realized_pnl"]:.2f}')
                                last_bar_ts[symbol] = cur_bar_ts
                                continue
                
                # ========== 动态止盈止损监控（空头） ==========
                if short_size > 0 and short_entry > 0 and price > 0:
                    pnl_pct_s = (short_entry - price) / short_entry
                    try:
                        ct_val = float(load_market_info(symbol).get('ctVal') or 0)
                    except Exception:
                        ct_val = 0.0
                    unreal = short_size * ct_val * (short_entry - price)
                    
                    # 动态止损价 = 开仓价 + ATR倍数
                    dynamic_sl_price = short_entry + (SL_ATR_MULTIPLIER * curr_atr)
                    
                    # 峰值追踪（空头用“谷值”）：更新空头的最佳价（最低价）并计算从谷值的回撤（反向上涨幅）
                    if TRAIL_ENABLE:
                        prev_peak_s = symbol_state.get(symbol, {}).get('peak_short')
                        new_peak_s = min(prev_peak_s or price, price)
                        symbol_state[symbol]['peak_short'] = new_peak_s
                        drawup_from_valley = (price - new_peak_s) / new_peak_s if new_peak_s > 0 else 0.0
                    else:
                        drawup_from_valley = 0.0
                    
                    log.debug(f'持仓 {symbol} 空头: 数量={short_size} 开仓={short_entry:.6f} 现价={price:.6f} 浮动={unreal:.2f} ({pnl_pct_s*100:.2f}%) 动态止损={dynamic_sl_price:.6f} 谷值={symbol_state.get(symbol, {}).get("peak_short")} 回升={drawup_from_valley*100:.2f}%')
                    
                    # 1. 峰值追踪止盈（优先级最高，先于 ATR 动态止损；空头为谷值反弹）
                    if TRAIL_ENABLE and pnl_pct_s > 0 and drawup_from_valley >= TRAIL_DD_PCT if TRAIL_REQUIRE_PROFIT else TRAIL_ENABLE and drawup_from_valley >= TRAIL_DD_PCT:
                        close_price = price
                        realized = short_size * ct_val * (short_entry - close_price)
                        ok = close_position_market(symbol, 'short', short_size)
                        if ok:
                            stats['trades'] += 1
                            if realized > 0:
                                stats['wins'] += 1
                            else:
                                stats['losses'] += 1
                            stats['realized_pnl'] += realized
                            log.info(f'{symbol} 空头谷值回升止盈: 回升={drawup_from_valley*100:.2f}% 已实现={realized:.2f} | 累计={stats["realized_pnl"]:.2f}')
                            notify_event('空头谷值回升止盈', f'{symbol} 回升={drawup_from_valley*100:.2f}% 已实现={realized:.2f} 累计={stats["realized_pnl"]:.2f}')
                            last_bar_ts[symbol] = cur_bar_ts
                            continue
                    
                    # 2. 动态止损
                    if price >= dynamic_sl_price:
                        close_price = price
                        realized = short_size * ct_val * (short_entry - close_price)
                        ok = close_position_market(symbol, 'short', short_size)
                        if ok:
                            stats['trades'] += 1
                            if realized > 0:
                                stats['wins'] += 1
                            else:
                                stats['losses'] += 1
                            stats['realized_pnl'] += realized
                            log.info(f'{symbol} 空头动态止损: ATR={curr_atr:.6f} 已实现={realized:.2f} | 累计={stats["realized_pnl"]:.2f}')
                            notify_event('空头动态止损', f'{symbol} ATR止损 已实现={realized:.2f} 累计={stats["realized_pnl"]:.2f}')
                            last_bar_ts[symbol] = cur_bar_ts
                            continue
                    
                    # 2. 固定止盈
                    if pnl_pct_s >= TP_PCT:
                        close_price = price
                        realized = short_size * ct_val * (short_entry - close_price)
                        ok = close_position_market(symbol, 'short', short_size)
                        if ok:
                            stats['trades'] += 1
                            if realized > 0:
                                stats['wins'] += 1
                            else:
                                stats['losses'] += 1
                            stats['realized_pnl'] += realized
                            log.info(f'{symbol} 空头固定止盈: {pnl_pct_s*100:.1f}% 已实现={realized:.2f} | 累计={stats["realized_pnl"]:.2f}')
                            notify_event('空头固定止盈', f'{symbol} 已实现={realized:.2f} 累计={stats["realized_pnl"]:.2f}')
                            last_bar_ts[symbol] = cur_bar_ts
                            continue
                    
                    # 3. 动态止盈：布林带追踪
                    lower_run = symbol_state.get(symbol, {}).get('lower_run', 0)
                    if lower_run >= 3:
                        if price >= curr_middle * (1 + PRICE_TOLERANCE):
                            close_price = price
                            realized = short_size * ct_val * (short_entry - close_price)
                            ok = close_position_market(symbol, 'short', short_size)
                            if ok:
                                stats['trades'] += 1
                                if realized > 0:
                                    stats['wins'] += 1
                                else:
                                    stats['losses'] += 1
                                stats['realized_pnl'] += realized
                                log.info(f'{symbol} 中轨追踪止盈(空): 已实现={realized:.2f} | 累计={stats["realized_pnl"]:.2f}')
                                notify_event('中轨追踪止盈(空)', f'{symbol} 已实现={realized:.2f} 累计={stats["realized_pnl"]:.2f}')
                                last_bar_ts[symbol] = cur_bar_ts
                                continue
                    else:
                        if price <= curr_lower * (1 + PRICE_TOLERANCE):
                            close_price = price
                            realized = short_size * ct_val * (short_entry - close_price)
                            ok = close_position_market(symbol, 'short', short_size)
                            if ok:
                                stats['trades'] += 1
                                if realized > 0:
                                    stats['wins'] += 1
                                else:
                                    stats['losses'] += 1
                                stats['realized_pnl'] += realized
                                log.info(f'{symbol} 触及下轨平空: 已实现={realized:.2f} | 累计={stats["realized_pnl"]:.2f}')
                                notify_event('触及下轨平空', f'{symbol} 已实现={realized:.2f} 累计={stats["realized_pnl"]:.2f}')
                                last_bar_ts[symbol] = cur_bar_ts
                                continue
                
                # ========== 收口观望 ==========
                if bandwidth_status == 'squeezing':
                    log.debug(f'{symbol} 收口阶段，观望等方向')
                    continue
                
                # ========== 开口向下 + 持多仓 -> 紧急平仓 ==========
                if bandwidth_status == 'expanding' and trend == 'down' and long_size > 0:
                    try:
                        close_price = float(exchange.fetch_ticker(symbol)['last'] or 0)
                        ct_val = float(load_market_info(symbol).get('ctVal') or 0)
                    except Exception:
                        close_price, ct_val = 0.0, 0.0
                    realized = long_size * ct_val * (close_price - long_entry)
                    ok = close_position_market(symbol, 'long', long_size)
                    if ok:
                        stats['trades'] += 1
                        if realized > 0:
                            stats['wins'] += 1
                        else:
                            stats['losses'] += 1
                        stats['realized_pnl'] += realized
                        log.info(f'{symbol} 开口向下紧急平多: 已实现={realized:.2f} | 累计={stats["realized_pnl"]:.2f}')
                        notify_event('开口向下紧急平多', f'{symbol} 已实现={realized:.2f} 累计={stats["realized_pnl"]:.2f}')
                        last_bar_ts[symbol] = cur_bar_ts
                        continue
                
                # 避免同一K线重复操作
                if acted_key == cur_bar_ts:
                    log.debug(f'{symbol} 该K线已处理过，跳过')
                    continue
                
                # ADX 震荡过滤
                if trend == 'flat' and adx_last < ADX_MIN_TREND:
                    log.debug(f"{symbol} ADX过低({adx_last:.1f})，震荡期需额外确认")
                
                # ========== 中线向上策略 ==========
                if trend == 'up':
                    # 回踩中轨开多
                    if price <= curr_middle * (1 + PRICE_TOLERANCE) and not (long_size > 0):
                        ok = place_market_order(symbol, 'buy', BUDGET_USDT)
                        if ok:
                            log.info(f'{symbol} 上升趋势回踩中轨开多')
                            last_bar_ts[symbol] = cur_bar_ts
                    
                    # 开口向上 + 已有多头 -> 加仓
                    if bandwidth_status == 'expanding' and long_size > 0:
                        ok = place_market_order(symbol, 'buy', BUDGET_USDT, position_ratio=0.3)
                        if ok:
                            log.info(f'{symbol} 开口向上加仓多头30%')
                            notify_event('开口向上加仓', f'{symbol} 追加30%')
                            last_bar_ts[symbol] = cur_bar_ts
                    
                    # 开口向上 + 持空仓 -> 紧急平空
                    if bandwidth_status == 'expanding' and short_size > 0:
                        close_price = float(exchange.fetch_ticker(symbol)['last'] or 0)
                        ct_val = float(load_market_info(symbol).get('ctVal') or 0)
                        realized = short_size * ct_val * (short_entry - close_price)
                        ok = close_position_market(symbol, 'short', short_size)
                        if ok:
                            stats['trades'] += 1
                            if realized > 0:
                                stats['wins'] += 1
                            else:
                                stats['losses'] += 1
                            stats['realized_pnl'] += realized
                            log.info(f'{symbol} 开口向上紧急平空: 已实现={realized:.2f} | 累计={stats["realized_pnl"]:.2f}')
                            notify_event('开口向上紧急平空', f'{symbol} 已实现={realized:.2f} 累计={stats["realized_pnl"]:.2f}')
                            last_bar_ts[symbol] = cur_bar_ts
                
                # ========== 中线向下策略 ==========
                elif trend == 'down':
                    # 有多头，中轨反弹即平
                    if long_size > 0 and price >= curr_middle * (1 - PRICE_TOLERANCE):
                        try:
                            close_price = float(exchange.fetch_ticker(symbol)['last'] or 0)
                            ct_val = float(load_market_info(symbol).get('ctVal') or 0)
                        except Exception:
                            close_price, ct_val = 0.0, 0.0
                        realized = long_size * ct_val * (close_price - long_entry)
                        ok = close_position_market(symbol, 'long', long_size)
                        if ok:
                            stats['trades'] += 1
                            if realized > 0:
                                stats['wins'] += 1
                            else:
                                stats['losses'] += 1
                            stats['realized_pnl'] += realized
                            log.info(f'{symbol} 下降趋势反弹中轨平多: 已实现={realized:.2f} | 累计={stats["realized_pnl"]:.2f}')
                            notify_event('下降趋势平多', f'{symbol} 已实现={realized:.2f} 累计={stats["realized_pnl"]:.2f}')
                            last_bar_ts[symbol] = cur_bar_ts
                    
                    # 抢反弹（高风险，默认禁用）
                    if ENABLE_DOWNTREND_BOUNCE and price <= curr_lower * (1 + PRICE_TOLERANCE) and not (long_size > 0):
                        ok = place_market_order(symbol, 'buy', BUDGET_USDT, position_ratio=DOWNTREND_POSITION_RATIO)
                        if ok:
                            log.info(f'{symbol} 下降趋势下轨抢反弹（{DOWNTREND_POSITION_RATIO*100:.0f}%）')
                            notify_event('抢反弹开仓', f'{symbol} 下轨 {DOWNTREND_POSITION_RATIO*100:.0f}%仓')
                            last_bar_ts[symbol] = cur_bar_ts
                
                # ========== 震荡市策略（多重确认） ==========
                elif trend == 'flat':
                    is_flat_env = (adx_last < ADX_MIN_TREND and bandwidth_status in ('stable', 'squeezing'))
                    
                    if is_flat_env:
                        # === 下轨开多：3重确认 ===
                        lower_touch_prev = (prev_close <= curr_lower * (1 + PRICE_TOLERANCE))
                        lower_reject_now = (price > curr_lower * (1 + PRICE_TOLERANCE))
                        
                        # K线形态确认：锤子线
                        has_hammer = check_hammer_pattern(ohlcv, -1) or check_hammer_pattern(ohlcv, -2)
                        
                        # 盈亏比确认
                        entry_est = price
                        target_est = curr_upper
                        stop_est = curr_lower - (SL_ATR_MULTIPLIER * curr_atr)
                        risk_reward = calculate_risk_reward(entry_est, target_est, stop_est)
                        
                        if lower_touch_prev and lower_reject_now and has_hammer and risk_reward >= MIN_RISK_REWARD and not (long_size > 0):
                            ok = place_market_order(symbol, 'buy', BUDGET_USDT, position_ratio=0.5)
                            if ok:
                                log.info(f'{symbol} 震荡市下轨多重确认开多(50%) RR={risk_reward:.2f}:1')
                                notify_event('震荡市确认开多', f'{symbol} 盈亏比={risk_reward:.2f}:1')
                                last_bar_ts[symbol] = cur_bar_ts
                        elif lower_touch_prev and lower_reject_now and not (long_size > 0):
                            log.debug(f'{symbol} 下轨信号但未通过确认: hammer={has_hammer} RR={risk_reward:.2f}')
                        
                        # === 上轨平多 ===
                        if long_size > 0 and price >= curr_upper * (1 - PRICE_TOLERANCE):
                            try:
                                close_price = float(exchange.fetch_ticker(symbol)['last'] or 0)
                                ct_val = float(load_market_info(symbol).get('ctVal') or 0)
                            except Exception:
                                close_price, ct_val = 0.0, 0.0
                            realized = long_size * ct_val * (close_price - long_entry)
                            ok = close_position_market(symbol, 'long', long_size)
                            if ok:
                                stats['trades'] += 1
                                if realized > 0:
                                    stats['wins'] += 1
                                else:
                                    stats['losses'] += 1
                                stats['realized_pnl'] += realized
                                log.info(f'{symbol} 震荡市上轨平多: 已实现={realized:.2f} | 累计={stats["realized_pnl"]:.2f}')
                                notify_event('震荡市平多', f'{symbol} 已实现={realized:.2f} 累计={stats["realized_pnl"]:.2f}')
                                last_bar_ts[symbol] = cur_bar_ts
                        
                        # === 上轨开空：3重确认 ===
                        upper_touch_prev = (prev_close >= curr_upper * (1 - PRICE_TOLERANCE))
                        upper_reject_now = (price < curr_upper * (1 - PRICE_TOLERANCE))
                        
                        # K线形态：流星线
                        has_star = check_shooting_star_pattern(ohlcv, -1) or check_shooting_star_pattern(ohlcv, -2)
                        
                        # 盈亏比确认（空头）
                        entry_est_s = price
                        target_est_s = curr_lower
                        stop_est_s = curr_upper + (SL_ATR_MULTIPLIER * curr_atr)
                        risk_reward_s = calculate_risk_reward(entry_est_s, target_est_s, stop_est_s)
                        
                        if upper_touch_prev and upper_reject_now and has_star and risk_reward_s >= MIN_RISK_REWARD and not (short_size > 0):
                            ok = place_market_order(symbol, 'sell', BUDGET_USDT, position_ratio=0.5)
                            if ok:
                                log.info(f'{symbol} 震荡市上轨多重确认开空(50%) RR={risk_reward_s:.2f}:1')
                                notify_event('震荡市确认开空', f'{symbol} 盈亏比={risk_reward_s:.2f}:1')
                                last_bar_ts[symbol] = cur_bar_ts
                        elif upper_touch_prev and upper_reject_now and not (short_size > 0):
                            log.debug(f'{symbol} 上轨信号但未通过确认: star={has_star} RR={risk_reward_s:.2f}')
                        
                        # === 下轨平空 ===
                        if short_size > 0 and price <= curr_lower * (1 + PRICE_TOLERANCE):
                            try:
                                close_price = float(exchange.fetch_ticker(symbol)['last'] or 0)
                                ct_val = float(load_market_info(symbol).get('ctVal') or 0)
                            except Exception:
                                close_price, ct_val = 0.0, 0.0
                            realized = short_size * ct_val * (short_entry - close_price)
                            ok = close_position_market(symbol, 'short', short_size)
                            if ok:
                                stats['trades'] += 1
                                if realized > 0:
                                    stats['wins'] += 1
                                else:
                                    stats['losses'] += 1
                                stats['realized_pnl'] += realized
                                log.info(f'{symbol} 震荡市下轨平空: 已实现={realized:.2f} | 累计={stats["realized_pnl"]:.2f}')
                                notify_event('震荡市平空', f'{symbol} 已实现={realized:.2f} 累计={stats["realized_pnl"]:.2f}')
                                last_bar_ts[symbol] = cur_bar_ts
                
            except Exception as e:
                log.warning(f'{symbol} 处理异常: {e}')
                continue
        
        time.sleep(SCAN_INTERVAL)
    except Exception as e:
        log.warning(f'主循环异常: {e}')
        time.sleep(SCAN_INTERVAL)