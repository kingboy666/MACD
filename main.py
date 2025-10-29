#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bollinger Bands Strategy (OKX USDT-SWAP, 30m)
- Indicator: Bollinger Bands(20,2) only
- Signal: 
  * Uptrend: Buy at middle band, sell at upper band
  * Downtrend: Avoid or minimal position
  * Sideways: Buy at lower band, sell at upper band
  * Expanding bands + uptrend: Add position
  * Expanding bands + downtrend: Close all
  * Squeezing bands: Wait for direction
- Leverage: per-symbol configurable
- TP/SL: continuous monitoring

Required env:
  OKX_API_KEY, OKX_SECRET_KEY, OKX_PASSPHRASE
Optional env:
  BUDGET_USDT (default 5)
  DEFAULT_LEVERAGE (default 20)
  TP_PCT (default 0.015 = 1.5%)
  SL_PCT (default 0.008 = 0.8%)
  SCAN_INTERVAL (seconds, default 10)
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
log = logging.getLogger('bollinger-30m')

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
TP_PCT = float(os.environ.get('TP_PCT', '0.012').strip() or 0.012)
SL_PCT = float(os.environ.get('SL_PCT', '0.006').strip() or 0.006)
SCAN_INTERVAL = int(float(os.environ.get('SCAN_INTERVAL', '10').strip() or 10))
USE_BALANCE_AS_MARGIN = os.environ.get('USE_BALANCE_AS_MARGIN', 'true').strip().lower() in ('1', 'true', 'yes')
MARGIN_UTILIZATION = float(os.environ.get('MARGIN_UTILIZATION', '0.95').strip() or 0.95)

# 布林带参数
BB_PERIOD = int(os.environ.get('BB_PERIOD', '18').strip() or 18)
BB_STD = float(os.environ.get('BB_STD', '2.0').strip() or 2.0)
BB_SLOPE_PERIOD = int(os.environ.get('BB_SLOPE_PERIOD', '5').strip() or 5)
# ADX 过滤参数
ADX_PERIOD = int(os.environ.get('ADX_PERIOD', '14').strip() or 14)
ADX_MIN_TREND = float(os.environ.get('ADX_MIN_TREND', '20').strip() or 20)

# 趋势判断阈值
SLOPE_UP_THRESH = float(os.environ.get('SLOPE_UP_THRESH', '0.0015').strip() or 0.0015)
SLOPE_DOWN_THRESH = float(os.environ.get('SLOPE_DOWN_THRESH', '-0.0015').strip() or -0.0015)
SLOPE_FLAT_RANGE = float(os.environ.get('SLOPE_FLAT_RANGE', '0.0008').strip() or 0.0008)

# 带宽变化阈值
BANDWIDTH_EXPAND_THRESH = float(os.environ.get('BANDWIDTH_EXPAND_THRESH', '0.12').strip() or 0.12)
BANDWIDTH_SQUEEZE_THRESH = float(os.environ.get('BANDWIDTH_SQUEEZE_THRESH', '-0.12').strip() or -0.12)

# 价格位置容差（判断是否在轨道上）
PRICE_TOLERANCE = float(os.environ.get('PRICE_TOLERANCE', '0.002').strip() or 0.002)

# 下降趋势抢反弹开关（默认关闭，太危险）
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
        log.warning(f'notify_event failed: {e}')
    return {'size': 0.0, 'side': None, 'entry': 0.0}

def place_market_order(symbol: str, side: str, budget_usdt: float, position_ratio: float = 1.0) -> bool:
    """
    position_ratio: 仓位比例，默认1.0=全仓，0.3=30%仓位（用于下降趋势抢反弹）
    """
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
    
    if DRY_RUN:
        log.info(f'[DRY_RUN] 模拟开仓 {symbol} {side} 数量={contracts}')
        return True
    try:
        exchange.privatePostTradeOrder(params)
        log.info(f'下单成功 {symbol}: 方向={side} 数量={contracts}, 预算={equity_usdt:.2f}USDT (仓位比例={position_ratio*100:.0f}%)')
        notify_event('开仓成功', f'{symbol} {side} 数量={contracts} 预算={equity_usdt:.2f}U 比例={position_ratio*100:.0f}%')
        return True
    except Exception as e:
        log.warning(f'下单失败 {symbol}: {e}')
        return False

def close_position_market(symbol: str, side_to_close: str, qty: float) -> bool:
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

# ========== 布林带计算 ==========

def calculate_bollinger_bands(closes: pd.Series, period: int = 20, std_multiplier: float = 2.0):
    """计算布林带"""
    middle = closes.rolling(window=period).mean()
    std = closes.rolling(window=period).std()
    upper = middle + std_multiplier * std
    lower = middle - std_multiplier * std
    bandwidth = (upper - lower) / middle
    return upper, middle, lower, bandwidth

def calc_adx(highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 14) -> pd.Series:
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
    # 避免除以0
    atr_safe = atr.replace(0, pd.NA)
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr_safe)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr_safe)
    denom = (plus_di + minus_di).replace(0, pd.NA)
    dx = (abs(plus_di - minus_di) / denom) * 100
    adx = dx.rolling(period).mean().fillna(0)
    return adx

def detect_bb_trend(middle: pd.Series, lookback: int = 5):
    """判断三线方向：up/down/flat"""
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

# ========== 主策略逻辑 ==========

last_bar_ts: Dict[str, int] = {}
symbol_state: Dict[str, Dict[str, Any]] = {}  # 记录每个交易对的状态

log.info('=' * 70)
log.info(f'Start Bollinger Bands Strategy - {TIMEFRAME} timeframe')
log.info(f'布林带参数: 周期={BB_PERIOD}, 标准差={BB_STD}倍')
log.info(f'趋势阈值: 上升>{SLOPE_UP_THRESH*100:.2f}%, 下降<{SLOPE_DOWN_THRESH*100:.2f}%')
log.info(f'带宽阈值: 开口>{BANDWIDTH_EXPAND_THRESH*100:.0f}%, 收口<{BANDWIDTH_SQUEEZE_THRESH*100:.0f}%')
log.info(f'下降趋势抢反弹: {"启用" if ENABLE_DOWNTREND_BOUNCE else "禁用"}')
log.info('=' * 70)

if not DRY_RUN:
    ensure_position_mode()
    for sym in SYMBOLS:
        ensure_leverage(sym)
else:
    log.warning('DRY_RUN 开启：跳过设置持仓模式与杠杆')
for sym in SYMBOLS:
    symbol_state[sym] = {'trend': 'unknown', 'bandwidth_status': 'unknown'}

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
        if cycle_count % 10 == 0:
            log.info(f"累计交易={stats['trades']} 胜率={stats['wins']/max(1,stats['trades'])*100:.1f}% 实现盈亏={stats['realized_pnl']:.2f}")
        
        for symbol in SYMBOLS:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=60)
                if not ohlcv or len(ohlcv) < BB_PERIOD + 10:
                    log.debug(f'{symbol} insufficient OHLCV: {0 if not ohlcv else len(ohlcv)}')
                    continue
                
                closes = pd.Series([c[4] for c in ohlcv])
                highs = pd.Series([c[2] for c in ohlcv])
                lows = pd.Series([c[3] for c in ohlcv])
                
                # 计算布林带
                upper, middle, lower, bandwidth = calculate_bollinger_bands(closes, BB_PERIOD, BB_STD)
                
                # 判断趋势和带宽状态
                trend = detect_bb_trend(middle, BB_SLOPE_PERIOD)
                bandwidth_status = detect_bandwidth_change(bandwidth, BB_SLOPE_PERIOD)
                adx = calc_adx(highs, lows, closes, ADX_PERIOD)
                
                # 获取当前价格和布林带值
                price = float(closes.iloc[-1])
                curr_upper = float(upper.iloc[-1])
                curr_middle = float(middle.iloc[-1])
                curr_lower = float(lower.iloc[-1])
                curr_bandwidth = float(bandwidth.iloc[-1])
                
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
                    'upper_run': upper_run,
                    'lower_run': (prev_state.get('lower_run', 0) + 1) if price <= curr_lower * (1 + PRICE_TOLERANCE) else 0,
                    'adx': float(adx.iloc[-1])
                }
                
                # 状态变化通知
                if prev_state['trend'] != trend or prev_state['bandwidth_status'] != bandwidth_status:
                    log.info(f'{symbol} 状态变化: 趋势={trend}, 带宽={bandwidth_status}, 价格={price:.6f}, 上轨={curr_upper:.6f}, 中轨={curr_middle:.6f}, 下轨={curr_lower:.6f}')
                
                # 获取当前持仓
                pos = get_position(symbol)
                size = float(pos['size'] or 0.0)
                side = pos['side']
                entry = float(pos['entry'] or 0.0)
                
                # 确保只在K线收盘后操作一次
                cur_bar_ts = get_last_closed_bar_ts(ohlcv[-1])
                acted_key = last_bar_ts.get(symbol)
                
                # ========== 止盈止损监控（多头） ==========
                if long_size > 0 and long_entry > 0 and price > 0:
                    size = long_size; entry = long_entry; side = 'long'
                    pnl_pct = (price - entry) / entry
                    try:
                        ct_val = float(load_market_info(symbol).get('ctVal') or 0)
                    except Exception:
                        ct_val = 0.0
                    unreal = size * ct_val * (price - entry)
                    
                    log.debug(f'持仓 {symbol} 多头 数量={size} 开仓价={entry:.6f} 现价={price:.6f} 浮动盈亏={unreal:.2f} ({pnl_pct*100:.2f}%)')
                    
                    # 止损
                    if pnl_pct <= -SL_PCT:
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
                            log.info(f'{symbol} 震荡市上轨平多: 已实现={realized:.2f} | 累计={stats["realized_pnl"]:.2f}')
                            notify_event('震荡市平多', f'{symbol} 已实现={realized:.2f} 累计={stats["realized_pnl"]:.2f}')
                            last_bar_ts[symbol] = cur_bar_ts
                            continue
                            stats['trades'] += 1
                            if realized > 0:
                                stats['wins'] += 1
                            else:
                                stats['losses'] += 1
                            stats['realized_pnl'] += realized
                            log.info(f'{symbol} 震荡市上轨平多: 已实现={realized:.2f} | 累计={stats["realized_pnl"]:.2f}')
                            notify_event('震荡市平多', f'{symbol} 已实现={realized:.2f} 累计={stats["realized_pnl"]:.2f}')
                            last_bar_ts[symbol] = cur_bar_ts
                            continue'] += 1
                            stats['realized_pnl'] += realized
                            log.info(f'触发止盈已平仓 {symbol}: 已实现盈亏={realized:.2f} | 累计={stats["realized_pnl"]:.2f}')
                            notify_event('触发止盈', f'{symbol} 已实现={realized:.2f} 累计={stats["realized_pnl"]:.2f}')
                            last_bar_ts[symbol] = cur_bar_ts
                            continue
                    
                    # 动态止盈：
                    # - 若价格连续在上轨附近运行>=3根K，则采用中轨追踪止盈：跌破中轨即平仓；
                    # - 否则，触及上轨直接平仓。
                    upper_run = symbol_state.get(symbol, {}).get('upper_run', 0)
                    if upper_run >= 3:
                        if price <= curr_middle * (1 - PRICE_TOLERANCE):
                            close_price = price
                            realized = size * ct_val * (close_price - entry)
                            ok = close_position_market(symbol, 'long', size)
                            if ok:
                                stats['trades'] += 1
                                if realized > 0:
                                    stats['wins'] += 1
                                else:
                                    stats['losses'] += 1
                                stats['realized_pnl'] += realized
                                log.info(f'中轨追踪止盈 {symbol}: 已实现盈亏={realized:.2f} | 累计={stats["realized_pnl"]:.2f}')
                                notify_event('中轨追踪止盈', f'{symbol} 已实现={realized:.2f} 累计={stats["realized_pnl"]:.2f}')
                                last_bar_ts[symbol] = cur_bar_ts
                                continue
                    else:
                        if price >= curr_upper * (1 - PRICE_TOLERANCE):
                            close_price = price
                            realized = size * ct_val * (close_price - entry)
                            ok = close_position_market(symbol, 'long', size)
                            if ok:
                                stats['trades'] += 1
                                if realized > 0:
                                    stats['wins'] += 1
                                else:
                                    stats['losses'] += 1
                                stats['realized_pnl'] += realized
                                log.info(f'触及上轨平仓 {symbol}: 已实现盈亏={realized:.2f} | 累计={stats["realized_pnl"]:.2f}')
                                notify_event('触及上轨平仓', f'{symbol} 已实现={realized:.2f} 累计={stats["realized_pnl"]:.2f}')
                                last_bar_ts[symbol] = cur_bar_ts
                                continue
                
                # ========== 止盈止损监控（空头） ==========
                if short_size > 0 and short_entry > 0 and price > 0:
                    pnl_pct_s = (short_entry - price) / short_entry
                    try:
                        ct_val = float(load_market_info(symbol).get('ctVal') or 0)
                    except Exception:
                        ct_val = 0.0
                    unreal = short_size * ct_val * (short_entry - price)
                    log.debug(f'持仓 {symbol} 空头 数量={short_size} 开仓价={short_entry:.6f} 现价={price:.6f} 浮动盈亏={unreal:.2f} ({pnl_pct_s*100:.2f}%)')
                    # 止损（对空头：价格上行）
                    if pnl_pct_s <= -SL_PCT:
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
                            log.info(f'空头触发止损已平仓 {symbol}: 已实现盈亏={realized:.2f} | 累计={stats["realized_pnl"]:.2f}')
                            notify_event('空头触发止损', f'{symbol} 已实现={realized:.2f} 累计={stats["realized_pnl"]:.2f}')
                            last_bar_ts[symbol] = cur_bar_ts
                    # 止盈（对空头：价格下行）
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
                            log.info(f'空头触发止盈已平仓 {symbol}: 已实现盈亏={realized:.2f} | 累计={stats["realized_pnl"]:.2f}')
                            notify_event('空头触发止盈', f'{symbol} 已实现={realized:.2f} 累计={stats["realized_pnl"]:.2f}')
                            last_bar_ts[symbol] = cur_bar_ts
                    # 动态止盈（空头对称逻辑）：
                    lower_run = symbol_state.get(symbol, {}).get('lower_run', 0)
                    if lower_run >= 3:
                        # 上穿中轨则平空
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
                                log.info(f'中轨追踪止盈(空) {symbol}: 已实现盈亏={realized:.2f} | 累计={stats["realized_pnl"]:.2f}')
                                notify_event('中轨追踪止盈(空)', f'{symbol} 已实现={realized:.2f} 累计={stats["realized_pnl"]:.2f}')
                                last_bar_ts[symbol] = cur_bar_ts
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
                                log.info(f'触及下轨平空 {symbol}: 已实现盈亏={realized:.2f} | 累计={stats["realized_pnl"]:.2f}')
                                notify_event('触及下轨平空', f'{symbol} 已实现={realized:.2f} 累计={stats["realized_pnl"]:.2f}')
                                last_bar_ts[symbol] = cur_bar_ts
                
                # ========== 收口观望 ==========
                if bandwidth_status == 'squeezing':
                    log.debug(f'{symbol} 收口阶段，观望等方向')
                    continue
                
                # ========== 开口向下 + 持仓 -> 立即平仓 ==========
                if bandwidth_status == 'expanding' and trend == 'down' and long_size > 0:
                    try:
                        close_price = float(exchange.fetch_ticker(symbol)['last'] or 0)
                        ct_val = float(load_market_info(symbol).get('ctVal') or 0)
                    except Exception:
                        close_price, ct_val = 0.0, 0.0
                    realized = size * ct_val * (close_price - entry)
                    ok = close_position_market(symbol, 'long', size)
                    if ok:
                        stats['trades'] += 1
                        if realized > 0:
                            stats['wins'] += 1
                        else:
                            stats['losses'] += 1
                        stats['realized_pnl'] += realized
                        log.info(f'开口向下紧急平仓 {symbol}: 已实现={realized:.2f} | 累计={stats["realized_pnl"]:.2f}')
                        notify_event('开口向下紧急平仓', f'{symbol} 已实现={realized:.2f} 累计={stats["realized_pnl"]:.2f}')
                        last_bar_ts[symbol] = cur_bar_ts
                        continue
                
                # 避免同一K线重复操作
                if acted_key == cur_bar_ts:
                    log.debug(f'{symbol} 该K线已处理过 {cur_bar_ts}，跳过')
                    continue
                
                # ADX 震荡过滤：仅影响开仓，不影响平仓
                if trend == 'flat' and float(symbol_state[symbol].get('adx', 0)) < ADX_MIN_TREND:
                    log.debug(f"{symbol} ADX过低({symbol_state[symbol].get('adx', 0):.1f})，过滤震荡期开仓")
                    continue
                
                # ========== 三线向上策略 ==========
                if trend == 'up':
                    # 买入条件：价格回踩到中轨附近（多）
                    if price <= curr_middle * (1 + PRICE_TOLERANCE) and not (long_size > 0):
                        ok = place_market_order(symbol, 'buy', BUDGET_USDT)
                        if ok:
                            log.info(f'{symbol} 上升趋势回踩中轨开多')
                            last_bar_ts[symbol] = cur_bar_ts
                            continue
                    
                    # 开口向上 + 已有多头 -> 加仓（谨慎）
                    if bandwidth_status == 'expanding' and long_size > 0:
                        ok = place_market_order(symbol, 'buy', BUDGET_USDT, position_ratio=0.3)
                        if ok:
                            log.info(f'{symbol} 开口向上加仓多头30%')
                            notify_event('开口向上加仓(多)', f'{symbol} 追加30%仓位')
                            last_bar_ts[symbol] = cur_bar_ts
                            continue
                    
                    # 开口向上 + 持有空头 -> 紧急平空（对称）
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
                            log.info(f'开口向上紧急平空 {symbol}: 已实现={realized:.2f} | 累计={stats["realized_pnl"]:.2f}')
                            notify_event('开口向上紧急平空', f'{symbol} 已实现={realized:.2f} 累计={stats["realized_pnl"]:.2f}')
                            last_bar_ts[symbol] = cur_bar_ts
                            continue
                
                # ========== 三线向下策略 ==========
                elif trend == 'down':
                    # 如果有多头，中轨反弹即平多
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
                            log.info(f'{symbol} 下降趋势反弹中轨平仓: 已实现={realized:.2f} | 累计={stats["realized_pnl"]:.2f}')
                            notify_event('下降趋势平仓', f'{symbol} 已实现={realized:.2f} 累计={stats["realized_pnl"]:.2f}')
                            last_bar_ts[symbol] = cur_bar_ts
                            continue
                    
                    # 抢反弹（高风险，默认禁用）
                    if ENABLE_DOWNTREND_BOUNCE and price <= curr_lower * (1 + PRICE_TOLERANCE) and not (size > 0):
                        ok = place_market_order(symbol, 'buy', BUDGET_USDT, position_ratio=DOWNTREND_POSITION_RATIO)
                        if ok:
                            log.info(f'{symbol} 下降趋势下轨抢反弹（{DOWNTREND_POSITION_RATIO*100:.0f}%仓位）')
                            notify_event('抢反弹开仓', f'{symbol} 下轨抢反弹 {DOWNTREND_POSITION_RATIO*100:.0f}%仓')
                            last_bar_ts[symbol] = cur_bar_ts
                            continue
                
                # ========== 三线走平（震荡市）策略 ==========
                elif trend == 'flat':
                    # 下轨买入（多）
                    if price <= curr_lower * (1 + PRICE_TOLERANCE) and not (long_size > 0):
                        ok = place_market_order(symbol, 'buy', BUDGET_USDT)
                        if ok:
                            log.info(f'{symbol} 震荡市下轨买入')
                            last_bar_ts[symbol] = cur_bar_ts
                            continue
                    
                    # 上轨卖出（多头平仓）
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
                            continue