#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone Simple MACD Strategy (OKX USDT-SWAP, 30m)
- Indicator: MACD(6,16,9) only
- Signal: Golden cross -> open long; Death cross -> close long; no shorting
- Symbols: use the previous 11 symbols
- Leverage: set per-symbol (configurable)
- TP/SL: local guard by percent thresholds (configurable via env), continuous monitoring

Required env:
  OKX_API_KEY, OKX_SECRET_KEY, OKX_PASSPHRASE
Optional env:
  BUDGET_USDT (default 5)
  DEFAULT_LEVERAGE (default 20)
  TP_PCT (default 0.01 = 1%)
  SL_PCT (default 0.006 = 0.6%)
  SCAN_INTERVAL (seconds, default 10)
"""
import os
import time
import math
import logging
from typing import Dict, Any

import ccxt
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('simple-macd-30m')

API_KEY = os.environ.get('OKX_API_KEY', '').strip()
API_SECRET = os.environ.get('OKX_SECRET_KEY', '').strip()
API_PASS = os.environ.get('OKX_PASSPHRASE', '').strip()
if not API_KEY or not API_SECRET or not API_PASS:
    raise SystemExit('Missing OKX credentials: set OKX_API_KEY, OKX_SECRET_KEY, OKX_PASSPHRASE')

BUDGET_USDT = float(os.environ.get('BUDGET_USDT', '5').strip() or 5)
DEFAULT_LEVERAGE = int(float(os.environ.get('DEFAULT_LEVERAGE', '20').strip() or 20))
TP_PCT = float(os.environ.get('TP_PCT', '0.01').strip() or 0.01)
SL_PCT = float(os.environ.get('SL_PCT', '0.006').strip() or 0.006)
SCAN_INTERVAL = int(float(os.environ.get('SCAN_INTERVAL', '10').strip() or 10))

TIMEFRAME = '30m'
SYMBOLS = [
    'FIL/USDT:USDT', 'ZRO/USDT:USDT', 'WIF/USDT:USDT', 'WLD/USDT:USDT',
    'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'XRP/USDT:USDT',
    'ARB/USDT:USDT'
]

# Per-symbol leverage (can override DEFAULT_LEVERAGE)
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

# Helpers for OKX

def symbol_to_inst_id(sym: str) -> str:
    base = sym.split('/')[0]
    return f'{base}-USDT-SWAP'

# Market info cache
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

# Set leverage per symbol

def ensure_leverage(symbol: str):
    lev = int(SYMBOL_LEVERAGE.get(symbol, DEFAULT_LEVERAGE) or DEFAULT_LEVERAGE)
    inst_id = symbol_to_inst_id(symbol)
    try:
        # Cross margin by default; for hedge posSide must be specified, assume one-way here
        exchange.privatePostAccountSetLeverage({'instId': inst_id, 'mgnMode': 'cross', 'lever': str(lev)})
        log.info(f'Leverage set {symbol} -> {lev}x')
    except Exception as e:
        log.warning(f'Failed set leverage {symbol}: {e}')

# Positions

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
    except Exception:
        pass
    return {'size': 0.0, 'side': None, 'entry': 0.0}

# Orders

def place_market_order(symbol: str, side: str, budget_usdt: float) -> bool:
    # 使用全仓：读取账户可用USDT余额，将全部余额用于当前有信号的交易对
    try:
        balance = exchange.fetch_balance()
        avail = float(balance.get('USDT', {}).get('free') or balance.get('USDT', {}).get('available') or 0)
    except Exception:
        avail = 0.0
    budget_usdt = max(0.0, avail)
    if budget_usdt <= 0:
        log.warning('No available USDT balance to open position')
        return False

    info = load_market_info(symbol)
    inst_id = info['instId']
    # Get last price
    ticker = exchange.fetch_ticker(symbol)
    price = float(ticker['last'] or ticker['close'] or 0)
    if price <= 0:
        raise Exception('invalid price')
    ct_val = float(info.get('ctVal') or 0)
    if ct_val <= 0:
        # Fallback: assume linear swap 0.01 coin per contract
        ct_val = 0.01
    # contracts = (notional in base coin) / ctVal = (budget/price)/ctVal
    contracts = (budget_usdt / price) / ct_val
    # align to lotSz
    lot = float(info.get('lotSz') or 0)
    minsz = float(info.get('minSz') or 0)
    if lot > 0:
        contracts = math.floor(contracts / lot) * lot
    if contracts <= 0 or (minsz > 0 and contracts < minsz):
        log.warning(f'Computed contracts too small: {contracts}, minSz={minsz}, lotSz={lot}')
        return False
    side_okx = 'buy' if side == 'buy' else 'sell'
    params = {
        'instId': inst_id,
        'tdMode': 'cross',
        'side': side_okx,
        'ordType': 'market',
        'sz': str(contracts),
    }
    try:
        exchange.privatePostTradeOrder(params)
        log.info(f'Order success {symbol}: {side} sz={contracts}, budget={budget_usdt}USDT')
        return True
    except Exception as e:
        log.warning(f'Order failed {symbol}: {e}')
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
        log.warning(f'Close qty too small: {sz}')
        return False
    params = {
        'instId': inst_id,
        'tdMode': 'cross',
        'side': side_okx,
        'ordType': 'market',
        'sz': str(sz),
        'reduceOnly': True,
    }
    try:
        exchange.privatePostTradeOrder(params)
        log.info(f'Closed {symbol} {side_to_close} qty={sz}')
        return True
    except Exception as e:
        log.warning(f'Close failed {symbol}: {e}')
        return False

# MACD

def macd_6_16_9(closes: pd.Series):
    ema_fast = closes.ewm(span=6, adjust=False).mean()
    ema_slow = closes.ewm(span=16, adjust=False).mean()
    diff = ema_fast - ema_slow
    dea = diff.ewm(span=9, adjust=False).mean()
    hist = diff - dea
    return diff, dea, hist

last_bar_ts: Dict[str, int] = {}

log.info('=' * 70)
log.info('Start Standalone Simple MACD(6,16,9) Strategy - 30m timeframe')
log.info('=' * 70)

# Prepare leverage once
for sym in SYMBOLS:
    ensure_leverage(sym)

def get_last_closed_bar_ts(ohlcv_row):
    # ccxt returns [timestamp, open, high, low, close, volume]
    return int(ohlcv_row[0])

while True:
    try:
        for symbol in SYMBOLS:
            try:
                # Fetch 30m OHLCV
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=60)
                if not ohlcv or len(ohlcv) < 35:
                    continue
                closes = pd.Series([c[4] for c in ohlcv])
                diff, dea, _ = macd_6_16_9(closes)

                # Ensure we only act on closed bar once
                cur_bar_ts = get_last_closed_bar_ts(ohlcv[-1])
                prev_bar_ts = get_last_closed_bar_ts(ohlcv[-2])
                acted_key = last_bar_ts.get(symbol)

                prev_gc = diff.iloc[-2] <= dea.iloc[-2]
                latest_gc = diff.iloc[-1] > dea.iloc[-1]
                prev_dc = diff.iloc[-2] >= dea.iloc[-2]
                latest_dc = diff.iloc[-1] < dea.iloc[-1]
                golden_cross = prev_gc and latest_gc
                death_cross = prev_dc and latest_dc

                pos = get_position(symbol)
                size = float(pos['size'] or 0.0)
                side = pos['side']
                entry = float(pos['entry'] or 0.0)

                # Local TP/SL guard
                try:
                    price = float(exchange.fetch_ticker(symbol)['last'] or 0)
                except Exception:
                    price = 0.0
                if size > 0 and side == 'long' and entry > 0 and price > 0:
                    pnl_pct = (price - entry) / entry
                    if pnl_pct <= -SL_PCT:
                        close_position_market(symbol, 'long', size)
                        # Avoid duplicate actions within same bar
                        last_bar_ts[symbol] = cur_bar_ts
                        continue
                    if pnl_pct >= TP_PCT:
                        close_position_market(symbol, 'long', size)
                        last_bar_ts[symbol] = cur_bar_ts
                        continue

                # Act only once per closed bar
                if acted_key == cur_bar_ts:
                    continue

                # Golden cross -> open long if no long position
                if golden_cross:
                    if not (size > 0 and side == 'long'):
                        ok = place_market_order(symbol, 'buy', BUDGET_USDT)
                        if ok:
                            last_bar_ts[symbol] = cur_bar_ts
                            continue
                # Death cross -> close existing long
                if death_cross and size > 0 and side == 'long':
                    ok = close_position_market(symbol, 'long', size)
                    if ok:
                        last_bar_ts[symbol] = cur_bar_ts
                        continue

            except Exception as e_sym:
                log.warning(f'{symbol} loop error: {e_sym}')
        time.sleep(SCAN_INTERVAL)
    except Exception as e:
        log.warning(f'Cycle error: {e}')
        time.sleep(SCAN_INTERVAL)
