#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OKX Trading Bot - SuperTrend + QQE MOD + A-V2 Strategy
三指标共振策略：SuperTrend(趋势) + QQE MOD(过滤) + A-V2(止损)
"""
import os
import time
import math
import logging
from typing import Dict, Any, Optional
import json
import urllib.request
from datetime import datetime
from threading import Thread

import ccxt
import pandas as pd
import numpy as np
from flask import Flask, render_template_string, jsonify

# ==================== 日志配置 ====================
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').strip().upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s [%(levelname)s] %(message)s'
)
log = logging.getLogger('okx-three-indicators')

# ==================== 通知配置 ====================
NOTIFY_WEBHOOK = os.environ.get('NOTIFY_WEBHOOK', '').strip()
NOTIFY_TYPE = os.environ.get('NOTIFY_TYPE', '').strip().lower()
NOTIFY_MENTION_MOBILES = [m.strip() for m in os.environ.get('NOTIFY_MENTION_MOBILES', '').split(',') if m.strip()]
PUSHPLUS_TOKEN = os.environ.get('PUSHPLUS_TOKEN', '').strip()
WXPUSHER_APP_TOKEN = os.environ.get('WXPUSHER_APP_TOKEN', '').strip()
WXPUSHER_UID = os.environ.get('WXPUSHER_UID', '').strip()

def _post_json(url: str, payload: dict):
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        urllib.request.urlopen(req, timeout=5).read()
    except Exception as e:
        log.debug(f'POST failed: {e}')

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
                'content': {'text': f"【{title}】\n{message}"}
            }
            _post_json(NOTIFY_WEBHOOK, payload)
        elif NOTIFY_TYPE == 'ding':
            payload = {
                'msgtype': 'text',
                'text': {'content': f"【{title}】\n{message}"}
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

# ==================== OKX配置 ====================
API_KEY = os.environ.get('OKX_API_KEY', '').strip()
API_SECRET = os.environ.get('OKX_SECRET_KEY', '').strip()
API_PASS = os.environ.get('OKX_PASSPHRASE', '').strip()
DRY_RUN = os.environ.get('DRY_RUN', 'false').strip().lower() in ('1', 'true', 'yes')

if not API_KEY or not API_SECRET or not API_PASS:
    if not DRY_RUN:
        raise SystemExit('Missing OKX credentials')
    else:
        log.warning('Running in DRY_RUN mode')

# ==================== 交易配置 ====================
BUDGET_USDT = float(os.environ.get('BUDGET_USDT', '10').strip() or 10)
DEFAULT_LEVERAGE = int(float(os.environ.get('DEFAULT_LEVERAGE', '5').strip() or 5))
TIMEFRAME = os.environ.get('TIMEFRAME', '4h').strip()
SCAN_INTERVAL = int(float(os.environ.get('SCAN_INTERVAL', '300').strip() or 300))
USE_BALANCE_AS_MARGIN = os.environ.get('USE_BALANCE_AS_MARGIN', 'true').strip().lower() in ('1', 'true', 'yes')
MARGIN_UTILIZATION = float(os.environ.get('MARGIN_UTILIZATION', '0.95').strip() or 0.95)

# ==================== 风险管理 ====================
RISK_PER_TRADE = float(os.environ.get('RISK_PER_TRADE', '0.02').strip() or 0.02)
RISK_REWARD_RATIO = float(os.environ.get('RISK_REWARD_RATIO', '2.0').strip() or 2.0)
MAX_POSITION_SIZE = float(os.environ.get('MAX_POSITION_SIZE', '0.1').strip() or 0.1)

# ==================== 指标参数 ====================
# SuperTrend
SUPERTREND_PERIOD = int(os.environ.get('SUPERTREND_PERIOD', '10').strip() or 10)
SUPERTREND_MULTIPLIER = float(os.environ.get('SUPERTREND_MULTIPLIER', '3.0').strip() or 3.0)

# QQE MOD
QQE_RSI_PERIOD = int(os.environ.get('QQE_RSI_PERIOD', '14').strip() or 14)
QQE_SF = int(os.environ.get('QQE_SF', '5').strip() or 5)

# A-V2
AV2_PERIOD = int(os.environ.get('AV2_PERIOD', '10').strip() or 10)
AV2_ATR_MULTIPLIER = float(os.environ.get('AV2_ATR_MULTIPLIER', '2.0').strip() or 2.0)

# ==================== 交易对配置 ====================
SYMBOLS = [
    'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 
    'XRP/USDT:USDT', 'ARB/USDT:USDT'
]

SYMBOL_LEVERAGE: Dict[str, int] = {
    'BTC/USDT:USDT': 100,
    'ETH/USDT:USDT': 100,
    'SOL/USDT:USDT': 50,
    'XRP/USDT:USDT': 50,
    'ARB/USDT:USDT': 50,
}

# ==================== 交易所初始化 ====================
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
        log.info(f'持仓模式设置 -> {"双向对冲" if POS_MODE == "hedge" else "单向净持仓"}')
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
    }
    markets_info[symbol] = info
    return info

def ensure_leverage(symbol: str):
    lev = int(SYMBOL_LEVERAGE.get(symbol, DEFAULT_LEVERAGE) or DEFAULT_LEVERAGE)
    inst_id = symbol_to_inst_id(symbol)
    try:
        exchange.privatePostAccountSetLeverage({'instId': inst_id, 'mgnMode': 'cross', 'lever': str(lev)})
        log.info(f'杠杆设置 {symbol} -> {lev}x')
    except Exception as e:
        log.warning(f'设置杠杆失败 {symbol}: {e}')

# ==================== 持仓管理 ====================
def get_positions_both(symbol: str) -> Dict[str, Dict[str, float]]:
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

def get_balance():
    try:
        balance = exchange.fetch_balance()
        usdt = balance.get('USDT', {})
        free = float(usdt.get('free') or usdt.get('available') or 0)
        used = float(usdt.get('used') or 0)
        total = float(usdt.get('total') or (free + used))
        return {'free': free, 'used': used, 'total': total}
    except Exception as e:
        log.warning(f'获取余额失败: {e}')
        return {'free': 0.0, 'used': 0.0, 'total': 0.0}

# ==================== 下单逻辑 ====================
def place_market_order(symbol: str, side: str, budget_usdt: float, position_ratio: float = 1.0) -> bool:
    if DRY_RUN:
        log.info(f'[DRY_RUN] 模拟开仓 {symbol} {side} 仓位比例={position_ratio*100:.0f}%')
        return True
    
    try:
        bal = get_balance()
        avail = bal['free']
    except Exception:
        avail = 0.0
    
    equity_usdt = max(0.0, avail) * position_ratio
    if equity_usdt <= 0:
        log.warning('余额不足')
        return False

    info = load_market_info(symbol)
    inst_id = info['instId']
    ticker = exchange.fetch_ticker(symbol)
    price = float(ticker.get('last') or ticker.get('close') or 0)
    if price <= 0:
        raise Exception('invalid price')
    
    ct_val = float(info.get('ctVal') or 0.01)

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
        log.warning(f'合约张数过小: {contracts}')
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
        log.info(f'下单成功 {symbol}: 方向={side} 数量={contracts}')
        notify_event('开仓成功', f'{symbol} {side} 数量={contracts}')
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
        log.info(f'平仓成功 {symbol} 方向={side_to_close} 数量={sz}')
        notify_event('平仓成功', f'{symbol} {side_to_close} 数量={sz}')
        return True
    except Exception as e:
        log.warning(f'平仓失败 {symbol}: {e}')
        return False

# ==================== 技术指标计算 ====================
class Indicators:
    @staticmethod
    def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
        """计算SuperTrend指标"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        hl_avg = (high + low) / 2
        upper_band = hl_avg + (multiplier * atr)
        lower_band = hl_avg - (multiplier * atr)
        
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)
        
        for i in range(period, len(df)):
            if i == period:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            else:
                if close.iloc[i] > supertrend.iloc[i-1]:
                    supertrend.iloc[i] = lower_band.iloc[i]
                    direction.iloc[i] = 1
                elif close.iloc[i] < supertrend.iloc[i-1]:
                    supertrend.iloc[i] = upper_band.iloc[i]
                    direction.iloc[i] = -1
                else:
                    supertrend.iloc[i] = supertrend.iloc[i-1]
                    direction.iloc[i] = direction.iloc[i-1]
                    
                    if direction.iloc[i] == 1 and lower_band.iloc[i] > supertrend.iloc[i-1]:
                        supertrend.iloc[i] = lower_band.iloc[i]
                    elif direction.iloc[i] == -1 and upper_band.iloc[i] < supertrend.iloc[i-1]:
                        supertrend.iloc[i] = upper_band.iloc[i]
        
        return supertrend, direction
    
    @staticmethod
    def calculate_qqe_mod(df: pd.DataFrame, rsi_period: int = 14, sf: int = 5):
        """计算QQE MOD指标"""
        close = df['close']
        
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        rsi_ma = rsi.ewm(span=sf, adjust=False).mean()
        atr_rsi = abs(rsi - rsi.shift()).rolling(window=rsi_period).mean()
        dar = atr_rsi.ewm(span=sf, adjust=False).mean() * 4.236
        
        long_band = rsi_ma - dar
        short_band = rsi_ma + dar
        
        trend = pd.Series(index=df.index, dtype=float)
        for i in range(rsi_period, len(df)):
            if rsi.iloc[i] > short_band.iloc[i]:
                trend.iloc[i] = 1
            elif rsi.iloc[i] < long_band.iloc[i]:
                trend.iloc[i] = -1
            else:
                trend.iloc[i] = 0
        
        return rsi, trend
    
    @staticmethod
    def calculate_a_v2(df: pd.DataFrame, period: int = 10, atr_multiplier: float = 2.0):
        """计算A-V2指标"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        ema = close.ewm(span=period, adjust=False).mean()
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        stop_loss_long = ema - (atr_multiplier * atr)
        stop_loss_short = ema + (atr_multiplier * atr)
        
        trend = pd.Series(index=df.index, dtype=int)
        trend[close > ema] = 1
        trend[close < ema] = -1
        
        return ema, stop_loss_long, stop_loss_short, trend, atr

# ==================== 策略逻辑 ====================
class TradingStrategy:
    def __init__(self):
        self.symbol_state: Dict[str, Dict[str, Any]] = {}
        self.last_bar_ts: Dict[str, int] = {}
        self.stats = {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'realized_pnl': 0.0,
            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        for sym in SYMBOLS:
            self.symbol_state[sym] = {
                'st_direction': 0,
                'qqe_trend': 0,
                'av2_trend': 0,
                'price': 0.0,
                'stop_loss_long': 0.0,
                'stop_loss_short': 0.0,
                'atr': 0.0,
                'signal': 'HOLD',
                'last_update': ''
            }
    
    def get_historical_data(self, symbol: str, limit: int = 100) -> Optional[pd.DataFrame]:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=limit)
            if not ohlcv or len(ohlcv) < 50:
                return None
            
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            log.warning(f'获取数据失败 {symbol}: {e}')
            return None
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        try:
            st, st_direction = Indicators.calculate_supertrend(
                df, SUPERTREND_PERIOD, SUPERTREND_MULTIPLIER
            )
            df['supertrend'] = st
            df['st_direction'] = st_direction
            
            rsi, qqe_trend = Indicators.calculate_qqe_mod(
                df, QQE_RSI_PERIOD, QQE_SF
            )
            df['rsi'] = rsi
            df['qqe_trend'] = qqe_trend
            
            ema, sl_long, sl_short, av2_trend, atr = Indicators.calculate_a_v2(
                df, AV2_PERIOD, AV2_ATR_MULTIPLIER
            )
            df['ema'] = ema
            df['stop_loss_long'] = sl_long
            df['stop_loss_short'] = sl_short
            df['av2_trend'] = av2_trend
            df['atr'] = atr
            
            return df
        except Exception as e:
            log.warning(f'指标计算失败: {e}')
            return None
    
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> str:
        if df is None or len(df) < 2:
            return 'HOLD'
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # 做多条件：三指标共振
        long_conditions = [
            current['st_direction'] == 1,
            current['qqe_trend'] == 1,
            current['av2_trend'] == 1,
            previous['st_direction'] != 1
        ]
        
        # 做空条件：三指标共振
        short_conditions = [
            current['st_direction'] == -1,
            current['qqe_trend'] == -1,
            current['av2_trend'] == -1,
            previous['st_direction'] != -1
        ]
        
        # 更新状态
        self.symbol_state[symbol].update({
            'st_direction': int(current['st_direction']),
            'qqe_trend': int(current['qqe_trend']),
            'av2_trend': int(current['av2_trend']),
            'price': float(current['close']),
            'stop_loss_long': float(current['stop_loss_long']),
            'stop_loss_short': float(current['stop_loss_short']),
            'atr': float(current['atr']),
            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
        if all(long_conditions):
            self.symbol_state[symbol]['signal'] = 'BUY'
            return 'BUY'
        elif all(short_conditions):
            self.symbol_state[symbol]['signal'] = 'SELL'
            return 'SELL'
        else:
            self.symbol_state[symbol]['signal'] = 'HOLD'
            return 'HOLD'
    
    def check_exit_conditions(self, symbol: str, df: pd.DataFrame) -> bool:
        both = get_positions_both(symbol)
        long_size = both['long']['size']
        long_entry = both['long']['entry']
        short_size = both['short']['size']
        short_entry = both['short']['entry']
        
        if long_size <= 0 and short_size <= 0:
            return False
        
        current = df.iloc[-1]
        price = float(current['close'])
        
        try:
            ct_val = float(load_market_info(symbol).get('ctVal') or 0.01)
        except:
            ct_val = 0.01
        
        # 多头出场
        if long_size > 0 and long_entry > 0:
            stop_loss = float(current['stop_loss_long'])
            pnl_pct = (price - long_entry) / long_entry
            
            # 止损
            if price <= stop_loss or current['st_direction'] == -1:
                realized = long_size * ct_val * (price - long_entry)
                ok = close_position_market(symbol, 'long', long_size)
                if ok:
                    self.stats['trades'] += 1
                    if realized > 0:
                        self.stats['wins'] += 1
                    else:
                        self.stats['losses'] += 1
                    self.stats['realized_pnl'] += realized
                    self.stats['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    log.info(f'{symbol} 多头止损/反转: 已实现={realized:.2f}')
                    notify_event('多头平仓', f'{symbol} 已实现={realized:.2f}')
                    return True
            
            # 止盈
            target_profit = RISK_REWARD_RATIO * abs(long_entry - stop_loss) / long_entry
            if pnl_pct >= target_profit:
                realized = long_size * ct_val * (price - long_entry)
                ok = close_position_market(symbol, 'long', long_size)
                if ok:
                    self.stats['trades'] += 1
                    if realized > 0:
                        self.stats['wins'] += 1
                    else:
                        self.stats['losses'] += 1
                    self.stats['realized_pnl'] += realized
                    self.stats['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    log.info(f'{symbol} 多头止盈: 已实现={realized:.2f}')
                    notify_event('多头止盈', f'{symbol} 已实现={realized:.2f}')
                    return True
        
        # 空头出场
        if short_size > 0 and short_entry > 0:
            stop_loss = float(current['stop_loss_short'])
            pnl_pct = (short_entry - price) / short_entry
            
            # 止损
            if price >= stop_loss or current['st_direction'] == 1:
                realized = short_size * ct_val * (short_entry - price)
                ok = close_position_market(symbol, 'short', short_size)
                if ok:
                    self.stats['trades'] += 1
                    if realized > 0:
                        self.stats['wins'] += 1
                    else:
                        self.stats['losses'] += 1
                    self.stats['realized_pnl'] += realized
                    self.stats['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    log.info(f'{symbol} 空头止损/反转: 已实现={realized:.2f}')
                    notify_event('空头平仓', f'{symbol} 已实现={realized:.2f}')
                    return True
            
            # 止盈
            target_profit = RISK_REWARD_RATIO * abs(stop_loss - short_entry) / short_entry
            if pnl_pct >= target_profit:
                realized = short_size * ct_val * (short_entry - price)
                ok = close_position_market(symbol, 'short', short_size)
                if ok:
                    self.stats['trades'] += 1
                    if realized > 0:
                        self.stats['wins'] += 1
                    else:
                        self.stats['losses'] += 1
                    self.stats['realized_pnl'] += realized
                    self.stats['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    log.info(f'{symbol} 空头止盈: 已实现={realized:.2f}')
                    notify_event('空头止盈', f'{symbol} 已实现={realized:.2f}')
                    return True
        
        return False
    
    def process_symbol(self, symbol: str):
        try:
            df = self.get_historical_data(symbol)
            if df is None:
                return
            
            df = self.calculate_all_indicators(df)
            if df is None:
                return
            
            # 检查出场条件
            exited = self.check_exit_conditions(symbol, df)
            if exited:
                cur_bar_ts = int(df.iloc[-1]['timestamp'].timestamp())
                self.last_bar_ts[symbol] = cur_bar_ts
                return
            
            # 检查入场信号
            both = get_positions_both(symbol)
            long_size = both['long']['size']
            short_size = both['short']['size']
            
            if long_size > 0 or short_size > 0:
                return
            
            # 防止重复操作
            cur_bar_ts = int(df.iloc[-1]['timestamp'].timestamp())
            if self.last_bar_ts.get(symbol) == cur_bar_ts:
                return
            
            signal = self.generate_signal(df, symbol)
            
            if signal == 'BUY':
                ok = place_market_order(symbol, 'buy', BUDGET_USDT)
                if ok:
                    log.info(f'{symbol} 三指标共振做多信号')
                    notify_event('开多仓', f'{symbol} SuperTrend+QQE+AV2共振')
                    self.last_bar_ts[symbol] = cur_bar_ts
            
            elif signal == 'SELL':
                ok = place_market_order(symbol, 'sell', BUDGET_USDT)
                if ok:
                    log.info(f'{symbol} 三指标共振做空信号')
                    notify_event('开空仓', f'{symbol} SuperTrend+QQE+AV2共振')
                    self.last_bar_ts[symbol] = cur_bar_ts
        
        except Exception as e:
            log.warning(f'{symbol} 处理异常: {e}')
    
    def run(self):
        log.info('=' * 70)
        log.info(f'三指标策略启动 - {TIMEFRAME}')
        log.info(f'SuperTrend: 周期={SUPERTREND_PERIOD}, 乘数={SUPERTREND_MULTIPLIER}')
        log.info(f'QQE MOD: RSI周期={QQE_RSI_PERIOD}, SF={QQE_SF}')
        log.info(f'A-V2: 周期={AV2_PERIOD}, ATR乘数={AV2_ATR_MULTIPLIER}')
        log.info('=' * 70)
        
        if not DRY_RUN:
            ensure_position_mode()
            for sym in SYMBOLS:
                ensure_leverage(sym)
        
        cycle = 0
        while True:
            try:
                cycle += 1
                bal = get_balance()
                winrate = (self.stats['wins'] / self.stats['trades'] * 100) if self.stats['trades'] > 0 else 0.0
                
                log.info(f'周期 {cycle}: 余额={bal["free"]:.2f}/{bal["total"]:.2f} USDT | '
                        f'累计盈亏={self.stats["realized_pnl"]:.2f} | '
                        f'胜率={winrate:.1f}% ({self.stats["wins"]}/{self.stats["trades"]})')
                
                for symbol in SYMBOLS:
                    self.process_symbol(symbol)
                
                time.sleep(SCAN_INTERVAL)
                
            except KeyboardInterrupt:
                log.info('收到退出信号')
                break
            except Exception as e:
                log.warning(f'主循环异常: {e}')
                time.sleep(SCAN_INTERVAL)

# ==================== Web界面 ====================
app = Flask(__name__)
strategy = TradingStrategy()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>OKX三指标策略监控</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
            min-height: 100vh;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        .header {
            background: white;
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }
        .header h1 { 
            font-size: 28px; 
            color: #667eea;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
        }
        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            margin-left: 15px;
        }
        .status-live { background: #10b981; color: white; }
        .status-dry { background: #f59e0b; color: white; }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .stat-label { font-size: 12px; opacity: 0.9; margin-bottom: 5px; }
        .stat-value { font-size: 28px; font-weight: 700; }
        .stat-small { font-size: 14px; margin-top: 5px; opacity: 0.8; }
        
        .positions-section {
            background: white;
            border-radius: 16px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }
        .section-title {
            font-size: 20px;
            font-weight: 700;
            color: #333;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
        }
        .section-title::before {
            content: '';
            width: 4px;
            height: 24px;
            background: #667eea;
            margin-right: 10px;
            border-radius: 2px;
        }
        
        .position-grid {
            display: grid;
            gap: 15px;
        }
        .position-card {
            background: #f9fafb;
            border-radius: 12px;
            padding: 20px;
            border-left: 4px solid #667eea;
        }
        .position-card.long { border-left-color: #10b981; }
        .position-card.short { border-left-color: #ef4444; }
        .position-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        .symbol { font-size: 18px; font-weight: 700; color: #111; }
        .position-badge {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }
        .badge-long { background: #d1fae5; color: #065f46; }
        .badge-short { background: #fee2e2; color: #991b1b; }
        .position-details {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }
        .detail-item {
            background: white;
            padding: 12px;
            border-radius: 8px;
        }
        .detail-label { font-size: 11px; color: #6b7280; margin-bottom: 4px; }
        .detail-value { font-size: 16px; font-weight: 600; color: #111; }
        .pnl-positive { color: #10b981; }
        .pnl-negative { color: #ef4444; }
        
        .symbols-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
        }
        .symbol-card {
            background: #f9fafb;
            border-radius: 12px;
            padding: 20px;
            border: 2px solid #e5e7eb;
            transition: all 0.3s;
        }
        .symbol-card:hover {
            border-color: #667eea;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
        }
        .symbol-card.signal-buy { border-color: #10b981; background: #f0fdf4; }
        .symbol-card.signal-sell { border-color: #ef4444; background: #fef2f2; }
        
        .symbol-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        .signal-badge {
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 700;
        }
        .signal-buy-badge { background: #10b981; color: white; }
        .signal-sell-badge { background: #ef4444; color: white; }
        .signal-hold-badge { background: #6b7280; color: white; }
        
        .indicators {
            display: flex;
            gap: 10px;
            margin-bottom: 10px;
        }
        .indicator {
            flex: 1;
            background: white;
            padding: 10px;
            border-radius: 8px;
            text-align: center;
        }
        .indicator-label { font-size: 10px; color: #6b7280; margin-bottom: 4px; }
        .indicator-value {
            font-size: 14px;
            font-weight: 700;
        }
        .indicator-up { color: #10b981; }
        .indicator-down { color: #ef4444; }
        .indicator-flat { color: #6b7280; }
        
        .price-info { font-size: 24px; font-weight: 700; color: #111; margin: 10px 0; }
        .last-update { font-size: 11px; color: #9ca3af; }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }
        .loading { animation: pulse 2s infinite; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>
                🚀 OKX三指标策略监控
                <span class="status-badge status-{{ 'dry' if dry_run else 'live' }}">
                    {{ 'DRY RUN' if dry_run else 'LIVE' }}
                </span>
            </h1>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">账户余额</div>
                    <div class="stat-value" id="balance">-</div>
                    <div class="stat-small" id="balance-used">-</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">累计盈亏</div>
                    <div class="stat-value" id="pnl">-</div>
                    <div class="stat-small">已实现</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">交易次数</div>
                    <div class="stat-value" id="trades">-</div>
                    <div class="stat-small" id="winrate">-</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">最后更新</div>
                    <div class="stat-value" style="font-size: 16px;" id="last-update">-</div>
                    <div class="stat-small">自动刷新中...</div>
                </div>
            </div>
        </div>
        
        <div class="positions-section">
            <div class="section-title">💼 当前持仓</div>
            <div class="position-grid" id="positions">
                <div style="text-align: center; padding: 40px; color: #9ca3af;">
                    暂无持仓
                </div>
            </div>
        </div>
        
        <div class="positions-section">
            <div class="section-title">📊 交易对状态</div>
            <div class="symbols-grid" id="symbols"></div>
        </div>
    </div>
    
    <script>
        function updateDashboard() {
            fetch('/api/status')
                .then(r => r.json())
                .then(data => {
                    // 更新统计
                    document.getElementById('balance').textContent = 
                        data.balance.free.toFixed(2) + ' USDT';
                    document.getElementById('balance-used').textContent = 
                        '已用: ' + data.balance.used.toFixed(2) + ' USDT';
                    
                    const pnl = data.stats.realized_pnl;
                    document.getElementById('pnl').textContent = 
                        (pnl >= 0 ? '+' : '') + pnl.toFixed(2) + ' USDT';
                    document.getElementById('pnl').className = 
                        'stat-value ' + (pnl >= 0 ? 'pnl-positive' : 'pnl-negative');
                    
                    document.getElementById('trades').textContent = data.stats.trades;
                    const winrate = data.stats.trades > 0 ? 
                        (data.stats.wins / data.stats.trades * 100).toFixed(1) : 0;
                    document.getElementById('winrate').textContent = 
                        '胜率: ' + winrate + '% (' + data.stats.wins + '/' + data.stats.trades + ')';
                    
                    document.getElementById('last-update').textContent = 
                        data.stats.last_update;
                    
                    // 更新持仓
                    const positionsHtml = data.positions.map(p => `
                        <div class="position-card ${p.side}">
                            <div class="position-header">
                                <span class="symbol">${p.symbol}</span>
                                <span class="position-badge badge-${p.side}">
                                    ${p.side.toUpperCase()}
                                </span>
                            </div>
                            <div class="position-details">
                                <div class="detail-item">
                                    <div class="detail-label">开仓价格</div>
                                    <div class="detail-value">${p.entry.toFixed(6)}</div>
                                </div>
                                <div class="detail-item">
                                    <div class="detail-label">当前价格</div>
                                    <div class="detail-value">${p.current_price.toFixed(6)}</div>
                                </div>
                                <div class="detail-item">
                                    <div class="detail-label">仓位大小</div>
                                    <div class="detail-value">${p.size}</div>
                                </div>
                                <div class="detail-item">
                                    <div class="detail-label">浮动盈亏</div>
                                    <div class="detail-value ${p.pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}">
                                        ${(p.pnl >= 0 ? '+' : '')}${p.pnl.toFixed(2)} USDT
                                    </div>
                                </div>
                                <div class="detail-item">
                                    <div class="detail-label">盈亏比例</div>
                                    <div class="detail-value ${p.pnl_pct >= 0 ? 'pnl-positive' : 'pnl-negative'}">
                                        ${(p.pnl_pct >= 0 ? '+' : '')}${p.pnl_pct.toFixed(2)}%
                                    </div>
                                </div>
                                <div class="detail-item">
                                    <div class="detail-label">止损价格</div>
                                    <div class="detail-value">${p.stop_loss.toFixed(6)}</div>
                                </div>
                            </div>
                        </div>
                    `).join('');
                    
                    document.getElementById('positions').innerHTML = 
                        positionsHtml || '<div style="text-align: center; padding: 40px; color: #9ca3af;">暂无持仓</div>';
                    
                    // 更新交易对状态
                    const symbolsHtml = data.symbols.map(s => `
                        <div class="symbol-card signal-${s.signal.toLowerCase()}">
                            <div class="symbol-header">
                                <span class="symbol">${s.symbol.split('/')[0]}</span>
                                <span class="signal-badge signal-${s.signal.toLowerCase()}-badge">
                                    ${s.signal}
                                </span>
                            </div>
                            <div class="price-info">${s.price.toFixed(6)}</div>
                            <div class="indicators">
                                <div class="indicator">
                                    <div class="indicator-label">SuperTrend</div>
                                    <div class="indicator-value ${s.st_direction > 0 ? 'indicator-up' : s.st_direction < 0 ? 'indicator-down' : 'indicator-flat'}">
                                        ${s.st_direction > 0 ? '↑ 看涨' : s.st_direction < 0 ? '↓ 看跌' : '→ 中性'}
                                    </div>
                                </div>
                                <div class="indicator">
                                    <div class="indicator-label">QQE MOD</div>
                                    <div class="indicator-value ${s.qqe_trend > 0 ? 'indicator-up' : s.qqe_trend < 0 ? 'indicator-down' : 'indicator-flat'}">
                                        ${s.qqe_trend > 0 ? '↑ 趋势' : s.qqe_trend < 0 ? '↓ 趋势' : '→ 震荡'}
                                    </div>
                                </div>
                                <div class="indicator">
                                    <div class="indicator-label">A-V2</div>
                                    <div class="indicator-value ${s.av2_trend > 0 ? 'indicator-up' : s.av2_trend < 0 ? 'indicator-down' : 'indicator-flat'}">
                                        ${s.av2_trend > 0 ? '↑ 多头' : s.av2_trend < 0 ? '↓ 空头' : '→ 中性'}
                                    </div>
                                </div>
                            </div>
                            <div class="last-update">更新: ${s.last_update}</div>
                        </div>
                    `).join('');
                    
                    document.getElementById('symbols').innerHTML = symbolsHtml;
                })
                .catch(err => console.error('Error:', err));
        }
        
        updateDashboard();
        setInterval(updateDashboard, 5000);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, dry_run=DRY_RUN)

@app.route('/api/status')
def api_status():
    try:
        bal = get_balance()
        
        positions = []
        for symbol in SYMBOLS:
            both = get_positions_both(symbol)
            
            if both['long']['size'] > 0:
                state = strategy.symbol_state.get(symbol, {})
                try:
                    ticker = exchange.fetch_ticker(symbol)
                    current_price = float(ticker['last'])
                    ct_val = float(load_market_info(symbol).get('ctVal') or 0.01)
                    pnl = both['long']['size'] * ct_val * (current_price - both['long']['entry'])
                    pnl_pct = (current_price - both['long']['entry']) / both['long']['entry'] * 100
                except:
                    current_price = 0
                    pnl = 0
                    pnl_pct = 0
                
                positions.append({
                    'symbol': symbol,
                    'side': 'long',
                    'size': both['long']['size'],
                    'entry': both['long']['entry'],
                    'current_price': current_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'stop_loss': state.get('stop_loss_long', 0)
                })
            
            if both['short']['size'] > 0:
                state = strategy.symbol_state.get(symbol, {})
                try:
                    ticker = exchange.fetch_ticker(symbol)
                    current_price = float(ticker['last'])
                    ct_val = float(load_market_info(symbol).get('ctVal') or 0.01)
                    pnl = both['short']['size'] * ct_val * (both['short']['entry'] - current_price)
                    pnl_pct = (both['short']['entry'] - current_price) / both['short']['entry'] * 100
                except:
                    current_price = 0
                    pnl = 0
                    pnl_pct = 0
                
                positions.append({
                    'symbol': symbol,
                    'side': 'short',
                    'size': both['short']['size'],
                    'entry': both['short']['entry'],
                    'current_price': current_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'stop_loss': state.get('stop_loss_short', 0)
                })
        
        symbols_status = []
        for symbol in SYMBOLS:
            state = strategy.symbol_state.get(symbol, {})
            symbols_status.append({
                'symbol': symbol,
                'price': state.get('price', 0),
                'signal': state.get('signal', 'HOLD'),
                'st_direction': state.get('st_direction', 0),
                'qqe_trend': state.get('qqe_trend', 0),
                'av2_trend': state.get('av2_trend', 0),
                'last_update': state.get('last_update', '')
            })
        
        return jsonify({
            'balance': bal,
            'stats': strategy.stats,
            'positions': positions,
            'symbols': symbols_status
        })
    except Exception as e:
        log.error(f'API error: {e}')
        return jsonify({'error': str(e)}), 500

def run_web_server():
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

# ==================== 主程序 ====================
if __name__ == "__main__":
    # 启动Web服务器（后台线程）
    web_thread = Thread(target=run_web_server, daemon=True)
    web_thread.start()
    log.info(f'Web界面已启动: http://0.0.0.0:{os.environ.get("PORT", 8080)}')
    
    # 启动策略主循环
    strategy.run()
