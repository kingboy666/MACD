#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纯MACD策略 (OKX USDT-SWAP, 30m)
- 指标: MACD(14,30,10) 金叉做多/死叉做空
- 交易时段: 仅在高峰时段 (UTC 12:00-16:00, 20:00-24:00)
- 风控: 动态ATR止损 + 固定止盈
"""
import os
import time
import math
import logging
from datetime import datetime
from typing import Dict, Any
import json
import urllib.request

import ccxt
import pandas as pd

LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').strip().upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), 
                   format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('macd-simple')

# ========== 配置区 ==========
API_KEY = os.environ.get('OKX_API_KEY', '').strip()
API_SECRET = os.environ.get('OKX_SECRET_KEY', '').strip()
API_PASS = os.environ.get('OKX_PASSPHRASE', '').strip()
DRY_RUN = os.environ.get('DRY_RUN', 'false').strip().lower() in ('1', 'true', 'yes')

# 资金与杠杆
BUDGET_USDT = float(os.environ.get('BUDGET_USDT', '5').strip() or 5)
DEFAULT_LEVERAGE = int(float(os.environ.get('DEFAULT_LEVERAGE', '20').strip() or 20))

# 止盈止损
TP_PCT = float(os.environ.get('TP_PCT', '0.04').strip() or 0.04)  # 4% 固定止盈
SL_ATR_MULTIPLIER = float(os.environ.get('SL_ATR_MULTIPLIER', '2.5').strip() or 2.5)  # ATR止损倍数

# MACD参数
MACD_FAST = int(os.environ.get('MACD_FAST', '14').strip() or 14)
MACD_SLOW = int(os.environ.get('MACD_SLOW', '30').strip() or 30)
MACD_SIGNAL = int(os.environ.get('MACD_SIGNAL', '10').strip() or 10)

# ATR参数
ATR_PERIOD = int(os.environ.get('ATR_PERIOD', '14').strip() or 14)

# 交易时段 (UTC时间)
TRADING_HOURS = [
    (12, 16),  # 北京晚上 20:00-24:00
    (20, 24),  # 北京早上 04:00-08:00
]

# 扫描间隔
SCAN_INTERVAL = int(float(os.environ.get('SCAN_INTERVAL', '60').strip() or 60))

# 交易对
TIMEFRAME = '30m'
SYMBOLS = [
    'FIL/USDT:USDT', 'ZRO/USDT:USDT', 'WIF/USDT:USDT', 'WLD/USDT:USDT',
    'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'XRP/USDT:USDT',
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
}

# ========== 通知配置 ==========
NOTIFY_WEBHOOK = os.environ.get('NOTIFY_WEBHOOK', '').strip()
NOTIFY_TYPE = os.environ.get('NOTIFY_TYPE', '').strip().lower()

def notify_event(title: str, message: str):
    """简化通知函数"""
    if not NOTIFY_WEBHOOK or not NOTIFY_TYPE:
        return
    try:
        if NOTIFY_TYPE == 'wecom':
            payload = {
                'msgtype': 'text',
                'text': {'content': f"【{title}】\n{message}"}
            }
        elif NOTIFY_TYPE == 'feishu':
            payload = {
                'msg_type': 'text',
                'content': {'text': f"【{title}】\n{message}"}
            }
        else:
            payload = {'title': title, 'message': message}
        
        req = urllib.request.Request(
            NOTIFY_WEBHOOK,
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        log.warning(f'通知发送失败: {e}')

# ========== 初始化交易所 ==========
if not API_KEY or not API_SECRET or not API_PASS:
    if not DRY_RUN:
        raise SystemExit('缺少OKX凭证: 请设置 OKX_API_KEY, OKX_SECRET_KEY, OKX_PASSPHRASE')
    log.warning('DRY_RUN模式: 无需API密钥')

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

POS_MODE = 'net'  # 单向持仓模式

# ========== 工具函数 ==========
markets_info: Dict[str, Dict[str, Any]] = {}

def symbol_to_inst_id(sym: str) -> str:
    base = sym.split('/')[0]
    return f'{base}-USDT-SWAP'

def load_market_info(symbol: str) -> Dict[str, Any]:
    """加载合约信息"""
    if symbol in markets_info:
        return markets_info[symbol]
    inst_id = symbol_to_inst_id(symbol)
    resp = exchange.publicGetPublicInstruments({'instType': 'SWAP', 'instId': inst_id})
    data = (resp.get('data') or [])[0]
    info = {
        'instId': inst_id,
        'ctVal': float(data.get('ctVal', 0) or 0),
        'lotSz': float(data.get('lotSz', 0) or 0),
        'minSz': float(data.get('minSz', 0) or 0),
        'tickSz': float(data.get('tickSz', 0) or 0),
    }
    markets_info[symbol] = info
    return info

def ensure_leverage(symbol: str):
    """设置杠杆"""
    lev = int(SYMBOL_LEVERAGE.get(symbol, DEFAULT_LEVERAGE) or DEFAULT_LEVERAGE)
    inst_id = symbol_to_inst_id(symbol)
    try:
        exchange.privatePostAccountSetLeverage({
            'instId': inst_id, 
            'mgnMode': 'cross', 
            'lever': str(lev)
        })
        log.info(f'已设置杠杆 {symbol} -> {lev}x')
    except Exception as e:
        log.warning(f'设置杠杆失败 {symbol}: {e}')

def get_position(symbol: str) -> Dict[str, Any]:
    """获取当前持仓"""
    inst_id = symbol_to_inst_id(symbol)
    try:
        resp = exchange.privateGetAccountPositions({
            'instType': 'SWAP', 
            'instId': inst_id
        })
        for p in resp.get('data', []):
            if p.get('instId') == inst_id:
                pos = float(p.get('pos', 0) or 0)
                if pos == 0:
                    continue
                size = abs(pos)
                side = 'long' if pos > 0 else 'short'
                entry = float(p.get('avgPx') or 0)
                return {'size': size, 'side': side, 'entry': entry}
    except Exception as e:
        log.debug(f'获取持仓失败: {e}')
    return {'size': 0.0, 'side': None, 'entry': 0.0}

def place_market_order(symbol: str, side: str) -> bool:
    """市价开仓"""
    if DRY_RUN:
        log.info(f'[DRY_RUN] 模拟开仓 {symbol} {side}')
        return True
    
    try:
        info = load_market_info(symbol)
        inst_id = info['instId']
        ticker = exchange.fetch_ticker(symbol)
        price = float(ticker.get('last') or 0)
        
        if price <= 0:
            log.warning(f'{symbol} 价格异常')
            return False
        
        ct_val = float(info.get('ctVal') or 0)
        if ct_val <= 0:
            ct_val = 0.01
        
        # 计算合约张数
        contracts = (BUDGET_USDT / price) / ct_val
        lot = float(info.get('lotSz') or 0)
        minsz = float(info.get('minSz') or 0)
        
        if lot > 0:
            contracts = math.floor(contracts / lot) * lot
        
        if contracts < minsz:
            log.warning(f'{symbol} 张数不足: {contracts} < {minsz}')
            return False
        
        side_okx = 'buy' if side == 'long' else 'sell'
        params = {
            'instId': inst_id,
            'tdMode': 'cross',
            'side': side_okx,
            'ordType': 'market',
            'sz': str(contracts),
        }
        
        exchange.privatePostTradeOrder(params)
        log.info(f'✅ 开仓成功 {symbol} {side} 数量={contracts} 价格≈{price:.4f}')
        notify_event('开仓成功', f'{symbol} {side} {contracts}张 @{price:.4f}')
        return True
        
    except Exception as e:
        log.warning(f'❌ 开仓失败 {symbol}: {e}')
        return False

def close_position_market(symbol: str, side: str, qty: float) -> bool:
    """市价平仓"""
    if DRY_RUN:
        log.info(f'[DRY_RUN] 模拟平仓 {symbol} {side} {qty}')
        return True
    
    try:
        info = load_market_info(symbol)
        inst_id = info['instId']
        side_okx = 'sell' if side == 'long' else 'buy'
        
        lot = float(info.get('lotSz') or 0)
        sz = qty
        if lot > 0:
            sz = math.floor(sz / lot) * lot
        
        params = {
            'instId': inst_id,
            'tdMode': 'cross',
            'side': side_okx,
            'ordType': 'market',
            'sz': str(sz),
            'reduceOnly': True,
        }
        
        exchange.privatePostTradeOrder(params)
        log.info(f'✅ 平仓成功 {symbol} {side} 数量={sz}')
        return True
        
    except Exception as e:
        log.warning(f'❌ 平仓失败 {symbol}: {e}')
        return False

# ========== 技术指标计算 ==========
def calc_macd(closes: pd.Series, fast: int = 14, slow: int = 30, signal: int = 10):
    """计算MACD"""
    ema_fast = closes.ewm(span=fast, adjust=False).mean()
    ema_slow = closes.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def calc_atr(highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 14):
    """计算ATR"""
    tr_components = pd.concat([
        (highs - lows).abs(),
        (highs - closes.shift()).abs(),
        (lows - closes.shift()).abs()
    ], axis=1)
    tr = tr_components.max(axis=1)
    return tr.rolling(period).mean()

def is_trading_time() -> bool:
    """检查是否在交易时段"""
    now = datetime.utcnow()
    current_hour = now.hour
    
    for start, end in TRADING_HOURS:
        if start <= current_hour < end:
            return True
    return False

# ========== 主策略 ==========
last_bar_ts: Dict[str, int] = {}
stats = {'trades': 0, 'wins': 0, 'losses': 0, 'realized_pnl': 0.0}

log.info('=' * 70)
log.info(f'MACD纯策略启动 - {TIMEFRAME}周期')
log.info(f'MACD参数: ({MACD_FAST},{MACD_SLOW},{MACD_SIGNAL})')
log.info(f'止盈: {TP_PCT*100}% | ATR止损倍数: {SL_ATR_MULTIPLIER}')
log.info(f'交易时段 (UTC): {TRADING_HOURS}')
log.info(f'预算: {BUDGET_USDT} USDT | 模拟: {DRY_RUN}')
log.info('=' * 70)

if not DRY_RUN:
    for sym in SYMBOLS:
        ensure_leverage(sym)

cycle = 0
while True:
    try:
        cycle += 1
        
        # 检查交易时段
        if not is_trading_time():
            if cycle % 10 == 1:  # 每10次循环提示一次
                now = datetime.utcnow()
                log.info(f'非交易时段 (当前UTC: {now.hour:02d}:{now.minute:02d}), 等待中...')
            time.sleep(SCAN_INTERVAL)
            continue
        
        # 获取账户信息
        if not DRY_RUN:
            try:
                balance = exchange.fetch_balance()
                usdt = balance.get('USDT', {})
                free = float(usdt.get('free') or 0)
                total = float(usdt.get('total') or 0)
            except:
                free, total = 0.0, 0.0
        else:
            free, total = 0.0, 0.0
        
        winrate = (stats['wins']/stats['trades']*100) if stats['trades']>0 else 0
        log.info(f'[周期 {cycle}] 账户: {free:.2f}/{total:.2f} USDT | '
                f'盈亏: {stats["realized_pnl"]:.2f} | 胜率: {winrate:.1f}%')
        
        for symbol in SYMBOLS:
            try:
                # 获取K线数据
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=100)
                if not ohlcv or len(ohlcv) < max(MACD_SLOW, ATR_PERIOD) + 20:
                    continue
                
                closes = pd.Series([x[4] for x in ohlcv])
                highs = pd.Series([x[2] for x in ohlcv])
                lows = pd.Series([x[3] for x in ohlcv])
                
                # 计算指标
                macd_line, signal_line, hist = calc_macd(
                    closes, MACD_FAST, MACD_SLOW, MACD_SIGNAL
                )
                atr = calc_atr(highs, lows, closes, ATR_PERIOD)
                
                # 当前值
                price = float(closes.iloc[-1])
                macd = float(macd_line.iloc[-1])
                signal = float(signal_line.iloc[-1])
                macd_prev = float(macd_line.iloc[-2])
                signal_prev = float(signal_line.iloc[-2])
                curr_atr = float(atr.iloc[-1])
                
                # MACD金叉/死叉
                golden_cross = (macd_prev <= signal_prev) and (macd > signal)
                dead_cross = (macd_prev >= signal_prev) and (macd < signal)
                
                # 获取持仓
                pos = get_position(symbol)
                has_long = pos['size'] > 0 and pos['side'] == 'long'
                has_short = pos['size'] > 0 and pos['side'] == 'short'
                
                # ========== 风控: 止盈止损 ==========
                if has_long:
                    entry = pos['entry']
                    pnl_pct = (price - entry) / entry
                    sl_price = entry - (SL_ATR_MULTIPLIER * curr_atr)
                    
                    # 动态止损
                    if price <= sl_price:
                        info = load_market_info(symbol)
                        ct_val = float(info.get('ctVal', 0) or 0)
                        realized = pos['size'] * ct_val * (price - entry)
                        
                        if close_position_market(symbol, 'long', pos['size']):
                            stats['trades'] += 1
                            stats['losses'] += 1
                            stats['realized_pnl'] += realized
                            log.info(f'🛑 {symbol} 多头ATR止损 {realized:.2f}U')
                            notify_event('多头止损', f'{symbol} {realized:.2f}U')
                            continue
                    
                    # 固定止盈
                    if pnl_pct >= TP_PCT:
                        info = load_market_info(symbol)
                        ct_val = float(info.get('ctVal', 0) or 0)
                        realized = pos['size'] * ct_val * (price - entry)
                        
                        if close_position_market(symbol, 'long', pos['size']):
                            stats['trades'] += 1
                            stats['wins'] += 1
                            stats['realized_pnl'] += realized
                            log.info(f'✅ {symbol} 多头止盈 {pnl_pct*100:.1f}% {realized:.2f}U')
                            notify_event('多头止盈', f'{symbol} {realized:.2f}U')
                            continue
                
                if has_short:
                    entry = pos['entry']
                    pnl_pct = (entry - price) / entry
                    sl_price = entry + (SL_ATR_MULTIPLIER * curr_atr)
                    
                    # 动态止损
                    if price >= sl_price:
                        info = load_market_info(symbol)
                        ct_val = float(info.get('ctVal', 0) or 0)
                        realized = pos['size'] * ct_val * (entry - price)
                        
                        if close_position_market(symbol, 'short', pos['size']):
                            stats['trades'] += 1
                            stats['losses'] += 1
                            stats['realized_pnl'] += realized
                            log.info(f'🛑 {symbol} 空头ATR止损 {realized:.2f}U')
                            notify_event('空头止损', f'{symbol} {realized:.2f}U')
                            continue
                    
                    # 固定止盈
                    if pnl_pct >= TP_PCT:
                        info = load_market_info(symbol)
                        ct_val = float(info.get('ctVal', 0) or 0)
                        realized = pos['size'] * ct_val * (entry - price)
                        
                        if close_position_market(symbol, 'short', pos['size']):
                            stats['trades'] += 1
                            stats['wins'] += 1
                            stats['realized_pnl'] += realized
                            log.info(f'✅ {symbol} 空头止盈 {pnl_pct*100:.1f}% {realized:.2f}U')
                            notify_event('空头止盈', f'{symbol} {realized:.2f}U')
                            continue
                
                # ========== 交易信号 ==========
                # 反向信号先平仓
                if has_long and dead_cross:
                    info = load_market_info(symbol)
                    ct_val = float(info.get('ctVal', 0) or 0)
                    realized = pos['size'] * ct_val * (price - pos['entry'])
                    
                    if close_position_market(symbol, 'long', pos['size']):
                        stats['trades'] += 1
                        if realized > 0:
                            stats['wins'] += 1
                        else:
                            stats['losses'] += 1
                        stats['realized_pnl'] += realized
                        log.info(f'🔄 {symbol} MACD死叉平多 {realized:.2f}U')
                        notify_event('死叉平多', f'{symbol} {realized:.2f}U')
                        continue
                
                if has_short and golden_cross:
                    info = load_market_info(symbol)
                    ct_val = float(info.get('ctVal', 0) or 0)
                    realized = pos['size'] * ct_val * (pos['entry'] - price)
                    
                    if close_position_market(symbol, 'short', pos['size']):
                        stats['trades'] += 1
                        if realized > 0:
                            stats['wins'] += 1
                        else:
                            stats['losses'] += 1
                        stats['realized_pnl'] += realized
                        log.info(f'🔄 {symbol} MACD金叉平空 {realized:.2f}U')
                        notify_event('金叉平空', f'{symbol} {realized:.2f}U')
                        continue
                
                # 避免同一根K线重复操作
                cur_bar_ts = int(ohlcv[-1][0])
                if last_bar_ts.get(symbol) == cur_bar_ts:
                    continue
                
                # 开仓信号
                if golden_cross and not has_long and not has_short:
                    if place_market_order(symbol, 'long'):
                        log.info(f'🔥 {symbol} MACD金叉做多 @{price:.4f}')
                        last_bar_ts[symbol] = cur_bar_ts
                
                if dead_cross and not has_short and not has_long:
                    if place_market_order(symbol, 'short'):
                        log.info(f'🔥 {symbol} MACD死叉做空 @{price:.4f}')
                        last_bar_ts[symbol] = cur_bar_ts
                
            except Exception as e:
                log.warning(f'{symbol} 处理异常: {e}')
                continue
        
        time.sleep(SCAN_INTERVAL)
        
    except KeyboardInterrupt:
        log.info('用户中断，退出...')
        break
    except Exception as e:
        log.error(f'主循环异常: {e}')
        time.sleep(SCAN_INTERVAL)