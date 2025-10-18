#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — Adaptive MACD+RSI Futures Bot with Auto-Learning (L1)
Integrated features:
 - Market state recognition (ATR, Bollinger width, EMA slope)
 - Strategy switching (trend: MACD+RSI, range: BB revert, spike: momentum)
 - Dynamic position & leverage (ATR-based scaling)
 - TP/SL (fixed + ATR-based) and simple trailing suggestion
 - Emotion filter (volume spike + funding rate best-effort)
 - Cross-market alignment (best-effort)
 - AutoParameterLearner (L1): per-symbol online parameter adjustment (every 10 trades)
 - Detailed logging (info + debug)
 - Paper mode by default; real mode uses ccxt (okx) if available and env provided
Usage:
 - Install requirements: pip install ccxt pandas numpy
 - Configure env variables (see ENV section below)
 - Run: python3 main.py
"""

import os
import time
import math
import json
import logging
import traceback
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import random

import numpy as np
import pandas as pd

# optional ccxt for real trading
try:
    import ccxt
except Exception:
    ccxt = None

# ----------------- ENVIRONMENT & LOGGING -----------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("adaptive_bot")

# ENV variables (Railway)
OKX_API_KEY = os.getenv("OKX_API_KEY", "").strip()
OKX_SECRET = os.getenv("OKX_SECRET_KEY", os.getenv("OKX_SECRET", "")).strip()
OKX_PASSPHRASE = os.getenv("OKX_PASSPHRASE", "").strip()
RISK_PERCENT = float(os.getenv("RISK_PERCENT", "0.5"))
TIMEFRAME = os.getenv("TIMEFRAME", "15m")
TRADE_SYMBOLS = [s.strip() for s in os.getenv("TRADE_SYMBOLS", "WIF/USDT,PEPE/USDT,DOGE/USDT,ARB/USDT").split(",") if s.strip()]
RUN_MODE = os.getenv("RUN_MODE", "paper").lower()
MIN_PER_SYMBOL_USDT = float(os.getenv("MIN_PER_SYMBOL_USDT", "0.05"))
SIGNAL_THRESHOLD = int(os.getenv("SIGNAL_THRESHOLD", "65"))
SIM_BALANCE = float(os.getenv("SIM_BALANCE", "1000.0"))

logger.info(f"RUN_MODE={RUN_MODE} TIMEFRAME={TIMEFRAME} SYMBOLS={TRADE_SYMBOLS} RISK%={RISK_PERCENT} LOG_LEVEL={LOG_LEVEL}")

# ----------------- UTIL FUNCTIONS -----------------
def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(com=period - 1, min_periods=period).mean()
    ma_down = down.ewm(com=period - 1, min_periods=period).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window).mean()
    return df

def calc_bbands(df: pd.DataFrame, window: int = 20, k: float = 2.0) -> pd.DataFrame:
    df['bb_ma'] = df['close'].rolling(window).mean()
    df['bb_std'] = df['close'].rolling(window).std()
    df['bb_upper'] = df['bb_ma'] + k * df['bb_std']
    df['bb_lower'] = df['bb_ma'] - k * df['bb_std']
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_ma']
    return df

def calc_macd(df: pd.DataFrame, fast=8, slow=21, signal=9) -> pd.DataFrame:
    df['ema_fast'] = ema(df['close'], fast)
    df['ema_slow'] = ema(df['close'], slow)
    df['macd_diff'] = df['ema_fast'] - df['ema_slow']
    df['macd_dea'] = df['macd_diff'].ewm(span=signal, adjust=False).mean()
    df['macd_hist'] = df['macd_diff'] - df['macd_dea']
    return df

# ----------------- MACDStrategy CLASS -----------------
class MACDStrategy:
    def __init__(self):
        # Basic config
        self.timeframe = TIMEFRAME
        self.symbols = TRADE_SYMBOLS
        self.run_mode = RUN_MODE  # 'paper' or 'real'
        self.exchange = None
        self.other_exchange = None  # optional for external benchmark
        self.position_weights = {s: 1.0 for s in self.symbols}
        self.symbol_leverage = {s: 20 for s in self.symbols}

        # per-symbol params (defaults for 15m)
        self.macd_params = {
            "WIF/USDT": (8, 21, 9),
            "PEPE/USDT": (9, 23, 9),
            "DOGE/USDT": (9, 25, 9),
            "ARB/USDT": (10, 26, 9),
        }
        self.rsi_params = {
            "WIF/USDT": 9,
            "PEPE/USDT": 9,
            "DOGE/USDT": 10,
            "ARB/USDT": 10,
        }
        self.tp_sl_params = {
            "WIF/USDT": (1.0, 0.6),
            "PEPE/USDT": (1.0, 0.6),
            "DOGE/USDT": (1.0, 0.7),
            "ARB/USDT": (1.2, 0.8),
        }

        # thresholds and misc
        self.signal_threshold = SIGNAL_THRESHOLD
        self.position_percentage = float(os.getenv("POSITION_PERCENTAGE", "1.0"))
        self.atr_tp_m = float(os.getenv("ATR_TP_M", "1.5"))
        self.atr_sl_n = float(os.getenv("ATR_SL_N", "1.0"))

        # adaptive config tuned for 15m
        self._adaptive_cfg = {
            'bb_window': 20,
            'bb_k': 2.0,
            'atr_period': 14,
            'ema_fast_for_state': 10,
            'ema_slow_for_state': 50,
            'bb_width_range_thresh': 0.006,
            'bb_width_trend_thresh': 0.010,
            'atr_spike_ratio': 0.012,
            'vol_spike_multiplier': 2.5,
            'min_klines': 40,
            'max_position_per_symbol_pct': 0.25
        }

        # safety & paper-mode simulation storage
        self.min_per_symbol_usdt = MIN_PER_SYMBOL_USDT
        self.set_leverage_on_start = False
        self._paper_positions = {}  # symbol -> {'side','size','entry','notional'}
        self._paper_trade_history = []  # list of trade dicts for statistics

        # Auto-learn storage
        self.learned_file = "learned_params.json"
        self.learn_log = {s: [] for s in self.symbols}  # per-symbol list of 0/1 for last trades
        self._learn_trade_buffer = {s: [] for s in self.symbols}  # store last N trades details if needed

        # Initialize exchange if real
        if self.run_mode == 'real':
            if ccxt is None:
                logger.error("ccxt not installed: cannot run in real mode")
                raise RuntimeError("ccxt required for real mode")
            self.exchange = ccxt.okx({
                'apiKey': OKX_API_KEY,
                'secret': OKX_SECRET,
                'password': OKX_PASSPHRASE,
                'enableRateLimit': True,
            })
            logger.info("OKX exchange initialized for real mode")
        else:
            logger.info("Running in paper mode (no real orders)")

        # Load any learned params
        self._load_learned_params()

        # Useful stats
        self.stats = {'total_trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0}

        # safety leash: pause trading when global drawdown high (can be improved)
        self.max_dd_pct = float(os.getenv("MAX_DD_PCT", "0.30"))  # if account drawdown > this, pause trading (optional)

    # -------------------- KLINE / DATA FETCH --------------------
    def get_klines(self, symbol: str, limit: int = 200) -> pd.DataFrame:
        """
        Fetch klines. Use user's original method if present; else fallback to ccxt or simulated.
        Expected columns: ['timestamp','open','high','low','close','volume'].
        """
        # If user previously defined a method outside, it might be bound; prefer that.
        # But since we are in integrated file, implement fallback robustly.
        try:
            if self.run_mode == 'real' and self.exchange:
                # ccxt fetch_ohlcv
                try:
                    ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=self.timeframe, limit=limit)
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    return df
                except Exception as e:
                    logger.warning(f"fetch_ohlcv failed for {symbol}: {e}")
            # Paper or fallback: synthetic recent walk around base
            base_price = 1.0 + random.random() * 0.1
            n = max(limit, 120)
            rnd = np.random.normal(0, 0.001, size=n)
            close = np.cumsum(rnd) + base_price
            high = close + np.random.rand(n) * 0.002
            low = close - np.random.rand(n) * 0.002
            openp = np.concatenate(([close[0]], close[:-1]))
            volume = np.random.rand(n) * 1000
            df = pd.DataFrame({
                'timestamp': pd.date_range(end=pd.Timestamp.utcnow(), periods=n, freq='T'),
                'open': openp,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
            return df
        except Exception as e:
            logger.error(f"get_klines fallback error: {e}")
            return pd.DataFrame()

    # -------------------- INDICATORS --------------------
    def calculate_indicators(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        try:
            rsi_p = int(self.rsi_params.get(symbol, 9))
            df = calculate_rsi(df, rsi_p)
            df = calculate_atr(df, self._adaptive_cfg['atr_period'])
            macd_cfg = self.macd_params.get(symbol, (8, 21, 9))
            df = calc_macd(df, *macd_cfg)
            df = calc_bbands(df, window=self._adaptive_cfg['bb_window'], k=self._adaptive_cfg['bb_k'])
            df = df.dropna()
            return df
        except Exception as e:
            logger.warning(f"calculate_indicators {symbol} error: {e}")
            return df

    # -------------------- SIGNALS --------------------
    def check_long_signal(self, df: pd.DataFrame, symbol: str) -> float:
        try:
            fast, slow, sig = self.macd_params.get(symbol, (8,21,9))
            macd_line = df['macd_diff']
            signal_line = df['macd_dea']
            hist = df['macd_hist']
            rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50
            strength = 0.0
            if macd_line.iloc[-1] > signal_line.iloc[-1]:
                strength += 50.0
                slope = (hist.iloc[-1] - hist.iloc[-2])
                if slope > 0:
                    strength += min(30.0, slope * 100.0)
            if rsi < 30:
                strength += 5.0
            if rsi > 85:
                strength -= 20.0
            return max(0.0, min(100.0, float(strength)))
        except Exception as e:
            logger.debug(f"check_long_signal error {symbol}: {e}")
            return 0.0

    def check_short_signal(self, df: pd.DataFrame, symbol: str) -> float:
        try:
            macd_line = df['macd_diff']
            signal_line = df['macd_dea']
            hist = df['macd_hist']
            rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50
            strength = 0.0
            if macd_line.iloc[-1] < signal_line.iloc[-1]:
                strength += 50.0
                slope = (hist.iloc[-1] - hist.iloc[-2])
                if slope < 0:
                    strength += min(30.0, -slope * 100.0)
            if rsi > 70:
                strength += 5.0
            if rsi < 15:
                strength -= 20.0
            return max(0.0, min(100.0, float(strength)))
        except Exception as e:
            logger.debug(f"check_short_signal error {symbol}: {e}")
            return 0.0

    # -------------------- MARKET STATE RECOGNITION --------------------
    def determine_market_state(self, symbol: str, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        try:
            if df is None or df.empty:
                df = self.get_klines(symbol, limit=max(self._adaptive_cfg['min_klines'], 120))
            if df is None or df.empty or len(df) < self._adaptive_cfg['min_klines']:
                return {'state': 'unknown', 'metrics': {}}
            df = self.calculate_indicators(df, symbol)
            bb_width = float(df['bb_width'].iloc[-1] if 'bb_width' in df.columns else 0.0)
            atr = float(df['atr'].iloc[-1] if 'atr' in df.columns else 0.0)
            price = float(df['close'].iloc[-1])
            atr_ratio = atr / max(1e-12, price)
            ema_fast = float(df['close'].ewm(span=self._adaptive_cfg['ema_fast_for_state'], adjust=False).mean().iloc[-1])
            ema_slow = float(df['close'].ewm(span=self._adaptive_cfg['ema_slow_for_state'], adjust=False).mean().iloc[-1])
            ema_slope = (ema_fast - ema_slow) / max(1e-12, ema_slow)
            last_pct = (df['close'].iloc[-1] - df['close'].iloc[-2]) / max(1e-12, df['close'].iloc[-2])
            vol_ma = df['volume'].rolling(self._adaptive_cfg['bb_window']).mean().iloc[-1] if 'volume' in df.columns else 0.0
            vol_now = float(df['volume'].iloc[-1] if 'volume' in df.columns else 0.0)
            vol_spike = (vol_ma > 0 and vol_now / max(1e-12, vol_ma) >= self._adaptive_cfg['vol_spike_multiplier'])
            if abs(last_pct) >= self._adaptive_cfg['atr_spike_ratio'] and vol_spike:
                state = 'spike' if last_pct > 0 else 'crash'
            elif bb_width < self._adaptive_cfg['bb_width_range_thresh']:
                state = 'range'
            elif abs(ema_slope) >= 0.001 and bb_width >= self._adaptive_cfg['bb_width_trend_thresh']:
                state = 'trend'
            elif atr_ratio > (self._adaptive_cfg['atr_spike_ratio'] * 0.8):
                state = 'volatile'
            else:
                state = 'normal'
            metrics = {
                'bb_width': bb_width,
                'atr': atr,
                'atr_ratio': atr_ratio,
                'ema_slope': ema_slope,
                'last_pct': last_pct,
                'vol_now': vol_now,
                'vol_ma': vol_ma,
                'vol_spike': vol_spike
            }
            logger.debug(f"determine_market_state {symbol}: {state} metrics={metrics}")
            return {'state': state, 'metrics': metrics}
        except Exception as e:
            logger.warning(f"determine_market_state error {symbol}: {e}")
            return {'state': 'unknown', 'metrics': {}}

    # -------------------- STRATEGY SWITCHER --------------------
    def choose_strategy_for_state(self, symbol: str, market_state: str) -> str:
        base = getattr(self, 'strategy_by_symbol', {}).get(symbol, 'combo')
        if market_state == 'trend':
            return 'macd_rsi'
        if market_state == 'range':
            return 'bb_revert'
        if market_state in ('spike', 'crash', 'volatile'):
            return 'momentum'
        return base

    # -------------------- DYNAMIC POSITION & LEVERAGE --------------------
    def compute_dynamic_position_and_leverage(self, symbol: str, price: float, atr: float) -> Dict[str, Any]:
        try:
            balance = safe_float(self.get_account_balance())
            if balance <= 0:
                return {'notional': 0.0, 'leverage': int(self.symbol_leverage.get(symbol, 20))}
            atr_ratio = atr / max(1e-12, price)
            base_risk_usdt = max(0.05, balance * (RISK_PERCENT / 100.0))
            scale = min(3.0, max(0.3, 0.02 / max(1e-12, atr_ratio)))
            notional = base_risk_usdt * scale * float(self.position_weights.get(symbol, 1.0))
            cap = balance * float(self._adaptive_cfg.get('max_position_per_symbol_pct', 0.25))
            if notional > cap:
                notional = cap
            if notional < self.min_per_symbol_usdt and balance >= self.min_per_symbol_usdt:
                notional = self.min_per_symbol_usdt
            base_lev = int(self.symbol_leverage.get(symbol, 20))
            lev = max(3, min(50, int(base_lev * max(0.4, min(1.6, 0.12 / max(1e-12, atr_ratio)))) ))
            # Safety: reduce leverage if account drawdown large (basic heuristics; enhance as needed)
            # (Implementing drawdown calc would require equity history; placeholder for now)
            return {'notional': float(round(notional, 4)), 'leverage': int(lev)}
        except Exception as e:
            logger.warning(f"compute_dynamic_position_and_leverage error {symbol}: {e}")
            return {'notional': 0.0, 'leverage': int(self.symbol_leverage.get(symbol, 20))}

    # -------------------- EMOTION FILTER --------------------
    def apply_emotion_filter(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        try:
            vol_ma = df['volume'].rolling(self._adaptive_cfg['bb_window']).mean().iloc[-1] if 'volume' in df.columns else 0.0
            vol_now = float(df['volume'].iloc[-1]) if 'volume' in df.columns else 0.0
            if vol_ma > 0 and vol_now / max(1e-12, vol_ma) >= self._adaptive_cfg['vol_spike_multiplier']:
                logger.info(f"{symbol}: emotion filter - volume spike (now={vol_now:.1f} ma={vol_ma:.1f})")
                return {'pass': False, 'reason': 'volume_spike'}
            # funding rate check (best-effort; many ccxt wrappers differ)
            try:
                if self.run_mode == 'real' and self.exchange:
                    inst = self.symbol_to_inst_id(symbol)
                    fr = None
                    try:
                        fr = self._safe_call(self.exchange.publicGetMarketFundingRateHistory, {'instId': inst})
                    except Exception:
                        fr = None
                    if fr and isinstance(fr, dict):
                        data = fr.get('data') or []
                        if isinstance(data, list) and data:
                            rate = safe_float(data[0].get('fundingRate') or data[0].get('rate') or 0.0)
                            if abs(rate) >= 0.01:
                                logger.info(f"{symbol}: emotion filter - funding extreme {rate}")
                                return {'pass': False, 'reason': 'funding_extreme'}
            except Exception:
                pass
            return {'pass': True, 'reason': 'ok'}
        except Exception as e:
            logger.warning(f"apply_emotion_filter error {symbol}: {e}")
            return {'pass': True, 'reason': 'error'}

    # -------------------- MARKET BENCHMARK ALIGN --------------------
    def align_market_benchmark(self, symbol: str, price: float) -> Dict[str, Any]:
        try:
            okx_price = price
            other_price = None
            if hasattr(self, 'other_exchange') and self.other_exchange:
                try:
                    other_ticker = self.other_exchange.fetch_ticker(symbol.replace('/USDT', '/USDT'))
                    other_price = safe_float(other_ticker.get('last') or other_ticker.get('price') or 0.0)
                except Exception:
                    other_price = None
            if other_price:
                diff_pct = abs(okx_price - other_price) / max(1e-12, (okx_price + other_price) / 2.0)
                if diff_pct > 0.005:
                    logger.info(f"{symbol}: market align fail diff={diff_pct:.3%} okx={okx_price} other={other_price}")
                    return {'aligned': False, 'okx_price': okx_price, 'other_price': other_price, 'diff_pct': diff_pct}
                return {'aligned': True, 'okx_price': okx_price, 'other_price': other_price, 'diff_pct': diff_pct}
            return {'aligned': True, 'okx_price': okx_price, 'other_price': other_price, 'diff_pct': 0.0}
        except Exception as e:
            logger.warning(f"align_market_benchmark error {symbol}: {e}")
            return {'aligned': True, 'okx_price': price, 'other_price': None, 'diff_pct': 0.0}

    # -------------------- AUTO-LEARN MODULE --------------------
    def _load_learned_params(self):
        try:
            if os.path.exists(self.learned_file):
                with open(self.learned_file, 'r') as f:
                    data = json.load(f)
                for sym, cfg in data.items():
                    if 'macd' in cfg and sym in self.macd_params:
                        macd = tuple(cfg['macd'])
                        # enforce bounds
                        macd = tuple(max(5, min(30, int(x))) for x in macd)
                        self.macd_params[sym] = macd
                    if 'rsi' in cfg and sym in self.rsi_params:
                        rsi = int(cfg['rsi'])
                        rsi = max(5, min(25, rsi))
                        self.rsi_params[sym] = rsi
                logger.info(f"Loaded learned params from {self.learned_file}")
        except Exception as e:
            logger.warning(f"_load_learned_params failed: {e}")

    def _save_learned_params(self):
        try:
            data = {}
            for s in self.symbols:
                data[s] = {'macd': list(self.macd_params.get(s, (8,21,9))), 'rsi': int(self.rsi_params.get(s, 9))}
            with open(self.learned_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved learned params to {self.learned_file}")
        except Exception as e:
            logger.warning(f"_save_learned_params failed: {e}")

    def record_trade_result(self, symbol: str, pnl: float):
        try:
            # record 1 if profit>0 else 0
            arr = self.learn_log.get(symbol, [])
            arr.append(1 if pnl > 0 else 0)
            if len(arr) > 10:
                arr.pop(0)
            self.learn_log[symbol] = arr
            # store trade buffer for analysis
            buf = self._learn_trade_buffer.get(symbol, [])
            buf.append({'pnl': pnl, 'time': datetime.utcnow().isoformat()})
            if len(buf) > 50:
                buf.pop(0)
            self._learn_trade_buffer[symbol] = buf
            logger.info(f"AutoLearn record: {symbol} pnl={pnl:.4f} last10_winrate={sum(arr)/len(arr):.2f}")
            if len(arr) == 10:
                self._adjust_parameters(symbol)
        except Exception as e:
            logger.warning(f"record_trade_result error: {e}")

    def _adjust_parameters(self, symbol: str):
        try:
            arr = self.learn_log.get(symbol, [])
            if len(arr) < 10:
                return
            win_rate = sum(arr) / len(arr)
            macd = list(self.macd_params.get(symbol, (8,21,9)))
            rsi = int(self.rsi_params.get(symbol, 9))
            # adaptive magnitude scales with distance from 0.5
            if win_rate < 0.45:
                # degrade to more sensitive: reduce periods more when much lower than 0.45
                delta = int(max(1, min(3, round((0.5 - win_rate) * 10))))
                macd = [max(5, m - delta) for m in macd]
                rsi = max(5, rsi - 1)
                change = "more_sensitive"
            elif win_rate > 0.65:
                delta = int(max(1, min(3, round((win_rate - 0.6) * 10))))
                macd = [min(30, m + delta) for m in macd]
                rsi = min(25, rsi + 1)
                change = "less_sensitive"
            else:
                change = "stable"
            # enforce bounds
            macd = [max(5, min(30, int(x))) for x in macd]
            rsi = max(5, min(25, int(rsi)))
            self.macd_params[symbol] = tuple(macd)
            self.rsi_params[symbol] = rsi
            logger.info(f"🤖 AutoLearn [{symbol}] win_rate={win_rate:.2f} -> MACD={self.macd_params[symbol]} RSI={self.rsi_params[symbol]} change={change}")
            self._save_learned_params()
        except Exception as e:
            logger.warning(f"_adjust_parameters error: {e}")

    # -------------------- ADAPTIVE TICK (MAIN INTEGRATION) --------------------
    def adaptive_tick(self, symbol: str):
        try:
            df = self.get_klines(symbol, limit=max(self._adaptive_cfg['min_klines'], 120))
            if df is None or df.empty or len(df) < self._adaptive_cfg['min_klines']:
                logger.debug(f"{symbol}: klines insufficient for adaptive tick")
                return
            df = self.calculate_indicators(df, symbol)
            ms = self.determine_market_state(symbol, df)
            state = ms.get('state', 'unknown')
            metrics = ms.get('metrics', {})
            strat = self.choose_strategy_for_state(symbol, state)
            logger.info(f"[{symbol}] state={state} strat={strat} metrics={metrics}")

            # Apply emotion filter
            emo = self.apply_emotion_filter(symbol, df)
            if not emo.get('pass', True):
                logger.info(f"[{symbol}] blocked by emotion filter: {emo.get('reason')}")
                return

            # Price alignment
            price_now = float(df['close'].iloc[-1])
            align = self.align_market_benchmark(symbol, price_now)
            if not align.get('aligned', True):
                logger.info(f"[{symbol}] market alignment failed diff={align.get('diff_pct'):.3%}")
                return

            # Dynamic size & leverage
            atr_val = float(metrics.get('atr') or df['atr'].iloc[-1])
            posinfo = self.compute_dynamic_position_and_leverage(symbol, price_now, atr_val)
            notional = float(posinfo.get('notional', 0.0))
            lev = int(posinfo.get('leverage', int(self.symbol_leverage.get(symbol, 20))))
            logger.debug(f"[{symbol}] dynamic pos={notional}U lev={lev} atr={atr_val:.6f}")

            # Get signals
            long_s = self.check_long_signal(df, symbol)
            short_s = self.check_short_signal(df, symbol)
            signal = 'hold'
            if long_s >= self.signal_threshold and long_s > short_s:
                signal = 'buy'
            elif short_s >= self.signal_threshold and short_s > long_s:
                signal = 'sell'
            else:
                # fallback simple checks depending on strat
                if strat == 'macd_rsi':
                    if df['macd_diff'].iloc[-1] > df['macd_dea'].iloc[-1] and df['rsi'].iloc[-1] > 50:
                        signal = 'buy'
                    elif df['macd_diff'].iloc[-1] < df['macd_dea'].iloc[-1] and df['rsi'].iloc[-1] < 50:
                        signal = 'sell'
                elif strat == 'bb_revert':
                    if df['close'].iloc[-1] < df['bb_lower'].iloc[-1] and df['rsi'].iloc[-1] < 40:
                        signal = 'buy'
                    elif df['close'].iloc[-1] > df['bb_upper'].iloc[-1] and df['rsi'].iloc[-1] > 60:
                        signal = 'sell'
                elif strat == 'momentum':
                    recent_high = df['high'].iloc[-6:-1].max()
                    recent_low = df['low'].iloc[-6:-1].min()
                    if df['close'].iloc[-1] > recent_high:
                        signal = 'buy'
                    elif df['close'].iloc[-1] < recent_low:
                        signal = 'sell'

            # simple existing position check
            pos = self.get_position(symbol, force_refresh=False)
            if pos and pos.get('size', 0) > 0:
                side = pos.get('side')
                if (side == 'long' and signal == 'buy') or (side == 'short' and signal == 'sell'):
                    logger.debug(f"[{symbol}] existing same-side position; managing only")
                    if hasattr(self, 'manage_positions'):
                        try:
                            self.manage_positions(symbol, df)
                        except Exception as e:
                            logger.debug(f"manage_positions error: {e}")
                    return
                elif signal in ('buy', 'sell') and side in ('long','short'):
                    logger.info(f"[{symbol}] reverse detected -> close existing then wait")
                    try:
                        if hasattr(self, 'close_position'):
                            self.close_position(symbol)
                        else:
                            self.cancel_all_orders(symbol)
                    except Exception as e:
                        logger.warning(f"close existing failed: {e}")
                    return

            # Execute order if signal and notional > 0
            if signal in ('buy','sell') and notional > 0:
                try:
                    # set leverage best-effort
                    if self.run_mode == 'real' and self.set_leverage_on_start and self.exchange:
                        try:
                            inst_id = self.symbol_to_inst_id(symbol)
                            self._safe_call(self.exchange.privatePostAccountSetLeverage, {'instId': inst_id, 'lever': str(lev), 'mgnMode': 'cross'})
                        except Exception:
                            pass
                except Exception:
                    pass

                try:
                    ok = self.create_order(symbol, signal, notional)
                    if ok:
                        logger.info(f"[{symbol}] ORDER placed {signal} notional={notional:.4f}U lev={lev}")
                        # place TP/SL using ATR multipliers (or existing method)
                        try:
                            entry_price = price_now
                            tp_mult, sl_mult = self.tp_sl_params.get(symbol, (1.0, 0.6))
                            # tp_sl are percents in this mapping; convert to prices
                            if signal == 'buy':
                                tp = entry_price * (1 + tp_mult/100.0)
                                sl = entry_price * (1 - sl_mult/100.0)
                            else:
                                tp = entry_price * (1 - tp_mult/100.0)
                                sl = entry_price * (1 + sl_mult/100.0)
                            if hasattr(self, 'place_okx_tp_sl'):
                                self.place_okx_tp_sl(symbol, entry_price, signal, atr_val)
                            else:
                                logger.info(f"[{symbol}] Suggest TP={tp:.6f} SL={sl:.6f}")
                        except Exception as e:
                            logger.debug(f"post-order TP/SL error: {e}")
                    else:
                        logger.warning(f"[{symbol}] create_order returned falsy")
                except Exception as e:
                    logger.error(f"[{symbol}] create_order exception: {e}", exc_info=True)
            else:
                logger.debug(f"[{symbol}] no action (signal={signal} notional={notional if 'notional' in locals() else 0})")
        except Exception as e:
            logger.error(f"adaptive_tick unhandled error for {symbol}: {e}", exc_info=True)

    # -------------------- ORDER / POSITION HELPERS (Paper Simulation + Real hooks) --------------------
    def create_order(self, symbol: str, side: str, notional_usdt: float) -> bool:
        """
        Wrapper for order creation.
        In paper mode: simulate and store in self._paper_positions.
        In real mode: try best-effort ccxt market order (user should adapt to their API implementation).
        """
        try:
            logger.debug(f"create_order called {symbol} {side} notional={notional_usdt}")
            if self.run_mode != 'real':
                price = self.get_fake_price(symbol)
                amount = notional_usdt / max(1e-12, price)
                # Simulate storing a position (full notional used)
                self._paper_positions[symbol] = {
                    'side': 'long' if side == 'buy' else 'short',
                    'size': amount,
                    'entry_price': price,
                    'notional': notional_usdt,
                    'timestamp': datetime.utcnow().isoformat()
                }
                logger.info(f"[PAPER] Simulated {side.upper()} {symbol} entry={price:.6f} size={amount:.6f} notional={notional_usdt:.4f}")
                return True
            # real mode (best-effort)
            if not self.exchange:
                logger.error("Exchange not configured")
                return False
            ticker = self._safe_call(self.exchange.fetch_ticker, symbol)
            price = safe_float(ticker.get('last') or 0.0)
            if price <= 0:
                logger.error(f"Invalid price for {symbol}")
                return False
            amount = notional_usdt / price
            try:
                order = self.exchange.create_market_order(symbol, 'buy' if side == 'buy' else 'sell', amount)
                logger.info(f"Real order response: {order}")
                return True
            except Exception as e:
                logger.error(f"Real create market order failed: {e}", exc_info=True)
                return False
        except Exception as e:
            logger.error(f"create_order wrapper error: {e}", exc_info=True)
            return False

    def get_fake_price(self, symbol: str) -> float:
        # Helper to retrieve synthetic price for paper mode from last klines
        try:
            df = self.get_klines(symbol, limit=5)
            if df is None or df.empty:
                return 1.0
            return float(df['close'].iloc[-1])
        except Exception:
            return 1.0

    def close_position(self, symbol: str) -> bool:
        """
        Close any existing position.
        In paper mode, compute PnL and call record_trade_result.
        """
        try:
            if self.run_mode != 'real':
                pos = self._paper_positions.get(symbol)
                if not pos:
                    logger.debug(f"[PAPER] no position to close for {symbol}")
                    return True
                exit_price = self.get_fake_price(symbol)
                entry = pos.get('entry_price', 0.0)
                notional = pos.get('notional', 0.0)
                side = pos.get('side')
                # Simple PnL: for long: (exit-entry)/entry * notional
                pnl = 0.0
                if side == 'long':
                    pnl = (exit_price - entry) / max(1e-12, entry) * notional
                else:
                    pnl = (entry - exit_price) / max(1e-12, entry) * notional
                logger.info(f"[PAPER] Closing {symbol} {side} entry={entry:.6f} exit={exit_price:.6f} pnl={pnl:.4f}")
                # record for stats
                self._paper_trade_history.append({'symbol': symbol, 'side': side, 'entry': entry, 'exit': exit_price, 'pnl': pnl, 'time': datetime.utcnow().isoformat()})
                # record for learning
                try:
                    self.record_trade_result(symbol, pnl)
                except Exception as e:
                    logger.warning(f"record_trade_result failed during close: {e}")
                # remove position
                del self._paper_positions[symbol]
                # update stats
                self.stats['total_trades'] += 1
                if pnl > 0:
                    self.stats['wins'] += 1
                else:
                    self.stats['losses'] += 1
                self.stats['pnl'] += pnl
                return True
            # real mode: user should implement close via exchange
            try:
                # best-effort: cancel orders and leave manual
                self.cancel_all_orders(symbol)
                logger.info(f"Requested close on real position for {symbol} (implement API call per exchange)")
                return True
            except Exception as e:
                logger.error(f"close_position real error: {e}", exc_info=True)
                return False
        except Exception as e:
            logger.error(f"close_position wrapper error: {e}", exc_info=True)
            return False

    def cancel_all_orders(self, symbol: str) -> bool:
        try:
            if self.run_mode != 'real' or not self.exchange:
                logger.debug(f"[PAPER] cancel_all_orders {symbol}")
                return True
            try:
                self._safe_call(self.exchange.cancel_all_orders, symbol)
                return True
            except Exception as e:
                logger.warning(f"cancel_all_orders failed: {e}")
                return False
        except Exception as e:
            logger.debug(f"cancel_all_orders error: {e}")
            return False

    # -------------------- SAFE CALL WRAPPER --------------------
    def _safe_call(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            logger.debug(f"_safe_call exception: {e}")
            return None

    def symbol_to_inst_id(self, symbol: str) -> str:
        return symbol.replace('/', '-') + '-SWAP'

    # -------------------- RUN LOOP --------------------
    def run(self, loop_delay: float = 1.0):
        logger.info("Starting adaptive bot main loop")
        try:
            while True:
                for symbol in self.symbols:
                    try:
                        self.adaptive_tick(symbol)
                    except Exception as e:
                        logger.error(f"adaptive_tick error for {symbol}: {e}", exc_info=True)
                    time.sleep(loop_delay)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt - stopping")
        except Exception as e:
            logger.error(f"Run loop unhandled exception: {e}", exc_info=True)

# ----------------- ENTRYPOINT -----------------
if __name__ == "__main__":
    strat = MACDStrategy()
    # Optional: attach other exchange for benchmark comparisons
    if ccxt:
        try:
            # Example: attach Binance for cross-check if keys available (not required)
            # strat.other_exchange = ccxt.binance({'enableRateLimit': True})
            pass
        except Exception:
            pass
    # Quick health check logging
    try:
        bal = strat.get_account_balance() if hasattr(strat, 'get_account_balance') else SIM_BALANCE
        logger.info(f"Initial account balance (sim/real): {bal}")
    except Exception:
        pass
    # Run (use a small loop delay; the strategy uses 15m K lines but polls frequently to detect new bars)
    strat.run(loop_delay=float(os.getenv("SYMBOL_LOOP_DELAY", "0.3")))
