#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MACD+RSI策略实现 - RAILWAY平台版本
扩展到11个币种,包含BTC/ETH/SOL/DOGE/XRP/PEPE/ARB
25倍杠杆,无限制交易,带挂单识别和状态同步
增加胜率统计和盈亏显示
进一步优化版:增强模块化(1)、性能(2)、错误处理(3)、日志(5)、其他(9);TP/SL&BB验证无问题,但添加更多日志和dry-run模拟
新增:布林带开口过滤(>0.8*mean保留信号) 与 动态止盈调节(趋势强时放宽TP距离)
修复:检测posMode并调整posSide参数,避免one-way模式错误
集成RSI.txt中的优化MACD+RSI策略和参数

🔧 修复内容:
1. TP/SL触发后增加兜底平仓逻辑
2. 追踪止损可条件性更新到交易所
3. 开仓后立即设置TP/SL保护
"""

import time
import logging
import datetime
import os
import json
from typing import Dict, Any, List, Optional, Literal
import pytz

import ccxt
import pandas as pd
import numpy as np
import math
import traceback
import random
import re

# 配置日志 - 使用中国时区和UTF-8编码 
class ChinaTimeFormatter(logging.Formatter):
    """中国时区的日志格式化器"""
    def formatTime(self, record, datefmt=None):
        dt = datetime.datetime.fromtimestamp(record.created, tz=pytz.timezone('Asia/Shanghai'))
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            s = dt.strftime('%Y-%m-%d %H:%M:%S')
        return s

# 配置日志 - 确保RAILWAY平台兼容
handler = logging.StreamHandler()
log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
handler.setLevel(getattr(logging, log_level, logging.INFO))
formatter = ChinaTimeFormatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, log_level, logging.INFO))
logger.addHandler(handler)
logger.propagate = False

# 手动RSI计算函数
def calculate_rsi(df, window):
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ema_up = up.ewm(com=window-1, min_periods=window).mean()
    ema_down = down.ewm(com=window-1, min_periods=window).mean()
    rs = ema_up / ema_down
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

# 手动ATR计算函数
def calculate_atr(df, window):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['atr'] = true_range.rolling(window).mean()
    return df

class TradingStats:
    """交易统计类"""
    def __init__(self, stats_file: str = 'trading_stats.json'):
        self.stats_file = stats_file
        self.stats = {
            'total_trades': 0,
            'win_trades': 0,
            'loss_trades': 0,
            'total_pnl': 0.0,
            'total_win_pnl': 0.0,
            'total_loss_pnl': 0.0,
            'trades_history': []
        }
        self.load_stats()
    
    def load_stats(self):
        """加载统计数据"""
        try:
            if os.path.exists(self.stats_file):
                with open(self.stats_file, 'r') as f:
                    self.stats = json.load(f)
                logger.info(f"✅ 加载历史统计数据:总交易{self.stats['total_trades']}笔")
        except Exception as e:
            logger.warning(f"⚠️ 加载统计数据失败: {str(e)} - {traceback.format_exc()},使用新数据")
    
    def save_stats(self):
        """保存统计数据"""
        try:
            with open(self.stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            logger.error(f"❌ 保存统计数据失败: {str(e)} - {traceback.format_exc()}")
    
    def add_trade(self, symbol: str, side: str, pnl: float):
        """添加交易记录"""
        self.stats['total_trades'] += 1
        self.stats['total_pnl'] += pnl
        
        if pnl > 0:
            self.stats['win_trades'] += 1
            self.stats['total_win_pnl'] += pnl
        else:
            self.stats['loss_trades'] += 1
            self.stats['total_loss_pnl'] += pnl
        
        # 添加交易历史
        china_tz = pytz.timezone('Asia/Shanghai')
        trade_record = {
            'timestamp': datetime.datetime.now(china_tz).strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': symbol,
            'side': side,
            'pnl': round(pnl, 4)
        }
        self.stats['trades_history'].append(trade_record)
        
        if len(self.stats['trades_history']) > 100:
            self.stats['trades_history'] = self.stats['trades_history'][-100:]
        
        self.save_stats()
    
    def get_win_rate(self) -> float:
        """计算胜率"""
        if self.stats['total_trades'] == 0:
            return 0.0
        return (self.stats['win_trades'] / self.stats['total_trades']) * 100
    
    def get_summary(self) -> str:
        """获取统计摘要"""
        win_rate = self.get_win_rate()
        return (f"📊 交易统计: 总计{self.stats['total_trades']}笔 | "
                f"胜{self.stats['win_trades']}笔 负{self.stats['loss_trades']}笔 | "
                f"胜率{win_rate:.1f}% | "
                f"总盈亏{self.stats['total_pnl']:.2f}U | "
                f"盈利{self.stats['total_win_pnl']:.2f}U 亏损{self.stats['total_loss_pnl']:.2f}U")

class MACDStrategy:
    """MACD+RSI策略类 - 扩展到11个币种"""
    PER_SYMBOL_OVERRIDES: Dict[str, Dict[str, object]] = {
        'WIF/USDT:USDT': {
            'TRAIL_ACTIVATE_PCT': 0.05,
            'trail_pct': 0.012,
            'INITIAL_SL_FLOOR_PCT': 0.02,
            'INITIAL_TP_TARGET_PCT': 0.12,
            'PARTIAL_TP_TIERS': '',
        },
        'ARB/USDT:USDT': {
            'TRAIL_ACTIVATE_PCT': 0.05,
            'trail_pct': 0.012,
            'INITIAL_SL_FLOOR_PCT': 0.02,
            'INITIAL_TP_TARGET_PCT': 0.12,
            'PARTIAL_TP_TIERS': '',
        },
    }

    def get_sym_cfg(self, symbol: str, key: str, default):
        try:
            return self.PER_SYMBOL_OVERRIDES.get(symbol, {}).get(key, default)
        except Exception:
            return default

    def __init__(self, api_key: str, secret_key: str, passphrase: str):
        """初始化策略"""
        self._sar_cache: Dict[tuple, float] = {}
        self._klines_cache: Dict[str, Dict[float, List[Dict]]] = {}
        self._klines_ttl = 60

        # 交易所配置
        self.exchange = ccxt.okx({
            'apiKey': api_key,
            'secret': secret_key,
            'password': passphrase,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap',
                'types': ['swap'],
            }
        })
        
        self.okx_params = {'instType': 'SWAP'}

        def _symbol_to_inst_id(sym: str) -> str:
            try:
                base = sym.split('/')[0]
                return f"{base}-USDT-SWAP"
            except Exception:
                return ''
        self.symbol_to_inst_id = _symbol_to_inst_id
        
        # 交易对配置
        self.symbols = [
            'FIL/USDT:USDT',
            'ZRO/USDT:USDT',
            'WIF/USDT:USDT',
            'WLD/USDT:USDT',
            'BTC/USDT:USDT',
            'ETH/USDT:USDT',
            'SOL/USDT:USDT',
            'DOGE/USDT:USDT',
            'XRP/USDT:USDT',
            'PEPE/USDT:USDT',
            'ARB/USDT:USDT'
        ]
        
        self.timeframe = '15m'
        self.timeframe_map = {
            'BTC/USDT:USDT': '15m',
            'ETH/USDT:USDT': '15m',
            'FIL/USDT:USDT': '15m',
            'WLD/USDT:USDT': '15m',
            'SOL/USDT:USDT': '15m',
            'WIF/USDT:USDT': '5m',
            'ZRO/USDT:USDT': '15m',
            'ARB/USDT:USDT': '5m',
            'PEPE/USDT:USDT': '5m',
            'DOGE/USDT:USDT': '5m',
            'XRP/USDT:USDT': '15m',
        }
        
        self.coin_categories = {
            'blue_chip': ['BTC/USDT:USDT', 'ETH/USDT:USDT'],
            'mainnet': ['SOL/USDT:USDT', 'XRP/USDT:USDT', 'ARB/USDT:USDT'],
            'infrastructure': ['FIL/USDT:USDT'],
            'emerging': ['ZRO/USDT:USDT', 'WLD/USDT:USDT'],
            'meme': ['DOGE/USDT:USDT', 'WIF/USDT:USDT', 'PEPE/USDT:USDT']
        }
        
        self.macd_params = {
            'BTC/USDT:USDT': {'fast': 8, 'slow': 17, 'signal': 9},
            'ETH/USDT:USDT': {'fast': 8, 'slow': 17, 'signal': 9},
            'SOL/USDT:USDT': {'fast': 6, 'slow': 16, 'signal': 9},
            'XRP/USDT:USDT': {'fast': 7, 'slow': 17, 'signal': 9},
            'ARB/USDT:USDT': {'fast': 6, 'slow': 15, 'signal': 9},
            'FIL/USDT:USDT': {'fast': 6, 'slow': 16, 'signal': 9},
            'ZRO/USDT:USDT': {'fast': 6, 'slow': 16, 'signal': 9},
            'WLD/USDT:USDT': {'fast': 5, 'slow': 13, 'signal': 9},
            'DOGE/USDT:USDT': {'fast': 5, 'slow': 12, 'signal': 8},
            'WIF/USDT:USDT': {'fast': 6, 'slow': 16, 'signal': 9},
            'PEPE/USDT:USDT': {'fast': 4, 'slow': 11, 'signal': 8}
        }
        
        self.rsi_params = {
            'BTC/USDT:USDT': 14,
            'ETH/USDT:USDT': 14,
            'SOL/USDT:USDT': 11,
            'XRP/USDT:USDT': 12,
            'ARB/USDT:USDT': 11,
            'FIL/USDT:USDT': 9,
            'ZRO/USDT:USDT': 14,
            'WLD/USDT:USDT': 9,
            'DOGE/USDT:USDT': 7,
            'WIF/USDT:USDT': 7,
            'PEPE/USDT:USDT': 6
        }
        
        self.rsi_thresholds = {
            'BTC/USDT:USDT': {'overbought': 70, 'oversold': 30},
            'ETH/USDT:USDT': {'overbought': 70, 'oversold': 30},
            'SOL/USDT:USDT': {'overbought': 72, 'oversold': 28},
            'XRP/USDT:USDT': {'overbought': 70, 'oversold': 30},
            'ARB/USDT:USDT': {'overbought': 72, 'oversold': 28},
            'FIL/USDT:USDT': {'overbought': 73, 'oversold': 27},
            'ZRO/USDT:USDT': {'overbought': 75, 'oversold': 25},
            'WLD/USDT:USDT': {'overbought': 75, 'oversold': 25},
            'DOGE/USDT:USDT': {'overbought': 78, 'oversold': 22},
            'WIF/USDT:USDT': {'overbought': 78, 'oversold': 22},
            'PEPE/USDT:USDT': {'overbought': 80, 'oversold': 20}
        }
        
        self.strategy_mode_map = {
            'FIL/USDT:USDT': 'combo',
            'ZRO/USDT:USDT': 'zero_cross',
            'WLD/USDT:USDT': 'divergence',
            'WIF/USDT:USDT': 'golden_cross',
            'BTC/USDT:USDT': 'combo',
            'ETH/USDT:USDT': 'combo',
            'SOL/USDT:USDT': 'combo',
            'DOGE/USDT:USDT': 'combo',
            'XRP/USDT:USDT': 'combo',
            'PEPE/USDT:USDT': 'combo',
            'ARB/USDT:USDT': 'combo',
        }
        
        self.stop_loss = {
            'BTC/USDT:USDT': 2.0,
            'ETH/USDT:USDT': 2.0,
            'SOL/USDT:USDT': 2.5,
            'XRP/USDT:USDT': 2.3,
            'ARB/USDT:USDT': 2.5,
            'FIL/USDT:USDT': 2.8,
            'ZRO/USDT:USDT': 3.0,
            'WLD/USDT:USDT': 3.5,
            'DOGE/USDT:USDT': 3.5,
            'WIF/USDT:USDT': 4.0,
            'PEPE/USDT:USDT': 4.5
        }
        
        self.take_profit = {
            'BTC/USDT:USDT': [1.2, 2.5, 4.0],
            'ETH/USDT:USDT': [1.2, 2.5, 4.0],
            'SOL/USDT:USDT': [1.5, 3.5, 5.5],
            'XRP/USDT:USDT': [1.3, 3.0, 5.0],
            'ARB/USDT:USDT': [1.5, 3.5, 5.0],
            'FIL/USDT:USDT': [1.5, 3.5, 5.5],
            'ZRO/USDT:USDT': [2.0, 4.0, 6.5],
            'WLD/USDT:USDT': [2.0, 4.5, 7.0],
            'DOGE/USDT:USDT': [2.5, 5.0, 8.0],
            'WIF/USDT:USDT': [2.5, 5.5, 9.0],
            'PEPE/USDT:USDT': [3.0, 6.0, 10.0]
        }
        
        self.position_weights = {
            'BTC/USDT:USDT': 1.2,
            'ETH/USDT:USDT': 1.2,
            'SOL/USDT:USDT': 1.0,
            'XRP/USDT:USDT': 1.0,
            'ARB/USDT:USDT': 0.9,
            'FIL/USDT:USDT': 0.9,
            'ZRO/USDT:USDT': 0.8,
            'WLD/USDT:USDT': 0.7,
            'DOGE/USDT:USDT': 0.6,
            'WIF/USDT:USDT': 0.5,
            'PEPE/USDT:USDT': 0.4
        }
        
        self.positions = {}
        self.strategy_mode = 'combo'
        
        self.trade_stats = {symbol: {'wins': 0, 'losses': 0, 'total_pnl': 0} for symbol in self.symbols}
        self.learning_state: Dict[str, Dict[str, Any]] = {
            s: {
                'recent_outcomes': [],
                'recent_pnls': [],
                'risk_multiplier': 1.0,
                'rsi_overbought_delta': 0.0,
                'rsi_oversold_delta': 0.0,
                'range_threshold_delta': 0.0,
                'trend_threshold_delta': 0.0,
                'atr_n_delta': 0.0,
                'atr_m_delta': 0.0
            } for s in self.symbols
        }
        
        self.symbol_leverage: Dict[str, int] = {
            'FIL/USDT:USDT': 25,
            'WIF/USDT:USDT': 20,
            'WLD/USDT:USDT': 25,
            'ZRO/USDT:USDT': 20,
            'BTC/USDT:USDT': 30,
            'ETH/USDT:USDT': 30,
            'SOL/USDT:USDT': 25,
            'XRP/USDT:USDT': 25,
            'DOGE/USDT:USDT': 20,
            'PEPE/USDT:USDT': 15,
            'ARB/USDT:USDT': 25,
        }
        
        self.strategy_params: Dict[str, Dict[str, Any]] = {
            'BTC/USDT:USDT': {'strategy': 'macd_sar', 'bb_period': 20, 'bb_k': 2.0, 'sar_af_start': 0.02, 'sar_af_max': 0.20},
            'ETH/USDT:USDT': {'strategy': 'macd_sar', 'bb_period': 20, 'bb_k': 2.0, 'sar_af_start': 0.02, 'sar_af_max': 0.20},
            'SOL/USDT:USDT': {'strategy': 'macd_sar', 'bb_period': 20, 'bb_k': 2.0, 'sar_af_start': 0.02, 'sar_af_max': 0.20},
            'WIF/USDT:USDT': {'strategy': 'bb_sar', 'bb_period': 20, 'bb_k': 2.5, 'sar_af_start': 0.01, 'sar_af_max': 0.10},
            'PEPE/USDT:USDT': {'strategy': 'bb_sar', 'bb_period': 20, 'bb_k': 2.5, 'sar_af_start': 0.01, 'sar_af_max': 0.10},
            'DOGE/USDT:USDT': {'strategy': 'bb_sar', 'bb_period': 20, 'bb_k': 2.5, 'sar_af_start': 0.01, 'sar_af_max': 0.10},
            'ZRO/USDT:USDT': {'strategy': 'hybrid', 'bb_period': 20, 'bb_k': 2.2, 'sar_af_start': 0.03, 'sar_af_max': 0.25},
            'WLD/USDT:USDT': {'strategy': 'hybrid', 'bb_period': 20, 'bb_k': 2.2, 'sar_af_start': 0.03, 'sar_af_max': 0.25},
            'FIL/USDT:USDT': {'strategy': 'hybrid', 'bb_period': 20, 'bb_k': 2.2, 'sar_af_start': 0.03, 'sar_af_max': 0.25},
            'XRP/USDT:USDT': {'strategy': 'bb_sar', 'bb_period': 20, 'bb_k': 1.8, 'sar_af_start': 0.02, 'sar_af_max': 0.15},
            'ARB/USDT:USDT': {'strategy': 'bb_sar', 'bb_period': 20, 'bb_k': 1.8, 'sar_af_start': 0.02, 'sar_af_max': 0.15},
        }
        
        self.per_symbol_params: Dict[str, Dict[str, Any]] = {
            'FIL/USDT:USDT': {
                'macd': (8, 21, 9), 'atr_period': 16, 'adx_period': 12,
                'adx_min_trend': 26, 'sl_n': 2.0, 'tp_m': 3.5, 'allow_reverse': True
            },
            'ZRO/USDT:USDT': {
                'macd': (9, 26, 12), 'atr_period': 14, 'adx_period': 10,
                'adx_min_trend': 30, 'sl_n': 2.2, 'tp_m': 3.0, 'allow_reverse': True
            },
            'WIF/USDT:USDT': {
                'macd': (5, 13, 9), 'atr_period': 18, 'adx_period': 10,
                'adx_min_trend': 24, 'sl_n': 2.1, 'tp_m': 4.5, 'tp_pct': 0.012, 'allow_reverse': True
            },
            'WLD/USDT:USDT': {
                'macd': (8, 21, 9), 'atr_period': 16, 'adx_period': 12,
                'adx_min_trend': 26, 'sl_n': 2.0, 'tp_m': 3.5, 'allow_reverse': True
            },
            'BTC/USDT:USDT': {
                'macd': (8, 21, 9), 'atr_period': 22, 'adx_period': 14,
                'adx_min_trend': 26, 'sl_n': 1.5, 'tp_m': 3.0, 'allow_reverse': True
            },
            'ETH/USDT:USDT': {
                'macd': (8, 21, 9), 'atr_period': 20, 'adx_period': 14,
                'adx_min_trend': 26, 'sl_n': 1.8, 'tp_m': 3.5, 'allow_reverse': True
            },
            'SOL/USDT:USDT': {
                'macd': (8, 21, 9), 'atr_period': 16, 'adx_period': 12,
                'adx_min_trend': 30, 'sl_n': 1.8, 'tp_m': 4.0, 'tp_pct': 0.012, 'allow_reverse': True
            },
            'XRP/USDT:USDT': {
                'macd': (8, 21, 9), 'atr_period': 18, 'adx_period': 14,
                'adx_min_trend': 26, 'sl_n': 1.8, 'tp_m': 3.5, 'allow_reverse': True
            },
            'DOGE/USDT:USDT': {
                'macd': (5, 13, 9), 'atr_period': 18, 'adx_period': 12,
                'adx_min_trend': 24, 'sl_n': 2.7, 'tp_m': 5.5, 'allow_reverse': True
            },
            'PEPE/USDT:USDT': {
                'macd': (5, 13, 9), 'atr_period': 18, 'adx_period': 10,
                'adx_min_trend': 24, 'sl_n': 3.2, 'tp_m': 6.5, 'allow_reverse': True
            },
            'ARB/USDT:USDT': {
                'macd': (6, 18, 9), 'atr_period': 18, 'adx_period': 12,
                'adx_min_trend': 24, 'sl_n': 2.4, 'tp_m': 4.3, 'allow_reverse': True
            }
        }
        
        self.position_percentage = 1.0
        
        self.positions_cache: Dict[str, Dict[str, Any]] = {}
        self.open_orders_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.last_sync_time: float = 0
        self.sync_interval: int = 60
        self.key_levels_cache: Dict[str, Dict[str, Any]] = {}
        
        self.markets_info: Dict[str, Dict[str, Any]] = {}
        
        self._last_api_ts: float = 0.0
        self._min_api_interval: float = 0.2

        self.symbol_loop_delay = 0.3
        try:
            self.risk_percent = float((os.environ.get('RISK_PERCENT') or '1.0').strip())
        except Exception:
            self.risk_percent = 1.0
        self.set_leverage_on_start = False
        
        self.stats = TradingStats()
        self.prev_positions: Dict[str, Dict[str, Any]] = {}

        self.strategy_by_symbol: Dict[str, str] = {
            'BTC/USDT:USDT': 'macd_sar',
            'ETH/USDT:USDT': 'macd_sar',
            'SOL/USDT:USDT': 'macd_sar',
            'WIF/USDT:USDT': 'bb_sar',
            'PEPE/USDT:USDT': 'bb_sar',
            'DOGE/USDT:USDT': 'bb_sar',
            'ZRO/USDT:USDT': 'hybrid',
            'WLD/USDT:USDT': 'hybrid',
            'FIL/USDT:USDT': 'hybrid',
            'XRP/USDT:USDT': 'bb_sar',
            'ARB/USDT:USDT': 'bb_sar',
        }
        self.bb_tp_offset = 0.003
        self.bb_sl_offset = 0.002
        
        try:
            self.starting_balance = float(self.get_account_balance() or 0.0)
        except Exception:
            self.starting_balance = 0.0
        self.hard_sl_max_loss_pct = 0.03
        self.account_dd_limit_pct = 0.20
        self.cb_close_all = True
        self.cb_enabled = False
        self.circuit_breaker_triggered =self.circuit_breaker_triggered = False
        self.partial_tp_done: Dict[str, set] = {}
        self.allow_cancel_pending = True
        self.safe_cancel_only_our_tpsl = True
        self.tpsl_cl_prefix = 'MACD_TPSL_'
        
        self.atr_sl_n = 1.8
        self.atr_tp_m = 2.2
        
        self.sl_tp_state: Dict[str, Dict[str, float]] = {}
        self.okx_tp_sl_placed: Dict[str, bool] = {}
        self.range_pt_state: Dict[str, Dict[str, Any]] = {}
        self.watchlist_symbols = list(self.symbols)
        self.market_state: Dict[str, str] = {}
        self.key_levels: Dict[str, Dict[str, List[float]]] = {}
        self.tp_boost_map: Dict[str, float] = {s: 1.0 for s in self.symbols}
        self.tp_sl_last_placed: Dict[str, float] = {}
        self.tp_sl_refresh_interval = 300
        self.tp_sl_min_delta_ticks = 2
        
        self.symbol_cfg: Dict[str, Dict[str, float | str]] = {
            "ZRO/USDT:USDT": {"period": 14, "n": 2.2, "m": 3.0, "trigger_pct": 0.010, "trail_pct": 0.006, "update_basis": "high"},
            "WIF/USDT:USDT": {"period": 14, "n": 2.5, "m": 4.0, "trigger_pct": 0.017, "trail_pct": 0.008, "update_basis": "high"},
            "WLD/USDT:USDT": {"period": 14, "n": 2.0, "m": 3.5, "trigger_pct": 0.010, "trail_pct": 0.006, "update_basis": "close"},
            "FIL/USDT:USDT": {"period": 14, "n": 2.0, "m": 3.5, "trigger_pct": 0.010, "trail_pct": 0.006, "update_basis": "high"},
            "BTC/USDT:USDT": {"period": 20, "n": 1.5, "m": 3.0, "trigger_pct": 0.008, "trail_pct": 0.004, "update_basis": "high"},
            "ETH/USDT:USDT": {"period": 18, "n": 1.8, "m": 3.5, "trigger_pct": 0.008, "trail_pct": 0.005, "update_basis": "high"},
            "SOL/USDT:USDT": {"period": 16, "n": 2.0, "m": 4.0, "trigger_pct": 0.012, "trail_pct": 0.007, "update_basis": "high"},
            "XRP/USDT:USDT": {"period": 16, "n": 1.8, "m": 3.5, "trigger_pct": 0.010, "trail_pct": 0.006, "update_basis": "close"},
            "DOGE/USDT:USDT": {"period": 16, "n": 2.5, "m": 5.0, "trigger_pct": 0.017, "trail_pct": 0.008, "update_basis": "high"},
            "PEPE/USDT:USDT": {"period": 14, "n": 3.0, "m": 6.0, "trigger_pct": 0.022, "trail_pct": 0.010, "update_basis": "high"},
            "ARB/USDT:USDT": {"period": 15, "n": 2.2, "m": 3.8, "trigger_pct": 0.014, "trail_pct": 0.006, "update_basis": "high"}
        }
        
        self.trailing_peak: Dict[str, float] = {}
        self.trailing_trough: Dict[str, float] = {}
        self.last_trade_candle_index: Dict[str, int] = {}
        self.stage3_done: Dict[str, bool] = {}
        self.ma_type = os.environ.get('MA_TYPE', 'sma').strip().lower() or 'sma'
        self.ma_fast = int(os.environ.get('MA_FAST', '5'))
        self.ma_slow = int(os.environ.get('MA_SLOW', '20'))
        self.vol_ma_period = int(os.environ.get('VOL_MA_PERIOD', '20'))
        self.vol_boost = float(os.environ.get('VOL_BOOST', '1.2'))
        self.long_body_pct = float(os.environ.get('LONG_BODY_PCT', '0.6'))
        self.cooldown_candles = int(os.environ.get('COOLDOWN_CANDLES', '3'))
        self.trail_stage_1 = float(os.environ.get('TRAIL_STAGE_1', '1.0'))
        self.trail_stage_2 = float(os.environ.get('TRAIL_STAGE_2', '1.75'))
        self.trail_stage_3 = float(os.environ.get('TRAIL_STAGE_3', '2.5'))
        self.trail_stage2_offset = float(os.environ.get('TRAIL_STAGE2_OFFSET', '0.8'))
        self.trail_sl_min_delta_atr = float(os.environ.get('TRAIL_SL_MIN_DELTA_ATR', '0.2'))
        self.partial_tp_ratio_stage3 = float(os.environ.get('PARTIAL_TP_RATIO_STAGE3', '0.3'))
        self.allow_strong_pa_override = (os.environ.get('ALLOW_STRONG_PA_OVERRIDE', 'true').lower() in ('1','true','yes'))
        
        self.last_position_state: Dict[str, str] = {}
        
        self._setup_exchange()
        self._load_markets()
        self.sync_all_status()
        self.handle_existing_positions_and_orders()
    
    def _sleep_with_throttle(self):
        """满足最小调用间隔,加入轻微抖动"""
        try:
            now = time.time()
            delta = now - float(self._last_api_ts or 0.0)
            min_int = float(self._min_api_interval or 0.2)
            if delta < min_int:
                jitter = float(np.random.uniform(0, min_int * 0.1))
                time.sleep(min_int - delta + jitter)
            self._last_api_ts = time.time()
        except Exception:
            time.sleep(float(self._min_api_interval or 0.2))

    def get_position_mode(self) -> str:
        """返回持仓模式,默认hedge(双向)"""
        try:
            opts = self.exchange.options or {}
            mode = str(opts.get('positionMode', 'hedge')).lower()
            return 'hedge' if mode not in ('net', 'oneway') else 'net'
        except Exception:
            return 'hedge'

    def _safe_call(self, func, *args, **kwargs):
        """API调用包装:先节流;遇到50011执行指数退避重试"""
        try:
            retries = int((os.environ.get('MAX_RETRIES') or '3').strip() or 3)
        except Exception:
            retries = 3
        try:
            base = float((os.environ.get('BACKOFF_BASE') or '0.8').strip() or 0.8)
        except Exception:
            base = 0.8
        try:
            max_wait = float((os.environ.get('BACKOFF_MAX') or '3.0').strip() or 3.0)
        except Exception:
            max_wait = 3.0

        for i in range(retries + 1):
            try:
                self._sleep_with_throttle()
                return func(*args, **kwargs)
            except Exception as e:
                msg = str(e)
                is_rate = ('50011' in msg) or ('Too Many Requests' in msg)
                if not is_rate or i >= retries:
                    raise
                wait = min(max_wait, base * (2 ** i)) + float(np.random.uniform(0, 0.2))
                logger.warning(f"⏳ 限频(50011) 第{i+1}次重试,等待 {wait:.2f}s")
                time.sleep(wait)
        return None

    def _setup_exchange(self):
        """设置交易所配置"""
        try:
            self.exchange.check_required_credentials()
            try:
                self.exchange.version = 'v5'
            except Exception:
                pass
            opts = self.exchange.options or {}
            opts.update({'defaultType': 'swap', 'defaultSettle': 'USDT', 'version': 'v5'})
            self.exchange.options = opts
            logger.info("✅ API连接验证成功")
            
            self.sync_exchange_time()
            
            try:
                self.exchange.load_markets(True, {'type': 'swap'})
                logger.info("✅ 预加载市场数据完成 (swap)")
            except Exception as e:
                logger.warning(f"⚠️ 预加载市场数据失败,将使用安全回退: {e}")
            
            if self.set_leverage_on_start:
                for symbol in self.symbols:
                    try:
                        lev = self.symbol_leverage.get(symbol, 20)
                        inst_id = self.symbol_to_inst_id(symbol)
                        try:
                            self.exchange.privatePostAccountSetLeverage({'instId': inst_id, 'lever': str(lev), 'mgnMode': 'cross', 'posSide': 'long'})
                        except Exception:
                            pass
                        try:
                            self.exchange.privatePostAccountSetLeverage({'instId': inst_id, 'lever': str(lev), 'mgnMode': 'cross', 'posSide': 'short'})
                        except Exception:
                            pass
                        logger.info(f"✅ 设置{symbol}杠杆为{lev}倍")
                    except Exception as e:
                        logger.warning(f"⚠️ 设置{symbol}杠杆失败(可能已设置): {e}")
            
            try:
                self.exchange.set_position_mode(True)
                logger.info("✅ 设置为双向持仓模式(多空分开)")
            except Exception as e:
                logger.warning(f"⚠️ 设置持仓模式失败(当前可能有持仓,跳过设置)")
                logger.info("ℹ️ 程序将继续运行,使用当前持仓模式")
            
        except Exception as e:
            logger.error(f"❌ 交易所设置失败: {e}")
            raise
    
    def _load_markets(self):
        """加载市场信息"""
        try:
            logger.info("📄 加载市场信息...")
            resp = self.exchange.publicGetPublicInstruments({'instType': 'SWAP'})
            data = resp.get('data') if isinstance(resp, dict) else resp
            spec_map = {}
            for it in (data or []):
                if it.get('settleCcy') == 'USDT':
                    spec_map[it.get('instId')] = it
            for symbol in self.symbols:
                inst_id = self.symbol_to_inst_id(symbol)
                it = spec_map.get(inst_id, {})
                min_sz = float(it.get('minSz') or 0) or 0.000001
                lot_sz = float(it.get('lotSz') or 0) or None
                tick_sz = float(it.get('tickSz') or 0) or 0.0001
                amt_prec = len(str(lot_sz).split('.')[-1]) if lot_sz and '.' in str(lot_sz) else 8
                px_prec = len(str(tick_sz).split('.')[-1]) if '.' in str(tick_sz) else 4
                self.markets_info[symbol] = {
                    'min_amount': min_sz,
                    'min_cost': 0.0,
                    'amount_precision': amt_prec,
                    'price_precision': px_prec,
                    'lot_size': lot_sz,
                }
                logger.info(f"📊 {symbol} - 最小数量:{min_sz:.8f} 步进:{(lot_sz or 0):.8f} Tick:{tick_sz:.8f}")
            logger.info("✅ 市场信息加载完成")
        except Exception as e:
            logger.error(f"❌ 加载市场信息失败: {e}")
            for symbol in self.symbols:
                self.markets_info[symbol] = {
                    'min_amount': 0.000001,
                    'min_cost': 0.1,
                    'amount_precision': 8,
                    'price_precision': 4,
                    'lot_size': None,
                }
    
    def sync_exchange_time(self):
        """同步交易所时间"""
        try:
            server_time = int(self.exchange.fetch_time() or 0)
            local_time = int(time.time() * 1000)
            time_diff = server_time - local_time
            
            china_tz = pytz.timezone('Asia/Shanghai')
            server_dt = datetime.datetime.fromtimestamp(server_time / 1000, tz=china_tz)
            local_dt = datetime.datetime.fromtimestamp(local_time / 1000, tz=china_tz)
            
            logger.info(f"🕐 交易所时间: {server_dt.strftime('%Y-%m-%d %H:%M:%S')} (北京时间)")
            logger.info(f"🕐 本地时间: {local_dt.strftime('%Y-%m-%d %H:%M:%S')} (北京时间)")
            logger.info(f"⏱️ 时间差: {time_diff}ms")
            
            if abs(time_diff) > 5000:
                logger.warning(f"⚠️ 时间差较大: {time_diff}ms,可能影响交易")
            
            return time_diff
            
        except Exception as e:
            logger.error(f"❌ 同步时间失败: {e}")
            return 0
    
    def get_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        """获取未成交订单"""
        try:
            inst_id = self.symbol_to_inst_id(symbol)
            resp = self._safe_call(self.exchange.privateGetTradeOrdersPending, {'instType': 'SWAP', 'instId': inst_id})
            data = resp.get('data') if isinstance(resp, dict) else resp
            results = []
            for o in (data or []):
                results.append({
                    'id': o.get('ordId') or o.get('clOrdId'),
                    'side': 'buy' if o.get('side') == 'buy' else 'sell',
                    'amount': float(o.get('sz') or 0),
                    'price': float(o.get('px') or 0) if o.get('px') else None,
                })
            return results
        except Exception as e:
            logger.error(f"❌ 获取{symbol}挂单失败: {e}")
            return []
    
    def cancel_all_orders(self, symbol: str) -> bool:
        """取消所有未成交订单"""
        try:
            orders = self.get_open_orders(symbol)
            if not orders:
                return True
            
            for order in orders:
                try:
                    self.exchange.cancel_order(order['id'], symbol)
                    logger.info(f"✅ 取消订单: {symbol} {order['id']}")
                except Exception as e:
                    logger.error(f"❌ 取消订单失败: {order['id']} - {e}")
            
            return True
        except Exception as e:
            logger.error(f"❌ 批量取消订单失败: {e}")
            return False

    def cancel_symbol_tp_sl(self, symbol: str) -> bool:
        """撤销该交易对在OKX侧已挂的TP/SL(算法单)"""
        try:
            inst_id = self.symbol_to_inst_id(symbol)
            if not inst_id:
                return True

            resp = self.exchange.privateGetTradeOrdersAlgoPending({'instType': 'SWAP', 'instId': inst_id})
            data = resp.get('data') if isinstance(resp, dict) else resp

            ours: List[Dict[str, str]] = []
            all_items: List[Dict[str, str]] = []
            for it in (data or []):
                try:
                    aid = str((it.get('algoId') or it.get('algoID') or it.get('id') or ''))
                    ord_type = str(it.get('ordType') or '').lower()
                    clid = str(it.get('clOrdId') or '')
                    if not aid or not ord_type:
                        continue
                    item = {'algoId': aid, 'ordType': ord_type}
                    all_items.append(item)
                    if self.tpsl_cl_prefix and clid.startswith(self.tpsl_cl_prefix):
                        ours.append(item)
                except Exception:
                    continue

            def _cancel(items: List[Dict[str, str]]) -> bool:
                if not items:
                    return False
                ok = False
                for it in items:
                    try:
                        self.exchange.privatePostTradeCancelAlgos({'algoId': it['algoId'], 'ordType': it['ordType'], 'instId': inst_id})
                        ok = True
                    except Exception as e1:
                        try:
                            self.exchange.privatePostTradeCancelAlgos({'algoId': it['algoId'], 'instId': inst_id})
                            ok = True
                        except Exception as e2:
                            logger.debug(f"🔧 撤销失败 {symbol}: algoId={it['algoId']} ordType={it['ordType']} err1={e1} err2={e2}")
                return ok

            total = 0
            if ours and _cancel(ours):
                total += len(ours)

            if total == 0 and all_items and _cancel(all_items):
                total += len(all_items)

            if total > 0:
                logger.info(f"✅ 撤销 {symbol} 条件单数量: {total}")
                time.sleep(0.3)
                return True

            logger.info(f"ℹ️ {symbol} 当前无可撤条件单")
            return True

        except Exception as e:
            logger.warning(f"⚠️ 撤销 {symbol} 条件单失败: {e}")
            return False
    
    def sync_all_status(self):
        """同步所有状态"""
        try:
            logger.info("📄 开始同步状态...")
            self.sync_exchange_time()
            
            has_positions = False
            has_orders = False
            
            for symbol in self.symbols:
                position = self.get_position(symbol, force_refresh=True)
                self.positions_cache[symbol] = position
                
                if position['size'] > 0:
                    self.last_position_state[symbol] = position['side']
                    try:
                        kl = self.get_klines(symbol, 50)
                        ps = self.per_symbol_params.get(symbol, {})
                        atr_p = ps.get('atr_period', 14)
                        atr_val = calculate_atr(kl, atr_p)['atr'].iloc[-1] if not kl.empty else 0.0
                        entry = float(position.get('entry_price', 0) or 0)
                        if atr_val > 0 and entry > 0:
                            okx_ok = self.place_okx_tp_sl(symbol, entry, position.get('side', 'long'), atr_val)
                            if okx_ok:
                                logger.info(f"📌 已为已有持仓补挂TP/SL {symbol}")
                            else:
                                logger.warning(f"⚠️ 补挂交易所侧TP/SL失败 {symbol}")
                    except Exception as _e:
                        logger.warning(f"⚠️ 补挂交易所侧TP/SL异常 {symbol}: {_e}")
                    has_positions = True
                else:
                    self.last_position_state[symbol] = 'none'
                
                orders = self.get_open_orders(symbol)
                self.open_orders_cache[symbol] = orders
                
                if position['size'] > 0:
                    logger.info(f"📊 {symbol} 持仓: {position['side']} {position['size']:.6f} @{position['entry_price']:.2f} PNL:{position['unrealized_pnl']:.2f}U 杠杆:{position['leverage']}x")
                
                if orders:
                    has_orders = True
                    logger.info(f"📋 {symbol} 挂单数量: {len(orders)}")
                    for order in orders:
                        logger.info(f"   └─ {order['side']} {order['amount']:.6f} @{order.get('price', 'market')}")
            
            if not has_positions:
                logger.info("ℹ️ 当前无持仓")
            
            if not has_orders:
                logger.info("ℹ️ 当前无挂单")
            
            self.last_sync_time = time.time()
            logger.info("✅ 状态同步完成")
            
        except Exception as e:
            logger.error(f"❌ 同步状态失败: {e}")
    
    def handle_existing_positions_and_orders(self):
        """处理程序启动时已有的持仓和挂单"""
        logger.info("=" * 70)
        logger.info("🔍 检查启动前的持仓和挂单状态...")
        logger.info("=" * 70)
        
        has_positions = False
        has_orders = False
        
        balance = self.get_account_balance()
        logger.info(f"💰 当前可用余额: {balance:.4f} USDT")
        logger.info(f"💡 11个币种交易:支持0.1U起的小额交易")
        
        for symbol in self.symbols:
            position = self.get_position(symbol, force_refresh=True)
            if position['size'] > 0:
                has_positions = True
                logger.warning(f"⚠️ 检测到{symbol}已有持仓: {position['side']} {position['size']:.6f} @{position['entry_price']:.4f} PNL:{position['unrealized_pnl']:.2f}U")
                self.last_position_state[symbol] = position['side']
            
            orders = self.get_open_orders(symbol)
            if orders:
                has_orders = True
                logger.warning(f"⚠️ 检测到{symbol}有{len(orders)}个未成交订单")
                for order in orders:
                    logger.info(f"   └─ {order['side']} {order['amount']:.6f} @{order.get('price', 'market')} ID:{order['id']}")
        
        if has_positions or has_orders:
            logger.info("=" * 70)
            logger.info("❓ 程序启动时检测到已有持仓或挂单")
            logger.info("💡 策略说明:")
            logger.info("   1. 已有持仓: 程序会根据MACD信号管理,出现反向信号时平仓")
            logger.info("   2. 已有挂单: 程序会在下次交易前自动取消")
            logger.info("   3. 程序会继续运行并根据信号执行交易")
            logger.info("=" * 70)
            logger.info("⚠️ 如果需要立即平仓所有持仓,请手动操作或重启程序前先手动平仓")
            logger.info("=" * 70)
        else:
            logger.info("✅ 启动前无持仓和挂单,可以正常运行")
            logger.info("=" * 70)
    
    def display_current_positions(self):
        """显示当前所有持仓状态"""
        logger.info("")
        logger.info("=" * 70)
        logger.info("📊 当前持仓状态")
        logger.info("=" * 70)
        
        has_positions = False
        total_pnl = 0.0
        
        for symbol in self.symbols:
            position = self.get_position(symbol, force_refresh=False)
            if position['size'] > 0:
                has_positions = True
                pnl = position['unrealized_pnl']
                total_pnl += pnl
                pnl_emoji = "📈" if pnl > 0 else "📉" if pnl < 0 else "➖"
                logger.info(f"{pnl_emoji} {symbol}: {position['side'].upper()} | 数量:{position['size']:.6f} | 入场价:{position['entry_price']:.2f} | 盈亏:{pnl:.2f}U | 杠杆:{position['leverage']}x")
        
        if has_positions:
            total_emoji = "💰" if total_pnl > 0 else "💸" if total_pnl < 0 else "➖"
            logger.info("-" * 70)
            logger.info(f"{total_emoji} 总浮动盈亏: {total_pnl:.2f} USDT")
        else:
            logger.info("ℹ️ 当前无持仓")
        
        logger.info("=" * 70)
        logger.info("")
    
    def check_sync_needed(self):
        """检查是否需要同步状态"""
        current_time = time.time()
        if current_time - self.last_sync_time >= self.sync_interval:
            self.sync_all_status()
    
    def get_account_balance(self) -> float:
        """获取账户余额"""
        try:
            resp = self.exchange.privateGetAccountBalance({})
            data = resp.get('data') if isinstance(resp, dict) else resp
            avail = 0.0
            for acc in (data or []):
                for d in (acc.get('details') or []):
                    if d.get('ccy') == 'USDT':
                        v = d.get('availBal') or d.get('cashBal') or '0'
                        try:
                            avail = float(v)
                        except Exception:
                            avail = 0.0
                        break
            return avail
        except Exception as e:
            logger.error(f"❌ 获取账户余额失败: {e}")
            return 0.0
    
    def get_klines(self, symbol: str, limit: int = 150) -> pd.DataFrame:
        """获取历史数据"""
        try:
            inst_id = self.symbol_to_inst_id(symbol)
            tf = self.timeframe_map.get(symbol, self.timeframe)
            params = {'instId': inst_id, 'bar': tf, 'limit': str(limit)}
            resp = self.exchange.publicGetMarketCandles(params)
            rows = resp.get('data') if isinstance(resp, dict) else resp
            result: List[Dict] = []
            for r in (rows or []):
                ts = int(r[0])
                o = float(r[1]); h = float(r[2]); l = float(r[3]); c = float(r[4]); v = float(r[5])
                result.append({
                    'timestamp': pd.to_datetime(ts, unit='ms'),
                    'open': o, 'high': h, 'low': l, 'close': c, 'volume': v
                })
            result.sort(key=lambda x: x['timestamp'])
            df = pd.DataFrame(result)
            return df
        except Exception as e:
            logger.error(f"❌ 获取{symbol}K线数据失败: {e}")
            return pd.DataFrame()

    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """计算ADX(平均趋势指标)"""
        if len(df) < period + 2:
            return 0.0
        high = df['high']
        low = df['low']
        close = df['close']
        plus_dm = (high.diff()).clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        plus_dm[plus_dm < minus_dm] = 0
        minus_dm[minus_dm < plus_dm] = 0
        tr1 = (high - low)
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).sum() / atr)
        minus_di = 100 * (minus_dm.rolling(period).sum() / atr)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
        adx = dx.rolling(period).mean()
        try:
            return float(adx.iloc[-1])
        except Exception:
            return 0.0

    def calculate_bb_width(self, df: pd.DataFrame, period: int = 20, k: float = 2.0) -> float:
        """计算布林带宽度"""
        if len(df) < period + 1:
            returnreturn 0.0
        mid = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std(ddof=0)
        upper = mid + k * std
        lower = mid - k * std
        width = (upper - lower) / mid.replace(0, np.nan)
        w = float(width.iloc[-1]) if not np.isnan(width.iloc[-1]) else 0.0
        return max(0.0, w)

    def ema_alignment(self, df: pd.DataFrame) -> str:
        """EMA9/20/50排列"""
        if len(df) < 50:
            return 'neutral'
        latest = df.iloc[-1]
        if latest['ema_9'] > latest['ema_20'] > latest['ema_50']:
            return 'bull'
        if latest['ema_9'] < latest['ema_20'] < latest['ema_50']:
            return 'bear'
        return 'neutral'

    def price_range_metric(self, df: pd.DataFrame, lookback: int = 30) -> float:
        """近30根K线波动幅度"""
        if len(df) < lookback:
            return 0.0
        sub = df.tail(lookback)
        hi = float(sub['high'].max())
        lo = float(sub['low'].min())
        if lo <= 0:
            return 0.0
        return (hi - lo) / lo

    def assess_market_state(self, df: pd.DataFrame) -> Dict[str, Any]:
        """综合判断市场状态与置信度"""
        adx = self.calculate_adx(df, period=14)
        bb_w = self.calculate_bb_width(df, period=20, k=2.0)
        ema_align = self.ema_alignment(df)
        pr = self.price_range_metric(df, lookback=30)

        trend_score = 0
        range_score = 0

        if adx > 25: trend_score += 40
        elif adx < 20: range_score += 40
        else: trend_score += 15; range_score += 15

        if bb_w > 0.06: trend_score += 25
        elif bb_w < 0.03: range_score += 25
        else: trend_score += 10; range_score += 10

        if ema_align == 'bull' or ema_align == 'bear':
            trend_score += 20
        else:
            range_score += 15

        if pr > 0.10: trend_score += 15
        else: range_score += 15

        if trend_score >= range_score and trend_score >= 60:
            state = 'trending'
            confidence = trend_score
        elif range_score > trend_score and range_score >= 60:
            state = 'ranging'
            confidence = range_score
        else:
            state = 'unclear'
            confidence = max(trend_score, range_score)

        return {'state': state, 'confidence': confidence, 'adx': adx, 'bb_width': bb_w, 'ema_align': ema_align, 'price_range': pr}

    def identify_key_levels(self, df: pd.DataFrame, window: int = 5, vol_ma_period: int = 20, tolerance: float = 0.005, lookback: int = 100) -> Dict[str, List[Dict[str, Any]]]:
        """支撑/压力识别"""
        if len(df) < max(vol_ma_period + window + 5, lookback):
            return {'supports': [], 'resistances': []}
        sub = df.tail(lookback).copy()
        sub['vol_ma'] = sub['volume'].rolling(vol_ma_period).mean()
        supports: List[Dict[str, Any]] = []
        resistances: List[Dict[str, Any]] = []

        rows = sub.reset_index(drop=True)

        for i in range(window, len(rows) - window):
            slice_ = rows.iloc[i-window:i+window+1]
            vol_ok = float(rows.iloc[i]['volume']) >= 0.8 * float(rows.iloc[i]['vol_ma'] or 1.0)
            if rows.iloc[i]['low'] == slice_['low'].min() and vol_ok:
                supports.append({'price': float(rows.iloc[i]['low']), 'idx': i, 'tests': 1, 'vol_mult': float(rows.iloc[i]['volume']) / max(1e-9, float(rows.iloc[i]['vol_ma'] or 1.0))})
            if rows.iloc[i]['high'] == slice_['high'].max() and vol_ok:
                resistances.append({'price': float(rows.iloc[i]['high']), 'idx': i, 'tests': 1, 'vol_mult': float(rows.iloc[i]['volume']) / max(1e-9, float(rows.iloc[i]['vol_ma'] or 1.0))})

        def cluster_levels(levels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            if not levels:
                return []
            levels_sorted = sorted(levels, key=lambda x: x['price'])
            clustered: List[Dict[str, Any]] = []
            cur = levels_sorted[0].copy()
            for lv in levels_sorted[1:]:
                if abs(lv['price'] - cur['price']) / cur['price'] <= tolerance:
                    cur['price'] = (cur['price'] * cur['tests'] + lv['price']) / (cur['tests'] + 1)
                    cur['tests'] += 1
                    cur['vol_mult'] = (cur['vol_mult'] + lv['vol_mult']) / 2.0
                else:
                    clustered.append(cur)
                    cur = lv.copy()
            clustered.append(cur)
            for it in clustered:
                it['strength'] = float(it['vol_mult']) * int(it['tests'])
            clustered.sort(key=lambda x: x.get('strength', 0), reverse=True)
            return clustered[:5]

        return {'supports': cluster_levels(supports), 'resistances': cluster_levels(resistances)}

    def update_learning_state(self, symbol: str, pnl_percent: float) -> None:
        """根据最新平仓盈亏更新学习状态"""
        try:
            st = self.learning_state.get(symbol)
            if not st:
                return
            outcome = 1 if pnl_percent >= 0 else -1
            st['recent_outcomes'].append(outcome)
            st['recent_pnls'].append(float(pnl_percent))
            if len(st['recent_outcomes']) > 50:
                st['recent_outcomes'] = st['recent_outcomes'][-50:]
            if len(st['recent_pnls']) > 50:
                st['recent_pnls'] = st['recent_pnls'][-50:]
            total = len(st['recent_outcomes'])
            wins = sum(1 for x in st['recent_outcomes'] if x > 0)
            winrate = (wins / total) * 100 if total > 0 else 50.0
            avg_pnl = np.mean(st['recent_pnls']) if st['recent_pnls'] else 0.0
            mul = 1.0 + (winrate - 50.0) / 100.0
            mul = max(0.6, min(1.4, mul))
            st['risk_multiplier'] = round(mul, 3)
            try:
                last3 = st['recent_outcomes'][-3:] if len(st['recent_outcomes']) >= 3 else []
                losing_streak = (len(last3) == 3 and sum(1 for x in last3 if x < 0) >= 3)
            except Exception:
                losing_streak = False
            step_rsi = 1.0 if losing_streak or winrate < 45.0 else (-1.0 if winrate > 60.0 else 0.0)
            st['rsi_overbought_delta'] = float(np.clip(st['rsi_overbought_delta'] + step_rsi, -5.0, 5.0))
            st['rsi_oversold_delta'] = float(np.clip(st['rsi_oversold_delta'] - step_rsi, -5.0, 5.0))
            step_score = 1.0 if losing_streak or winrate < 45.0 else (-1.0 if winrate > 60.0 else 0.0)
            st['range_threshold_delta'] = float(np.clip(st['range_threshold_delta'] + step_score, -5.0, 5.0))
            st['trend_threshold_delta'] = float(np.clip(st['trend_threshold_delta'] + step_score, -5.0, 5.0))
            step_atr = 0.02 if losing_streak or winrate < 45.0 else (-0.02 if winrate > 60.0 else 0.0)
            st['atr_n_delta'] = float(np.clip(st['atr_n_delta'] + step_atr, -0.10, 0.10))
            st['atr_m_delta'] = float(np.clip(st['atr_m_delta'] - step_atr, -0.10, 0.10))
            logger.debug(
                "🧠 学习更新 %s: winrate=%.1f%% mul=%.2f "
                "rsiΔ=(%+.1f,%+.1f) scoreΔ=(%+.1f,%+.1f) atrΔ=(%+.2f,%+.2f)" % (
                    symbol, winrate, st['risk_multiplier'],
                    st['rsi_overbought_delta'], st['rsi_oversold_delta'],
                    st['range_threshold_delta'], st['trend_threshold_delta'],
                    st['atr_n_delta'], st['atr_m_delta']
                )
            )
        except Exception as e:
            logger.debug(f"🔧 学习更新异常 {symbol}: {e}")

    def get_learning_adjustments(self, symbol: str) -> Dict[str, float]:
        """返回当前学习调整项"""
        st = self.learning_state.get(symbol, {})
        return {
            'risk_multiplier': float(st.get('risk_multiplier', 1.0) or 1.0),
            'rsi_overbought_delta': float(st.get('rsi_overbought_delta', 0.0) or 0.0),
            'rsi_oversold_delta': float(st.get('rsi_oversold_delta', 0.0) or 0.0),
            'range_threshold_delta': float(st.get('range_threshold_delta', 0.0) or 0.0),
            'trend_threshold_delta': float(st.get('trend_threshold_delta', 0.0) or 0.0),
            'atr_n_delta': float(st.get('atr_n_delta', 0.0) or 0.0),
            'atr_m_delta': float(st.get('atr_m_delta', 0.0) or 0.0),
        }

    def score_ranging_long(self, price: float, supports: List[Dict[str, Any]], rsi: float, rsi_threshold: float) -> Dict[str, Any]:
        """震荡市做多评分"""
        if price <= 0 or not supports:
            return {'score': 0, 'near_level': None}
        nearest = min(supports, key=lambda x: abs(price - x['price']))
        dist_pct = abs(price - nearest['price']) / nearest['price']
        score = 0
        if dist_pct < 0.01:
            score += 60
            if rsi < rsi_threshold:
                score += 25
            score += min(15, 5 * int(nearest.get('tests', 1)))
        return {'score': score, 'near_level': nearest}

    def score_ranging_short(self, price: float, resistances: List[Dict[str, Any]], rsi: float, rsi_threshold: float) -> Dict[str, Any]:
        """震荡市做空评分"""
        if price <= 0 or not resistances:
            return {'score': 0, 'near_level': None}
        nearest = min(resistances, key=lambda x: abs(price - x['price']))
        dist_pct = abs(price - nearest['price']) / nearest['price']
        score = 0
        if dist_pct < 0.01:
            score += 60
            if rsi > rsi_threshold:
                score += 25
            score += min(15, 5 * int(nearest.get('tests', 1)))
        return {'score': score, 'near_level': nearest}

    def score_trending_long(self, df: pd.DataFrame, resistances: List[Dict[str, Any]], adx: float) -> Dict[str, Any]:
        """趋势市做多评分"""
        if len(df) < 5:
            return {'score': 0, 'level': None}
        latest = df.iloc[-1]; prev = df.iloc[-2]
        macd_gc = (prev['macd_diff'] <= prev['macd_dea'] and latest['macd_diff'] > latest['macd_dea'])
        if not macd_gc:
            return {'score': 0, 'level': None}
        level = None
        if resistances:
            level = min(resistances, key=lambda x: abs(latest['close'] - x['price']))
            broke = (prev['close'] < level['price'] and latest['close'] > level['price'])
        else:
            broke = True
        vol_ok = latest.get('volume_ratio', 1.0) > 1.2
        if broke and vol_ok:
            score = 75 + 20 + (10 if adx > 30 else 0)
            return {'score': score, 'level': level}
        return {'score': 0, 'level': None}

    def score_trending_short(self, df: pd.DataFrame, supports: List[Dict[str, Any]], adx: float) -> Dict[str, Any]:
        """趋势市做空评分"""
        if len(df) < 5:
            return {'score': 0, 'level': None}
        latest = df.iloc[-1]; prev = df.iloc[-2]
        macd_dc = (prev['macd_diff'] >= prev['macd_dea'] and latest['macd_diff'] < latest['macd_dea'])
        if not macd_dc:
            return {'score': 0, 'level': None}
        level = None
        if supports:
            level = min(supports, key=lambda x: abs(latest['close'] - x['price']))
            broke = (prev['close'] > level['price'] and latest['close'] < level['price'])
        else:
            broke = True
        vol_ok = latest.get('volume_ratio', 1.0) > 1.2
        if broke and vol_ok:
            score = 75 + 20 + (10 if adx > 30 else 0)
            return {'score': score, 'level': level}
        return {'score': 0, 'level': None}

    def analyze_symbol_adaptive(self, symbol: str) -> Dict[str, str]:
        """自适应策略分析"""
        try:
            df = self.get_klines(symbol, 150)
            if df.empty or len(df) < 60:
                return {'signal': 'hold', 'reason': '数据不足'}
            df = self.calculate_indicators(df, symbol)

            ms = self.assess_market_state(df)
            latest = df.iloc[-1]
            rsi_th = self.rsi_thresholds.get(symbol, {'overbought': 70, 'oversold': 30})
            adj = self.get_learning_adjustments(symbol)
            rsi_th = {
                'overbought': max(50, min(90, rsi_th['overbought'] + adj.get('rsi_overbought_delta', 0.0))),
                'oversold':   max(10, min(50, rsi_th['oversold']   + adj.get('rsi_oversold_delta', 0.0))),
            }
            ranging_min = int(70 + adj.get('range_threshold_delta', 0.0))
            trending_min = int(75 + adj.get('trend_threshold_delta', 0.0))

            now_ts = time.time()
            cache = self.key_levels_cache.get(symbol, {})
            if (not cache) or (now_ts - float(cache.get('ts', 0)) > 3600):
                levels = self.identify_key_levels(df, window=5, vol_ma_period=self.vol_ma_period, tolerance=0.005, lookback=100)
                self.key_levels_cache[symbol] = {'ts': now_ts, 'supports': levels['supports'], 'resistances': levels['resistances']}
                logger.info(f"🔍 更新关键位 {symbol}: 支撑{len(levels['supports'])} 压力{len(levels['resistances'])}")
            else:
                levels = {'supports': cache.get('supports', []), 'resistances': cache.get('resistances', [])}

            if ms['state'] == 'ranging' and ms['confidence'] >= 60:
                long_eval = self.score_ranging_long(latest['close'], levels['supports'], latest['rsi'], rsi_th['oversold'])
                short_eval = self.score_ranging_short(latest['close'], levels['resistances'], latest['rsi'], rsi_th['overbought'])
                if long_eval['score'] >= ranging_min:
                    return {'signal': 'buy', 'reason': f"震荡市支撑反弹,总分{long_eval['score']}(支撑{long_eval['near_level']['price']:.4f} 测试{long_eval['near_level']['tests']}次)"}
                if short_eval['score'] >= ranging_min:
                    return {'signal': 'sell', 'reason': f"震荡市压力回落,总分{short_eval['score']}(压力{short_eval['near_level']['price']:.4f} 测试{short_eval['near_level']['tests']}次)"}
                return {'signal': 'hold', 'reason': '震荡市未达阈值'}

            if ms['state'] == 'trending' and ms['confidence'] >= 60:
                long_eval = self.score_trending_long(df, levels['resistances'], ms['adx'])
                short_eval = self.score_trending_short(df, levels['supports'], ms['adx'])
                if long_eval['score'] >= trending_min:
                    desc = f"趋势市金叉突破,总分{long_eval['score']}" + (f"(突破{long_eval['level']['price']:.4f})" if long_eval['level'] else "")
                    return {'signal': 'buy', 'reason': desc}
                if short_eval['score'] >= trending_min:
                    desc = f"趋势市死叉下破,总分{short_eval['score']}" + (f"(跌破{short_eval['level']['price']:.4f})" if short_eval['level'] else "")
                    return {'signal': 'sell', 'reason': desc}
                return {'signal': 'hold', 'reason': '趋势市未达阈值'}

            prev = df.iloc[-2]
            macd_gc = (prev['macd_diff'] <= prev['macd_dea'] and latest['macd_diff'] > latest['macd_dea'])
            macd_dc = (prev['macd_diff'] >= prev['macd_dea'] and latest['macd_diff'] < latest['macd_dea'])
            if macd_gc and latest['rsi'] < rsi_th['overbought']:
                return {'signal': 'buy', 'reason': '保守策略:金叉+RSI不过热(降低仓位)'}
            if macd_dc and latest['rsi'] > rsi_th['oversold']:
                return {'signal': 'sell', 'reason': '保守策略:死叉+RSI不过冷(降低仓位)'}

            return {'signal': 'hold', 'reason': '市场不明确/无信号'}

        except Exception as e:
            logger.error(f"❌ 自适应分析失败 {symbol}: {e}")
            return {'signal': 'hold', 'reason': f'分析异常: {e}'}
    
    def get_position(self, symbol: str, force_refresh: bool = False) -> Dict[str, Any]:
        """获取当前持仓"""
        try:
            if not force_refresh and symbol in self.positions_cache:
                return self.positions_cache[symbol]
            
            inst_id = self.symbol_to_inst_id(symbol)
            resp = self._safe_call(self.exchange.privateGetAccountPositions, {'instType': 'SWAP', 'instId': inst_id})
            data = resp.get('data') if isinstance(resp, dict) else resp
            for p in (data or []):
                if p.get('instId') == inst_id and float(p.get('pos', 0) or 0) != 0:
                    size = abs(float(p.get('pos', 0) or 0))
                    # 🔧 修复: 严格判断持仓方向
                    pos_side_raw = str(p.get('posSide', '')).lower()
                    if pos_side_raw == 'long':
                        side = 'long'
                    elif pos_side_raw == 'short':
                        side = 'short'
                    else:
                        side = 'none'  # 未知方向标记为none
                    
                    try:
                        entry_price = float(p.get('avgPx') or p.get('lastAvgPrice') or p.get('avgPrice') or 0)
                    except Exception:
                        entry_price = float(p.get('avgPx', 0) or 0)
                    leverage = float(p.get('lever', 0) or 0)
                    unreal = float(p.get('upl', 0) or 0)
                    margin_mode = str(p.get('mgnMode', 'cross') or 'cross').lower()
                    pos_data = {
                        'size': size,
                        'side': side,
                        'entry_price': entry_price,
                        'unrealized_pnl': unreal,
                        'leverage': leverage,
                        'margin_mode': margin_mode,
                    }
                    self.positions_cache[symbol] = pos_data
                    return pos_data
            
            pos_data = {'size': 0, 'side': 'none', 'entry_price': 0, 'unrealized_pnl': 0, 'leverage': 0}
            self.positions_cache[symbol] = pos_data
            return pos_data
            
        except Exception as e:
            logger.error(f"❌ 获取{symbol}持仓失败: {e}")
            if symbol in self.positions_cache:
                return self.positions_cache[symbol]
            return {'size': 0, 'side': 'none', 'entry_price': 0, 'unrealized_pnl': 0, 'leverage': 0}
    
    def has_open_orders(self, symbol: str) -> bool:
        """检查是否有未成交订单"""
        try:
            orders = self.get_open_orders(symbol)
            has_orders = len(orders) > 0
            if has_orders:
                logger.info(f"⚠️ {symbol}存在{len(orders)}个未成交订单")
            return has_orders
        except Exception as e:
            logger.error(f"❌ 检查挂单失败: {e}")
            return False
    
    def calculate_order_amount(self, symbol: str, active_count: Optional[int] = None) -> float:
        """计算下单金额"""
        try:
            balance = self.get_account_balance()
            if balance <= 0:
                logger.warning(f"⚠️ 余额不足,无法为 {symbol} 分配资金 (余额:{balance:.4f}U)")
                return 0.0

            target_str = os.environ.get('TARGET_NOTIONAL_USDT', '').strip()
            if target_str:
                try:
                    target = max(0.0, float(target_str))
                    logger.info(f"💵 使用固定目标名义金额: {target:.4f}U")
                except Exception:
                    logger.warning(f"⚠️ TARGET_NOTIONAL_USDT 无效: {target_str}")
                    target = 0.0
            else:
                try:
                    target = max(0.0, float((os.environ.get('DEFAULT_ORDER_USDT') or '1.0').strip()))
                except Exception:
                    target = 1.0

            try:
                factor = max(1.0, float((os.environ.get('ORDER_NOTIONAL_FACTOR') or '1').strip()))
            except Exception:
                factor = 1.0
            try:
                adj = self.get_learning_adjustments(symbol)
                risk_mul = float(adj.get('risk_multiplier', 1.0) or 1.0)
            except Exception:
                risk_mul = 1.0
            target *= factor * risk_mul

            def _to_float(env_name: str, default: float) -> float:
                try:
                    s = os.environ.get(env_name, '').strip()
                    return float(s) if s else default
                except Exception:
                    return default

            min_floor = max(0.0, _to_float('MIN_PER_SYMBOL_USDT', 0.1))
            max_cap = max(0.0, _to_float('MAX_PER_SYMBOL_USDT', 0.0))

            if min_floor > 0 and target < min_floor:
                target = min_floor
            if max_cap > 0 and target > max_cap:
                target = max_cap

            if target <= 0:
                logger.warning(f"⚠️ {symbol} 目标金额为0,跳过")
                return 0.0

            try:
                lev = float(self.symbol_leverage.get(symbol, 20) or 20)
                required_margin = target / max(1.0, lev)
                if balance < required_margin * 1.02:
                    logger.warning(f"⚠️ 保证金不足,跳过 {symbol}: 余额={balance:.4f}U 需保证金≈{required_margin:.4f}U (lev={lev:.1f}x, 目标={target:.4f}U)")
                    return 0.0
            except Exception:
                logger.warning(f"⚠️ 保证金估算失败,谨慎起见跳过 {symbol}")
                return 0.0

            logger.info(f"💵 单币分配: 模式=逐币下单, 余额={balance:.4f}U, 因子={factor:.2f}, 本币目标={target:.4f}U")
            return target

        except Exception as e:
            logger.error(f"❌ 计算{symbol}下单金额失败: {e}")
            return 0.0
    
    def create_order(self, symbol: str, side: str, amount: float) -> bool:
        """🔧 修复3: 开仓后立即设置TP/SL保护"""
        try:
            if self.has_open_orders(symbol):
                logger.warning(f"⚠️ {symbol}存在未成交订单,先取消")
                self.cancel_all_orders(symbol)
                time.sleep(1)

            if amount <= 0:
                logger.warning(f"⚠️ {symbol}下单金额为0,跳过")
                return False

            market_info = self.markets_info.get(symbol, {})
            min_amount = float(market_info.get('min_amount', 0.001) or 0.001)
            amount_precision = int(market_info.get('amount_precision', 8) or 8)
            lot_sz = market_info.get('lot_size')

            inst_id = self.symbol_to_inst_id(symbol)
            try:
                tkr = self.exchange.publicGetMarketTicker({'instId': inst_id})
                if isinstance(tkr, dict):
                    d = tkr.get('data') or []
                    if isinstance(d, list) and d:
                        current_price = float(d[0].get('last') or d[0].get('lastPx') or 0.0)
                    else:
                        current_price = 0.0
                else:
                    current_price = 0.0
            except Exception as _e:
                logger.error(f"❌ 获取{symbol}最新价失败({inst_id}): {_e}")
                current_price = 0.0

            if not current_price or current_price <= 0:
                logger.error(f"❌ 无法获取{symbol}有效价格,跳过下单")
                return False

            contract_size = amount / current_price

            if contract_size < min_amount:
                contract_size = min_amount

            step = None
            if lot_sz:
                try:
                    step = float(lot_sz)
                    if step and step > 0:
                        contract_size = math.ceil(contract_size / step) * step
                except Exception:
                    step = None
            contract_size = round(contract_size, amount_precision)

            if contract_size <= 0 or contract_size < min_amount:
                contract_size = max(min_amount, 10 ** (-amount_precision))
                if lot_sz:
                    try:
                        step = float(lot_sz)
                        if step and step > 0:
                            contract_size = math.ceil(contract_size / step) * step
                    except Exception:
                        pass
                contract_size = round(contract_size, amount_precision)

            try:
                used_usdt = contract_size * current_price
                if used_usdt + 1e-12 < amount:
                    need_qty = (amount - used_usdt) / current_price
                    incr_step = step if (step and step > 0) else (10 ** (-amount_precision))
                    add_qty = math.ceil(need_qty / incr_step) * incr_step
                    contract_size = round(contract_size + add_qty, amount_precision)
                    if contract_size < min_amount:
                        contract_size = max(min_amount, 10 ** (-amount_precision))
                        if step and step > 0:
                            contract_size = math.ceil(contract_size / step) * step
                        contract_size = round(contract_size, amount_precision)
            except Exception:
                pass

            try:
                lev = float(self.symbol_leverage.get(symbol,20) or 20)
                est_cost0 = float(contract_size * current_price)
                est_margin0 = est_cost0 / max(1.0, lev)
                if est_margin0 < 0.5:
                    logger.warning(f"⚠️ 预估保证金过低(<0.5U),跳过下单 {symbol}: est_margin={est_margin0:.4f}U 价格={current_price:.6f} 数量={contract_size:.8f} 杠杆={lev}")
                    return False
                avail = float(self.get_account_balance() or 0.0)
                if avail > 0 and est_margin0 > avail * 0.98:
                    ratio = (avail * 0.98 * lev) / max(1e-12, est_cost0)
                    contract_size = max(0.0, contract_size * max(0.1, min(1.0, ratio)))
                    if lot_sz:
                        try:
                            step_pre = float(lot_sz)
                            if step_pre and step_pre > 0:
                                contract_size = math.ceil(contract_size / step_pre) * step_pre
                        except Exception:
                            pass
                    contract_size = round(contract_size, amount_precision)
                    if contract_size <= 0 or contract_size < min_amount:
                        contract_size = max(min_amount, 10 ** (-amount_precision))
                        if lot_sz:
                            try:
                                step_pre2 = float(lot_sz)
                                if step_pre2 and step_pre2 > 0:
                                    contract_size = math.ceil(contract_size / step_pre2) * step_pre2
                            except Exception:
                                pass
                        contract_size = round(contract_size, amount_precision)
                    logger.info(f"🔧 保证金预缩量: 可用={avail:.4f}U 杠杆={lev:.1f}x | 预估保证金={est_margin0:.4f}U → 新数量={contract_size:.8f}")
            except Exception:
                pass

            logger.info(f"🔖 准备下单: {symbol} {side} 金额:{amount:.4f}U 价格:{current_price:.4f} 数量:{contract_size:.8f}")
            try:
                est_cost = contract_size * current_price
                logger.info(f"🧮 下单成本对齐: 分配金额={amount:.4f}U | 预计成本={est_cost:.4f}U | 数量={contract_size:.8f} | minSz={min_amount} | lotSz={lot_sz}")
            except Exception:
                pass

            pos_side = 'long' if side == 'buy' else 'short'
            order_id = None
            last_err = None

            try:
                mi = self.markets_info.get(symbol, {}) or {}
                min_amount = float(mi.get('min_amount', 0) or 0.0)
                lot_sz_val = mi.get('lot_size')
                lot_sz = float(lot_sz_val) if (lot_sz_val not in (None, '')) else 0.0
            except Exception:
                min_amount, lot_sz = 0.0, 0.0
            last_px = 0.0
            try:
                inst_id_np = self.symbol_to_inst_id(symbol)
                tkr_np = self.exchange.publicGetMarketTicker({'instId': inst_id_np})
                d_np = tkr_np.get('data') if isinstance(tkr_np, dict) else tkr_np
                if isinstance(d_np, list) and d_np:
                    last_px = float(d_np[0].get('last') or d_np[0].get('lastPx') or 0.0)
            except Exception:
                last_px = 0.0
            if last_px <= 0:
                try:
                    df_np = self.get_klines(symbol, 10)
                    if isinstance(df_np, pd.DataFrame) and not df_np.empty:
                        last_px = float(df_np['close'].values[-1])
                except Exception:
                    last_px = 0.0
            size_adj = float(contract_size)
            if lot_sz and lot_sz > 0:
                steps = int(size_adj / lot_sz)
                size_adj = max(min_amount, steps * lot_sz) if steps > 0 else max(min_amount, lot_sz)
            else:
                size_adj = max(min_amount, size_adj)
            if last_px > 0 and size_adj * last_px < 0.5:
                logger.warning(f"⚠️ 名义金额过小(<0.5U),跳过下单 {symbol}: size={size_adj} last={last_px:.6f} notional={size_adj*last_px:.4f}U")
                return False
            try:
                avail_chk = float(self.get_account_balance() or 0.0)
                if avail_chk < 0.5:
                    logger.warning(f"⚠️ 可用余额不足(<0.5U),跳过下单 {symbol}: available={avail_chk:.4f}U")
                    return False
            except Exception:
                pass
            try:
                pos_cur = self.get_position(symbol, force_refresh=False) or {}
                pos_sz = float(pos_cur.get('size', 0) or 0.0)
                pos_side_cur = str(pos_cur.get('side', '')).lower()
                want_side = ('long' if str(side).lower() in ('buy','long') else 'short')
                if pos_sz > 0 and pos_side_cur and pos_side_cur != want_side:
                    logger.warning(f"⚠️ 已有{pos_side_cur}持仓,拒绝反向开仓 {symbol}: 现有size={pos_sz}")
                    return False
            except Exception:
                pass

            contract_size = size_adj

            native_only = (os.environ.get('USE_OKX_NATIVE_ONLY', '').strip().lower() in ('1', 'true', 'yes'))

            if not native_only:
                try:
                    side_ccxt = ('buy' if str(side).lower() in ('buy','long') else 'sell')
                    params = {'type': 'market', 'reduceOnly': False, 'posSide': pos_side}
                    order = self.exchange.create_order(symbol, 'market', side_ccxt, contract_size, params=params)
                    order_id = order.get('id')
                except Exception as e:
                    last_err = e
                    logger.warning(f"⚠️ CCXT下单失败: {str(e)} - 尝试OKX原生API")
            
            if order_id is None:
                try:
                    pos_mode = self.get_position_mode()
                    if pos_mode == 'hedge':
                        td_mode = 'cross'
                        pos_side_okx = pos_side
                    else:
                        td_mode = 'cross'
                        pos_side_okx = 'net'
                    
                    side_ccxt = ('buy' if str(side).lower() in ('buy','long') else 'sell')
                    params_okx = {
                        'instId': inst_id,
                        'tdMode': td_mode,
                        'side': side_ccxt,
                        'sz': str(contract_size),
                        'ordType': 'market'
                    }
                    if pos_mode == 'hedge':
                        params_okx['posSide'] = pos_side_okx
                    
                    resp = self.exchange.privatePostTradeOrder(params_okx)
                    data = resp.get('data') if isinstance(resp, dict) else resp
                    if data and isinstance(data, list) and data[0]:
                        order_id = data[0].get('ordId')
                except Exception as e:
                    last_err = e
                    logger.error(f"❌ OKX原生下单失败: {str(e)}")
                    return False
            
            if order_id is None:
                logger.error(f"❌ 下单失败 {symbol}: {last_err}")
                return False
            
            logger.info(f"🚀 下单成功 {symbol}: ID={order_id} {side} {contract_size:.8f} @{current_price:.6f}")
            
            # 🔧 修复3: 开仓后立即设置TP/SL保护
            logger.info(f"🔒 开仓后立即设置TP/SL保护 {symbol}...")
            time.sleep(1.5)  # 等待持仓刷新
            
            try:
                pos = self.get_position(symbol, force_refresh=True)
                if pos and pos['size'] > 0:
                    entry_price = float(pos.get('entry_price', 0) or 0)
                    if entry_price > 0:
                        # 计算ATR
                        df_atr = self.get_klines(symbol, 50)
                        if not df_atr.empty:
                            ps = self.per_symbol_params.get(symbol, {})
                            atr_period = int(ps.get('atr_period', 14))
                            df_atr = calculate_atr(df_atr, atr_period)
                            atr_val = float(df_atr['atr'].iloc[-1])
                            
                            # 初始化SL/TP状态
                            self._set_initial_sl_tp(symbol, entry_price, atr_val, side)
                            
                            # 挂交易所OCO订单
                            success = self.place_okx_tp_sl(symbol, entry_price, side, atr_val)
                            if success:
                                logger.info(f"✅ 开仓后TP/SL设置成功 {symbol}: entry={entry_price:.6f} atr={atr_val:.6f}")
                            else:
                                logger.error(f"❌ 开仓后TP/SL设置失败 {symbol} - 持仓处于无保护状态!")
                        else:
                            logger.error(f"❌ 无法获取K线计算ATR {symbol} - 持仓处于无保护状态!")
                    else:
                        logger.error(f"❌ 无效入场价 {symbol} - 持仓处于无保护状态!")
                else:
                    logger.warning(f"⚠️ 下单后未检测到持仓 {symbol}")
            except Exception as e:
                logger.error(f"❌ 开仓后设置TP/SL异常 {symbol}: {e}")
                logger.error(f"🚨 持仓处于无保护状态,请手动检查!")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 创建订单失败 {symbol}: {str(e)}")
            return False
    
    def _set_initial_sl_tp(self, symbol: str, entry: float, atr: float, side: str) -> bool:
        """初始化 SL/TP"""
        try:
            cfg = self.symbol_cfg.get(symbol, {})
            n = float(cfg.get('n', 2.0))
            m = float(cfg.get('m', 3.0))
            try:
                adj = self.get_learning_adjustments(symbol)
                n *= (1.0 + float(adj.get('atr_n_delta', 0.0) or 0.0))
                m *= (1.0 + float(adj.get('atr_m_delta', 0.0) or 0.0))
            except Exception:
                pass
            atr = max(0.0, float(atr or 0.0))
            entry = float(entry or 0.0)
            if entry <= 0:
                return False

            if str(side).lower() == 'long':
                sl = max(0.0, entry - n * atr)
                tp = max(0.0, entry + m * atr)
            else:
                sl = max(0.0, entry + n * atr)
                tp = max(0.0, entry - m * atr)

            px_prec = int(self.markets_info.get(symbol, {}).get('price_precision', 4) or 4)
            tick_sz = 10 ** (-px_prec)
            min_delta = max(10 * tick_sz, entry * 0.005)
            if str(side).lower() == 'long':
                sl = min(sl, entry - min_delta)
                tp = max(tp, entry + min_delta)
            else:
                sl = max(sl, entry + min_delta)
                tp = min(tp, entry - min_delta)
            sl = round(max(sl, tick_sz), px_prec)
            tp = round(max(tp, tick_sz), px_prec)
            if sl <= 0 or tp <= 0 or sl == tp:
                if str(side).lower() == 'long':
                    sl = round(max(tick_sz, entry - (min_delta + 5 * tick_sz)), px_prec)
                    tp = round(entry + (min_delta + 5 * tick_sz), px_prec)
                else:
                    sl = round(entry + (min_delta + 5 * tick_sz), px_prec)
                    tp = round(max(tick_sz, entry - (min_delta + 5 * tick_sz)), px_prec)

            self.sl_tp_state[symbol] = {
                'entry': entry,
                'sl': sl,
                'tp': tp
            }
            logger.info(f"🧩 初始化SL/TP {symbol} side={side}: entry={entry:.6f} SL={sl:.6f} TP={tp:.6f} (n={n}, m={m}, ATR={atr:.6f})")
            return True
        except Exception as e:
            logger.warning(f"⚠️ 初始化SL/TP失败 {symbol}: {e}")
            return False

    def _update_trailing_stop(self, symbol: str, price: float, atr: float, side: str) -> None:
        """基于峰值/谷值动态推进追踪止损"""
        try:
            st = self.sl_tp_state.get(symbol)
            if not st:
                return
            cfg = self.symbol_cfg.get(symbol, {})
            trigger_pct = float(cfg.get('trigger_pct', 0.01) or 0.01)
            trail_pct = float(cfg.get('trail_pct', 0.006) or 0.006)
            basis = str(cfg.get('update_basis', 'close') or 'close').lower()

            entry = float(st.get('entry', 0) or 0)
            if entry <= 0 or price <= 0:
                return

            activated = False
            if side == 'long':
                activated = (price >= entry * (1 + trigger_pct))
                prev_peak = float(self.trailing_peak.get(symbol, entry) or entry)
                now_basis = price if basis == 'close' else price
                peak = max(prev_peak, now_basis)
                self.trailing_peak[symbol] = peak
                if activated:
                    new_sl = peak * (1 - trail_pct)
                    if new_sl > float(st.get('sl', 0) or 0):
                        st['sl'] = new_sl
            else:
                activated = (price <= entry * (1 - trigger_pct))
                prev_trough = float(self.trailing_trough.get(symbol, entry) or entry)
                now_basis = price if basis == 'close' else price
                trough = min(prev_trough, now_basis)
                self.trailing_trough[symbol] = trough
                if activated:
                    new_sl = trough * (1 + trail_pct)
                    cur_sl = float(st.get('sl', 0) or 0)
                    if cur_sl == 0 or new_sl < cur_sl:
                        st['sl'] = new_sl
        except Exception as e:
            logger.debug(f"🔧 追踪止损更新异常 {symbol}: {e}")

    def _check_hard_stop(self, symbol: str, price: float, side: str) -> bool:
        """硬止损/止盈校验"""
        try:
            st = self.sl_tp_state.get(symbol)
            if not st:
                return False
            sl = float(st.get('sl', 0) or 0)
            tp = float(st.get('tp', 0) or 0)
            if sl <= 0 or tp <= 0 or price <= 0:
                return False
            if side == 'long':
                if price <= sl or price >= tp:
                    logger.info(f"⛔ 价格触达阈值(多) {symbol}: 价={price:.6f} SL={sl:.6f} TP={tp:.6f}")
                    return True
            else:
                if price >= sl or price <= tp:
                    logger.info(f"⛔ 价格触达阈值(空) {symbol}: 价={price:.6f} SL={sl:.6f} TP={tp:.6f}")
                    return True
            return False
        except Exception as e:
            logger.debug(f"🔧 硬止损校验异常 {symbol}: {e}")
            return False

    def _maybe_partial_take_profit(self, symbol: str, price: float, atr: float, side: str) -> None:
        """占位:程序内分批止盈"""
        try:
            st = self.sl_tp_state.get(symbol)
            if not st or atr <= 0:
                return
            entry = float(st.get('entry', 0) or 0)
            profit = (price - entry) if side == 'long' else (entry - price)
            if profit > 2.0 * atr:
                tp0 = float(st.get('tp', 0) or 0)
                if tp0 > 0:
                    if side == 'long':
                        st['tp'] = entry + (tp0 - entry) * 0.9
                    else:
                        st['tp'] = entry - (entry - tp0) * 0.9
                    logger.debug(f"🎯 动态前移TP {symbol}: 新TP={st['tp']:.6f}")
        except Exception:
            pass

    def compute_sl_tp_from_levels(self, symbol: str, side: str, entry: float, atr: float = 0.0) -> tuple[float, float]:
        """基于关键位生成SL/TP"""
        try:
            if entry <= 0:
                return 0.0, 0.0
            px_prec = int(self.markets_info.get(symbol, {}).get('price_precision', 4) or 4)
            tick_sz = 10 ** (-px_prec)
            levels = {}
            try:
                cache = self.key_levels_cache.get(symbol, {})
                levels = {'supports': cache.get('supports', []), 'resistances': cache.get('resistances', [])}
                if not levels['supports'] and not levels['resistances']:
                    df = self.get_klines(symbol, 120)
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        levels = self.identify_key_levels(df)
            except Exception:
                df = self.get_klines(symbol, 120)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    try:
                        levels = self.identify_key_levels(df)
                    except Exception:
                        levels = {'supports': [], 'resistances': []}
            supports = levels.get('supports') or []
            resistances = levels.get('resistances') or []
            sup_below = [x for x in supports if float(x.get('price', 0) or 0) < entry]
            res_above = [x for x in resistances if float(x.get('price', 0) or 0) > entry]
            sup_below.sort(key=lambda x: entry - float(x.get('price', 0) or 0))
            res_above.sort(key=lambda x: float(x.get('price', 0) or 0) - entry)
            sl = 0.0
            tp = 0.0
            if side == 'long':
                if sup_below:
                    base_sup = float(sup_below[0].get('price', 0) or 0)
                    sl = base_sup * 0.995
                else:
                    sl = entry * (1 - 0.005)
                if res_above:
                    tp = float(res_above[0].get('price', 0) or 0)
                else:
                    tp = entry * (1 + 0.05)
            else:
                if res_above:
                    base_res = float(res_above[0].get('price', 0) or 0)
                    sl = base_res * 1.005
                else:
                    sl = entry * (1 + 0.005)
                if sup_below:
                    tp = float(sup_below[0].get('price', 0) or 0)
                else:
                    tp = entry * (1 - 0.05)
            min_sl = (atr * 0.8) if atr > 0 else (entry * 0.005)
            min_tp = (atr * 1.5) if atr > 0 else (entry * 0.03)
            if side == 'long':
                sl = min(sl, entry - min_sl)
                tp = max(tp, entry + min_tp)
            else:
                sl = max(sl, entry + min_sl)
                tp = min(tp, entry - min_tp)
            min_delta = max(10 * tick_sz, entry * 0.005)
            if side == 'long':
                if sl >= entry: sl = entry - min_delta
                if tp <= entry: tp = entry + min_delta
            else:
                if sl <= entry: sl = entry + min_delta
                if tp >= entry: tp = entry - min_delta
            sl = round(sl, px_prec)
            tp = round(tp, px_prec)
            sl = max(sl, tick_sz)
            tp = max(tp, tick_sz)
            if sl <= 0 or tp <= 0 or abs(tp - sl) < tick_sz:
                return 0.0, 0.0
            return sl, tp
        except Exception:
            return 0.0, 0.0

    def get_current_algo_prices(self, symbol: str) -> tuple[float, float]:
        """🔧 修复2: 获取当前交易所侧算法单的SL/TP价格"""
        try:
            inst_id = self.symbol_to_inst_id(symbol)
            resp = self.exchange.privateGetTradeOrdersAlgoPending({'instType': 'SWAP', 'instId': inst_id})
            data = resp.get('data') if isinstance(resp, dict) else resp
            
            sl_price = 0.0
            tp_price = 0.0
            
            for it in (data or []):
                try:
                    ord_type = str(it.get('ordType', '')).lower()
                    if ord_type == 'oco':
                        # OCO订单包含TP和SL
                        tp_price = float(it.get('tpTriggerPx', 0) or 0)
                        sl_price = float(it.get('slTriggerPx', 0) or 0)
                        break
                except Exception:
                    continue
            
            return sl_price, tp_price
        except Exception as e:
            logger.debug(f"🔧 获取当前算法单价格失败 {symbol}: {e}")
            return 0.0, 0.0

    def place_okx_tp_sl(self, symbol: str, entry: float, side: str, atr: float = 0.0) -> bool:
        """🔧 修复2: 挂OKX侧TP/SL,支持条件性更新追踪止损"""
        try:
            inst_id = self.symbol_to_inst_id(symbol)
            if not inst_id:
                return False

            pos = self.get_position(symbol, force_refresh=True)
            if not pos or float(pos.get('size', 0) or 0) <= 0:
                logger.warning(f"⚠️ 无持仓,跳过交易所侧TP/SL {symbol}")
                return False

            st = self.sl_tp_state.get(symbol, {})
            sl = float(st.get('sl', 0.0) or 0.0)
            tp = float(st.get('tp', 0.0) or 0.0)

            if (sl <= 0 or tp <= 0) and entry > 0:
                try:
                    adj = self.get_learning_adjustments(symbol)
                    use_w = float(adj.get('use_levels_weight', 0.6) or 0.6)
                except Exception:
                    use_w = 0.6
                sl2, tp2 = self.compute_sl_tp_from_levels(symbol, side, entry, atr)
                if sl2 > 0 and tp2 > 0 and use_w >= 0.5:
                    sl, tp = sl2, tp2
                    logger.info(f"🔧 关键位生成SL/TP {symbol}: entry={entry:.6f} → SL={sl:.6f} TP={tp:.6f}")
                else:
                    base_sl = max(entry * 0.005, atr if atr > 0 else entry * 0.003)
                    base_tp = max(entry * 0.03, (atr * 2.0) if atr > 0 else entry * 0.02)
                    if side == 'long':
                        sl = entry - base_sl
                        tp = entry + base_tp
                    else:
                        sl = entry + base_sl
                        tp = entry - base_tp
                    logger.info(f"🔧 ATR/比例生成SL/TP {symbol}: entry={entry:.6f} atr={atr:.6f} → SL={sl:.6f} TP={tp:.6f}")

            try:
                boost = float(self.tp_boost_map.get(symbol, 1.0) or 1.0)
                if side == 'long' and boost > 1.0:
                    tp *= boost
            except Exception:
                pass

            px_prec = int(self.markets_info.get(symbol, {}).get('price_precision', 4) or 4)
            tick_sz = 10 ** (-px_prec)
            min_ticks = int(self.tp_sl_min_delta_ticks or 1)
            min_delta = max(10 * tick_sz, entry * 0.005)

            def _round_px(x: float) -> float:
                return round(x, px_prec)

            # 获取最新价
            last = 0.0
            try:
                tkr = self.exchange.publicGetMarketTicker({'instId': inst_id})
                if isinstance(tkr, dict):
                    d = tkr.get('data') or []
                    if isinstance(d, list) and d:
                        last = float(d[0].get('last') or d[0].get('lastPx') or 0.0)
            except Exception as _e:
                logger.warning(f"⚠️ 获取最新价失败 {symbol}: {_e}")
            if last <= 0:
                last = max(0.0, float(entry or 0.0))
            if last <= tick_sz:
                try:
                    df_last = self.get_klines(symbol, 20)
                    if df_last is not None and not df_last.empty:
                        last = float(df_last['close'].values[-1])
                except Exception:
                    pass
            if last <= tick_sz:
                if float(entry or 0.0) > tick_sz:
                    last = float(entry)
                else:
                    logger.warning(f"⚠️ 无有效价格参考,跳过 {symbol}")
                    return False

            if side == 'long':
                tp = max(tp, last + min_delta)
                sl = min(sl, last - min_delta)
            else:
                tp = min(tp, last - min_delta)
                sl = max(sl, last + min_delta)

            tp = _round_px(tp)
            sl = _round_px(sl)
            tp = max(tp, tick_sz)
            sl = max(sl, tick_sz)
            min_sep = max(10 * tick_sz, last * 0.005)
            if side == 'long':
                if tp - sl < min_sep:
                    tp = _round_px(max(tp, sl + min_sep))
            else:
                if sl - tp < min_sep:
                    sl = _round_px(max(sl, tp + min_sep))
            if side == 'long':
                if sl >= last:
                    sl = _round_px(max(tick_sz, last - max(min_delta, min_sep) - tick_sz))
                if tp <= last:
                    tp = _round_px(last + max(min_delta, min_sep) + tick_sz)
            else:
                if sl <= last:
                    sl = _round_px(last + max(min_delta, min_sep) + tick_sz)
                if tp >= last:
                    tp = _round_px(max(tick_sz, last - max(min_delta, min_sep) - tick_sz))
            if side == 'long':
                if sl >= last:
                    sl = _round_px(max(tick_sz, last - max(min_delta, min_sep) - 5 * tick_sz))
                if tp <= last:
                    tp = _round_px(last + max(min_delta, min_sep) + 5 * tick_sz)
            else:
                if sl <= last:
                    sl = _round_px(last + max(min_delta, min_sep) + 5 * tick_sz)
                if tp >= last:
                    tp = _round_px(max(tick_sz, last - max(min_delta, min_sep) - 5 * tick_sz))
            if tp <= 0 or sl <= 0 or tp == sl:
                logger.warning(f"⚠️ 触发价无效,跳过 {symbol}: last={last:.6f} tp={tp:.6f} sl={sl:.6f}")
                return False

            # 🔧 修复2: 检查是否需要更新(追踪止损优化)
            try:
                resp_pending = self.exchange.privateGetTradeOrdersAlgoPending({'instType': 'SWAP', 'instId': inst_id})
                pend = resp_pending.get('data') if isinstance(resp_pending, dict) else resp_pending
                has_algo = False
                for it in (pend or []):
                    aid = (it.get('algoId') or it.get('algoID') or it.get('id'))
                    if aid:
                        has_algo = True
                        break
                
                if has_algo:
                    # 🔧 修复2: 获取当前算法单的SL/TP价格
                    existing_sl, existing_tp = self.get_current_algo_prices(symbol)
                    
                    # 判断是否需要更新
                    should_update = False
                    
                    if existing_sl > 0:
                        # 如果新的SL更优(多头:更高;空头:更低),则更新
                        if side == 'long' and sl > existing_sl * 1.001:  # 新SL至少高0.1%
                            should_update = True
                            logger.info(f"🔄 追踪止损优化(多头) {symbol}: 旧SL={existing_sl:.6f} → 新SL={sl:.6f}")
                        elif side == 'short' and sl < existing_sl * 0.999:  # 新SL至少低0.1%
                            should_update = True
                            logger.info(f"🔄 追踪止损优化(空头) {symbol}: 旧SL={existing_sl:.6f} → 新SL={sl:.6f}")
                    
                    if should_update:
                        # 撤旧挂新
                        logger.info(f"🔄 检测到追踪止损优化,更新TP/SL {symbol}")
                        cancel_ok = self.cancel_symbol_tp_sl(symbol)
                        if not cancel_ok:
                            logger.warning(f"⚠️ 撤销旧订单失败,跳过更新 {symbol}")
                            return False
                        time.sleep(0.5)  # 等待撤销完成
                    else:
                        logger.info(f"ℹ️ 已有交易所侧TP/SL且无需优化,跳过重挂 {symbol}")
                        return True
            except Exception as e:
                logger.debug(f"🔧 检查现有订单异常 {symbol}: {e}")

            # OCO参数
            def _submit_oco(use_posside: bool = True):
                params_oco = {
                    'instId': inst_id,
                    'ordType': 'oco',
                    'side': 'sell' if side == 'long' else 'buy',
                    'tdMode': ('isolated' if str(pos.get('margin_mode', 'cross')).lower() == 'isolated' else 'cross'),
                    'reduceOnly': True,
                    'tpTriggerPx': str(tp),
                    'tpOrdPx': '-1',
                    'slTriggerPx': str(sl),
                    'slOrdPx': '-1',
                    'closeFraction': '1',
                }
                try:
                    if self.get_position_mode() == 'hedge':
                        ps = str(pos.get('side', 'long') or 'long')
                        params_oco['posSide'] = 'long' if ps == 'long' else 'short'
                except Exception:
                    pass
                resp = self.exchange.privatePostTradeOrderAlgo(params_oco)
                data = resp.get('data', []) if isinstance(resp, dict) else []
                item = data[0] if (isinstance(data, list) and data) else {}
                s_code = str(item.get('sCode', '1'))
                s_msg = str(item.get('sMsg', '') or '')
                return s_code, s_msg

            try:
                s_code, s_msg = _submit_oco(use_posside=True)
                if s_code == '0':
                    pass
                elif s_code == '51088':
                    logger.debug(f"ℹ️ 已存在整仓TP/SL,视为成功 {symbol}: code={s_code} msg={s_msg}")
                elif s_code == '51023':
                    logger.warning(f"⚠️ 挂OCO失败(51023) {symbol}: {s_msg}")
                    return False
                else:
                    logger.warning(f"⚠️ 挂OCO失败 {symbol}: code={s_code} msg={s_msg}")
                    return False
            except Exception as e:
                emsg = str(e)
                if '51088' in emsg:
                    logger.debug(f"ℹ️ 已存在整仓TP/SL(异常返回),视为成功 {symbol}: {emsg}")
                elif '51023' in emsg:
                    logger.warning(f"⚠️ 挂OCO失败(51023异常) {symbol}: {emsg}")
                    return False
                else:
                    logger.warning(f"⚠️ 挂OCO异常 {symbol}: {e}")
                    return False

            self.okx_tp_sl_placed[symbol] = True
            self.tp_sl_last_placed[symbol] = time.time()
            logger.info(f"✅ 挂OCO成功 {symbol}: side={side} last={last:.6f} SL={sl:.6f} TP={tp:.6f}")
            return True

        except Exception as e:
            logger.error(f"❌ 挂TP/SL失败 {symbol}: {e}")
            return False
        finally:
            try:
                self._track_position_stats()
            except Exception as _estat2:
                logger.debug(f"🔧 统计监听末尾异常: {_estat2}")
    
    def calculate_volatility(self, df):
        """计算波动率"""
        returns = df['close'].pct_change()
        volatility = returns.std() * 100
        return volatility
    
    def calculate_indicators(self, df, symbol):
        """计算技术指标"""
        macd_p = self.macd_params.get(symbol, {'fast': 6, 'slow': 16, 'signal': 9})
        rsi_p = self.rsi_params.get(symbol, 9)
        
        ema_fast = df['close'].ewm(span=macd_p['fast'], adjust=False).mean()
        ema_slow = df['close'].ewm(span=macd_p['slow'], adjust=False).mean()
        df['macd_diff'] = ema_fast - ema_slow
        df['macd_dea'] = df['macd_diff'].ewm(span=macd_p['signal'], adjust=False).mean()
        df['macd_histogram'] = df['macd_diff'] - df['macd_dea']
        
        df = calculate_rsi(df, rsi_p)
        
        df['volume_ma'] = df['volume'].rolling(window=5).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        df['volatility'] = df['close'].pct_change().rolling(window=20).std() * 100
        
        df = calculate_atr(df, 14)
        
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        return df
    
    def detect_divergence(self, df, lookback=25):
        """背离检测"""
        if len(df) < lookback:
            return {'bullish': False, 'bearish': False, 'strength': 0}
        
        recent_df = df.tail(lookback)
        
        price_lows = []
        macd_lows = []
        price_highs = []
        macd_highs = []
        
        for i in range(3, len(recent_df) - 3):
            if (recent_df.iloc[i]['low'] < recent_df.iloc[i-1]['low'] and 
                recent_df.iloc[i]['low'] < recent_df.iloc[i-2]['low'] and
                recent_df.iloc[i]['low'] < recent_df.iloc[i-3]['low'] and
                recent_df.iloc[i]['low'] < recent_df.iloc[i+1]['low'] and
                recent_df.iloc[i]['low'] < recent_df.iloc[i+2]['low'] and
                recent_df.iloc[i]['low'] < recent_df.iloc[i+3]['low']):
                price_lows.append((i, recent_df.iloc[i]['low']))
                macd_lows.append((i, recent_df.iloc[i]['macd_diff']))
            
            if (recent_df.iloc[i]['high'] > recent_df.iloc[i-1]['high'] and 
                recent_df.iloc[i]['high'] > recent_df.iloc[i-2]['high'] and
                recent_df.iloc[i]['high'] > recent_df.iloc[i-3]['high'] and
                recent_df.iloc[i]['high'] > recent_df.iloc[i+1]['high'] and
                recent_df.iloc[i]['high'] > recent_df.iloc[i+2]['high'] and
                recent_df.iloc[i]['high'] > recent_df.iloc[i+3]['high']):
                price_highs.append((i, recent_df.iloc[i]['high']))
                macd_highs.append((i, recent_df.iloc[i]['macd_diff']))
        
        bullish_div = False
        div_strength = 0
        if len(price_lows) >= 2 and len(macd_lows) >= 2:
            last_price_low = price_lows[-1][1]
            prev_price_low = price_lows[-2][1]
            last_macd_low = macd_lows[-1][1]
            prev_macd_low = macd_lows[-2][1]
            
            if last_price_low < prev_price_low and last_macd_low > prev_macd_low:
                bullish_div = True
                price_change = (prev_price_low - last_price_low) / prev_price_low
                macd_change = (last_macd_low - prev_macd_low) / abs(prev_macd_low)
                div_strength = (price_change + macd_change) * 100
        
        bearish_div = False
        if len(price_highs) >= 2 and len(macd_highs) >= 2:
            last_price_high = price_highs[-1][1]
            prev_price_high = price_highs[-2][1]
            last_macd_high = macd_highs[-1][1]
            prev_macd_high = macd_highs[-2][1]
            
            if last_price_high > prev_price_high and last_macd_high < prev_macd_high:
                bearish_div = True
                price_change = (last_price_high - prev_price_high) / prev_price_high
                macd_change = (prev_macd_high - last_macd_high) / abs(prev_macd_high)
                div_strength = (price_change + macd_change) * 100
        
        return {
            'bullish': bullish_div, 
            'bearish': bearish_div, 
            'strength': div_strength
        }
    
    def check_trend(self, df):
        """趋势识别"""
        latest = df.iloc[-1]
        
        ema_trend = 'up' if latest['ema_9'] > latest['ema_20'] > latest['ema_50'] else \
                   ('down' if latest['ema_9'] < latest['ema_20'] < latest['ema_50'] else 'neutral')
        
        macd_trend = 'up' if latest['macd_diff'] > 0 and latest['macd_histogram'] > 0 else \
                    ('down' if latest['macd_diff'] < 0 and latest['macd_histogram'] < 0 else 'neutral')
        
        price_position = 'above' if latest['close'] > latest['ema_20'] else 'below'
        
        return {
            'ema_trend': ema_trend,
            'macd_trend': macd_trend,
            'price_position': price_position,
            'strong_trend': ema_trend == macd_trend and ema_trend != 'neutral'
        }
    
    def get_category(self, symbol: str) -> str:
        """返回币种分类"""
        try:
            for cat, lst in (self.coin_categories or {}).items():
                if symbol in lst:
                    return cat
        except Exception:
            pass
        return 'unknown'

    def get_tick_size(self, symbol: str) -> float:
        """根据price_precision返回最小跳动单位"""
        try:
            px_prec = int(self.markets_info.get(symbol, {}).get('price_precision', 4) or 4)
            return 10 ** (-px_prec)
        except Exception:
            return 0.0001

    def check_long_signal(self, df, symbol):
        """优化版做多信号检测"""
        if len(df) < 5:
            return False, "数据不足", 0
        
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        
        thresholds = self.rsi_thresholds.get(symbol, {'overbought': 70, 'oversold': 30})
        divergence = self.detect_divergence(df)
        trend = self.check_trend(df)
        category = self.get_category(symbol)
        
        signal_strength = 0
        
        if divergence['bullish']:
            if (latest['rsi'] < thresholds['oversold'] + 10 and 
                latest['rsi'] > previous['rsi'] and
                latest['macd_histogram'] > previous['macd_histogram']):
                signal_strength = 90 + min(divergence['strength'], 10)
                return True, f"🔥底背离(强度{divergence['strength']:.1f})", signal_strength
        
        golden_cross = (
            previous['macd_diff'] <= previous['macd_dea'] and
            latest['macd_diff'] > latest['macd_dea'] and
            latest['macd_histogram'] > 0
        )
        
        if golden_cross:
            if latest['macd_diff'] < 0:
                if (latest['rsi'] > thresholds['oversold'] and 
                    latest['rsi'] < 50 and
                    latest['volume_ratio'] > 1.2):
                    signal_strength = 75
                    return True, "MACD零轴下金叉(抄底)", signal_strength
            elif trend['strong_trend'] and trend['ema_trend'] == 'up':
                if latest['rsi'] > 50:
                    signal_strength = 80
                    return True, "MACD零轴上金叉(趋势)", signal_strength
        
        if (previous['macd_diff'] < 0 and latest['macd_diff'] > 0 and
            latest['rsi'] > 50 and trend['price_position'] == 'above'):
            signal_strength = 70
            return True, "MACD零轴突破", signal_strength
        
        if category == 'meme':
            if (latest['rsi'] < thresholds['oversold'] and
                latest['rsi'] > previous['rsi'] and
                latest['macd_histogram'] > previous['macd_histogram'] and
                latest['volume_ratio'] > 1.5):
                signal_strength = 65
                return True, "RSI超卖反弹(MEME)", signal_strength
        
        return False, "", 0
    
    def check_short_signal(self, df, symbol):
        """优化版做空信号检测"""
        if len(df) < 5:
            return False, "数据不足", 0
        
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        
        thresholds = self.rsi_thresholds.get(symbol, {'overbought': 70, 'oversold': 30})
        divergence = self.detect_divergence(df)
        trend = self.check_trend(df)
        category = self.get_category(symbol)
        
        signal_strength = 0
        
        if divergence['bearish']:
            if (latest['rsi'] > thresholds['overbought'] - 10 and
                latest['rsi'] < previous['rsi'] and
                latest['macd_histogram'] < previous['macd_histogram']):
                signal_strength = 90 + min(divergence['strength'], 10)
                return True, f"🔥顶背离(强度{divergence['strength']:.1f})", signal_strength
        
        death_cross = (
            previous['macd_diff'] >= previous['macd_dea'] and
            latest['macd_diff'] < latest['macd_dea'] and
            latest['macd_histogram'] < 0
        )
        
        if death_cross:
            if latest['macd_diff'] > 0:
                if (latest['rsi'] < thresholds['overbought'] and
                    latest['rsi'] > 50 and
                    latest['volume_ratio'] > 1.2):
                    signal_strength = 75
                    return True, "MACD零轴上死叉(逃顶)", signal_strength
            elif trend['strong_trend'] and trend['ema_trend'] == 'down':
                if latest['rsi'] < 50:
                    signal_strength = 80
                    return True, "MACD零轴下死叉(趋势)", signal_strength
        
        if (previous['macd_diff'] > 0 and latest['macd_diff'] < 0 and
            latest['rsi'] < 50 and trend['price_position'] == 'below'):
            signal_strength = 70
            return True, "MACD零轴下破", signal_strength
        
        if category == 'meme':
            if (latest['rsi'] > thresholds['overbought'] and
                latest['rsi'] < previous['rsi'] and
                latest['macd_histogram'] < previous['macd_histogram'] and
                latest['volume_ratio'] > 1.5):
                signal_strength = 65
                return True, "RSI超买反弹(MEME)", signal_strength
        
        return False, "", 0
    
    def calculate_position_size(self, symbol, entry_price, signal_strength):
        """动态仓位计算"""
        try:
            balance = self.get_account_balance()
            usdt_balance = balance
            
            base_risk = usdt_balance * (self.risk_percent / 100)
            
            weight = self.position_weights.get(symbol, 1.0)
            
            strength_multiplier = 0.8 + (signal_strength - 60) / 100
            strength_multiplier = max(0.8, min(1.2, strength_multiplier))
            
            adjusted_risk = base_risk * weight * strength_multiplier
            
            stop_loss_percent = self.stop_loss.get(symbol, 3.0)
            position_size = adjusted_risk / (entry_price * stop_loss_percent / 100)
            
            return position_size, strength_multiplier
        except Exception as e:
            logger.error(f"❌ 计算仓位失败: {e}")
            return 0, 1.0
    
    def open_position(self, symbol, side, df, reason, signal_strength):
        """开仓"""
        try:
            latest = df.iloc[-1]
            entry_price = latest['close']
            category = self.get_category(symbol)
            
            position_size, multiplier = self.calculate_position_size(symbol, entry_price, signal_strength)
            
            if position_size <= 0:
                logger.warning(f"⚠️ 仓位计算错误,跳过 {symbol}")
                return
            
            if side == 'buy':
                stop_loss_price = entry_price * (1 - self.stop_loss[symbol] / 100)
                take_profit_prices = [
                    entry_price * (1 + tp / 100) 
                    for tp in self.take_profit[symbol]
                ]
            else:
                stop_loss_price = entry_price * (1 + self.stop_loss[symbol] / 100)
                take_profit_prices = [
                    entry_price * (1 - tp / 100) 
                    for tp in self.take_profit[symbol]
                ]
            
            self.positions[symbol] = {
                'side': side,
                'entry_price': entry_price,
                'size': position_size,
                'stop_loss': stop_loss_price,
                'take_profits': take_profit_prices,
                'tp_filled': [False, False, False],
                'entry_time': datetime.datetime.now(),
                'entry_reason': reason,
                'signal_strength': signal_strength,
                'category': category,
                'macd_diff': latest['macd_diff'],
                'rsi': latest['rsi']
            }
            
            emoji = '📈' if side == 'buy' else '📉'
            category_emoji = {'blue_chip': '💎', 'mainnet': '⛓️', 'infrastructure': '🏗️', 
                            'emerging': '🌱', 'meme': '🐸'}.get(category, '❓')
            
            logger.info(f"\n{'='*70}")
            logger.info(f"✅ {category_emoji} 开仓成功!")
            logger.info(f"币种: {symbol} ({category.upper()})")
            logger.info(f"方向: {emoji} {'做多' if side == 'buy' else '做空'}")
            logger.info(f"策略: {reason}")
            logger.info(f"信号强度: {signal_strength:.0f}/100 (仓位倍数: {multiplier:.2f}x)")
            logger.info(f"入场价: ${entry_price:.6f}")
            logger.info(f"仓位: {position_size:.4f}")
            logger.info(f"MACD: {latest['macd_diff']:.4f} | RSI: {latest['rsi']:.1f} | 成交量比: {latest['volume_ratio']:.2f}x")
            logger.info(f"止损: ${stop_loss_price:.6f} (-{self.stop_loss[symbol]:.1f}%)")
            logger.info(f"止盈: TP1=${take_profit_prices[0]:.6f}, TP2=${take_profit_prices[1]:.6f}, TP3=${take_profit_prices[2]:.6f}")
            logger.info(f"{'='*70}\n")
            
        except Exception as e:
            logger.error(f"❌ 开仓失败 {symbol}: {e}")
    
    def manage_positions(self):
        """持仓管理"""
        for symbol, pos in list(self.positions.items()):
            try:
                import random
                current_price = pos['entry_price'] * (1 + random.uniform(-0.03, 0.03))
                
                pnl_percent = 0
                if pos['side'] == 'buy':
                    pnl_percent = (current_price - pos['entry_price']) / pos['entry_price'] * 100
                    
                    if current_price <= pos['stop_loss']:
                        self.close_position(symbol, f"止损 ({pnl_percent:.2f}%)", pnl_percent)
                        continue
                    
                    if current_price >= pos['take_profits'][0] and pos['stop_loss'] < pos['entry_price']:
                        pos['stop_loss'] = pos['entry_price'] * 0.998
                        logger.info(f"📌 移动止损: {symbol} 止损移至保本价 ${pos['stop_loss']:.6f}")
                    
                    for i, tp_price in enumerate(pos['take_profits']):
                        if not pos['tp_filled'][i] and current_price >= tp_price:
                            self.partial_close(symbol, i, pnl_percent)
                else:
                    pnl_percent = (pos['entry_price'] - current_price) / pos['entry_price'] * 100
                    
                    if current_price >= pos['stop_loss']:
                        self.close_position(symbol, f"止损 ({pnl_percent:.2f}%)", pnl_percent)
                        continue
                    
                    if current_price <= pos['take_profits'][0] and pos['stop_loss'] > pos['entry_price']:
                        pos['stop_loss'] = pos['entry_price'] * 1.002
                        logger.info(f"📌 移动止损: {symbol} 止损移至保本价 ${pos['stop_loss']:.6f}")
                    
                    for i, tp_price in enumerate(pos['take_profits']):
                        if not pos['tp_filled'][i] and current_price <= tp_price:
                            self.partial_close(symbol, i, pnl_percent)
                            
            except Exception as e:
                logger.error(f"❌ 管理持仓失败 {symbol}: {e}")
    
    def partial_close(self, symbol, tp_index, pnl_percent):
        """分批止盈"""
        pos = self.positions[symbol]
        close_ratios = [0.5, 0.3, 0.2]
        
        try:
            close_size = pos['size'] * close_ratios[tp_index]
            side = 'sell' if pos['side'] == 'buy' else 'buy'
            
            pos['tp_filled'][tp_index] = True
            pos['size'] -= close_size
            
            logger.info(f"💰 止盈TP{tp_index+1}: {symbol}, 平仓{close_ratios[tp_index]*100:.0f}%, 当前盈利{pnl_percent:.2f}%")
            
            if all(pos['tp_filled']):
                self.trade_stats[symbol]['wins'] += 1
                self.trade_stats[symbol]['total_pnl'] += pnl_percent
                del self.positions[symbol]
                logger.info(f"✅ 完全平仓: {symbol}, 总盈利{pnl_percent:.2f}%")
                self.print_stats()
                
        except Exception as e:
            logger.error(f"❌ 分批止盈失败: {e}")
    
    def close_position(self, symbol, reason, pnl_percent):
        """完全平仓"""
        try:
            pos = self.positions[symbol]
            side = 'sell' if pos['side'] == 'buy' else 'buy'
            
            if pnl_percent < 0:
                self.trade_stats[symbol]['losses'] += 1
            else:
                self.trade_stats[symbol]['wins'] += 1
            
            self.trade_stats[symbol]['total_pnl'] += pnl_percent
            
            del self.positions[symbol]
            
            emoji = "🔴" if pnl_percent < 0 else "🟢"
            logger.info(f"{emoji} 平仓: {symbol} - {reason}")
            try:
                self.update_learning_state(symbol, float(pnl_percent))
            except Exception:
                pass
            
        except Exception as e:
            logger.error(f"❌ 平仓失败: {e}")
    
    def print_stats(self):
        """打印统计信息"""
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 交易统计")
        logger.info(f"{'='*70}")
        
        total_wins = sum(s['wins'] for s in self.trade_stats.values())
        total_losses = sum(s['losses'] for s in self.trade_stats.values())
        total_trades = total_wins + total_losses
        win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
        total_pnl = sum(s['total_pnl'] for s in self.trade_stats.values())
        
        logger.info(f"总交易: {total_trades} | 胜: {total_wins} | 负: {total_losses} | 胜率: {win_rate:.1f}%")
        logger.info(f"总盈亏: {total_pnl:+.2f}%")
        logger.info(f"\n各币种表现:")
        
        for symbol, stats in sorted(self.trade_stats.items(), key=lambda x: x[1]['total_pnl'], reverse=True):
            if stats['wins'] + stats['losses'] > 0:
                symbol_wr = stats['wins'] / (stats['wins'] + stats['losses']) * 100
                category = self.get_category(symbol)
                logger.info(f"  {symbol:12} | 胜率:{symbol_wr:5.1f}% | 盈亏:{stats['total_pnl']:+6.2f}% | 类型:{category}")
        
        logger.info(f"{'='*70}\n")
    
    def is_trading_time(self):
        """交易时段判断"""
        now = datetime.datetime.now()
        hour = now.hour
        
        avoid_hours = list(range(0, 2)) + list(range(8, 10))
        
        if now.weekday() >= 5:
            return False
        
        return hour not in avoid_hours
    
    def adaptive_parameter_adjustment(self, symbol, df):
        """自适应参数调整"""
        volatility = self.calculate_volatility(df)
        
        avg_volatility = df['volatility'].tail(50).mean()
        if volatility > avg_volatility * 1.5:
            adjusted_sl = self.stop_loss[symbol] * 1.3
            logger.info(f"⚠️ {symbol} 波动率异常 ({volatility:.2f}% vs {avg_volatility:.2f}%), 止损放宽至 {adjusted_sl:.1f}%")
            return adjusted_sl
        
        return self.stop_loss[symbol]
    
    def check_correlation(self):"""检查币种相关性"""
        if len(self.positions) < 2:
            return True
        
        meme_count = sum(1 for pos in self.positions.values() if pos['category'] == 'meme')
        if meme_count >= 3:
            logger.info(f"⚠️ MEME币持仓过多 ({meme_count}/3),暂停新的MEME币交易")
            return False
        
        return True
    
    def analyze_symbol(self, symbol: str) -> Dict[str, str]:
        """分析符号信号(加入1H趋势门控)"""
        try:
            df = self.get_klines(symbol, 150)
            if df.empty or len(df) < 50:
                return {'signal': 'hold', 'reason': '数据不足'}
            
            df = self.calculate_indicators(df, symbol)
            current_position = self.get_position(symbol, force_refresh=False)
            
            # 1H趋势门控
            try:
                inst_id = self.symbol_to_inst_id(symbol)
                resp1h = self.exchange.publicGetMarketCandles({'instId': inst_id, 'bar': '1H', 'limit': '120'})
                rows1h = resp1h.get('data') if isinstance(resp1h, dict) else resp1h
                hist1h = []
                for r in (rows1h or []):
                    ts = int(r[0]); c = float(r[4])
                    hist1h.append({'timestamp': pd.to_datetime(ts, unit='ms'), 'close': c})
                hist1h.sort(key=lambda x: x['timestamp'])
                df1h = pd.DataFrame(hist1h)
                allow_long = True
                allow_short = True
                min_long_strength = 65
                min_short_strength = 65
                if not df1h.empty and len(df1h) >= 35:
                    macd_p = self.macd_params.get(symbol, {'fast': 6, 'slow': 16, 'signal': 9})
                    ema_fast_1h = df1h['close'].ewm(span=macd_p['fast'], adjust=False).mean()
                    ema_slow_1h = df1h['close'].ewm(span=macd_p['slow'], adjust=False).mean()
                    macd_diff_1h = ema_fast_1h - ema_slow_1h
                    macd_dea_1h = macd_diff_1h.ewm(span=macd_p['signal'], adjust=False).mean()
                    rsi_win = self.rsi_params.get(symbol, 9)
                    delta = df1h['close'].diff()
                    up = delta.clip(lower=0)
                    down = -delta.clip(upper=0)
                    ema_up = up.ewm(com=rsi_win-1, min_periods=rsi_win).mean()
                    ema_down = down.ewm(com=rsi_win-1, min_periods=rsi_win).mean()
                    rs = ema_up / ema_down.replace(0, np.nan)
                    rsi1h = 100 - (100 / (1 + rs))
                    latest_diff_1h = float(macd_diff_1h.iloc[-1] if hasattr(macd_diff_1h, 'iloc') else macd_diff_1h)
                    latest_dea_1h = float(macd_dea_1h.iloc[-1] if hasattr(macd_dea_1h, 'iloc') else macd_dea_1h)
                    latest_rsi_1h = float(rsi1h.iloc[-1] if hasattr(rsi1h, 'iloc') else rsi1h)
                    bullish_1h = (latest_diff_1h > latest_dea_1h and latest_rsi_1h > 50)
                    bearish_1h = (latest_diff_1h < latest_dea_1h and latest_rsi_1h < 50)
                    if bullish_1h:
                        min_long_strength = 60
                        self.tp_boost_map[symbol] = 1.5
                    elif bearish_1h:
                        allow_long = False
                        self.tp_boost_map[symbol] = 1.0
                    else:
                        self.tp_boost_map[symbol] = 1.0
                else:
                    allow_long = True
                    allow_short = True
                    min_long_strength = 65
                    min_short_strength = 65
            except Exception:
                allow_long = True
                allow_short = True
                min_long_strength = 65
                min_short_strength = 65
            
            if current_position['size'] == 0:
                long_signal, long_reason, long_strength = self.check_long_signal(df, symbol)
                short_signal, short_reason, short_strength = self.check_short_signal(df, symbol)
                
                mode = self.strategy_mode_map.get(symbol, 'combo')
                def _is_mode_ok(reason: str, side: str) -> bool:
                    r = reason or ''
                    if mode == 'zero_cross':
                        return ('零轴突破' in r) if side == 'buy' else ('零轴下破' in r)
                    if mode == 'divergence':
                        return ('背离' in r)
                    if mode == 'golden_cross':
                        return ('金叉' in r) if side == 'buy' else ('死叉' in r)
                    return True
                if long_signal and allow_long and long_strength >= min_long_strength and _is_mode_ok(long_reason, 'buy'):
                    if float(self.tp_boost_map.get(symbol, 1.0) or 1.0) > 1.0:
                        logger.info(f"🌟 1H多头趋势:{symbol} 做多TP目标已放大至 1.5x")
                    return {'signal': 'buy', 'reason': long_reason}
                if short_signal and allow_short and short_strength >= min_short_strength and _is_mode_ok(short_reason, 'sell'):
                    return {'signal': 'sell', 'reason': short_reason}
                return {'signal': 'hold', 'reason': '无信号'}
            else:
                if current_position['side'] == 'long':
                    short_signal, short_reason, short_strength = self.check_short_signal(df, symbol)
                    if short_signal and short_strength >= 65:
                        return {'signal': 'close', 'reason': short_reason}
                else:
                    long_signal, long_reason, long_strength = self.check_long_signal(df, symbol)
                    if long_signal and long_strength >= 65:
                        return {'signal': 'close', 'reason': long_reason}
                return {'signal': 'hold', 'reason': '持仓中'}
            
        except Exception as e:
            logger.error(f"❌ 分析{symbol}失败: {e}")
            return {'signal': 'hold', 'reason': f'分析异常: {e}'}
    
    def ensure_tpsl_guard(self) -> None:
        """守护:逐币检查持仓,若交易所侧无TP/SL则立即补挂"""
        try:
            for symbol in self.symbols:
                try:
                    pos = self.get_position(symbol, force_refresh=True)
                    if not pos or float(pos.get('size', 0) or 0) <= 0:
                        continue
                    inst_id = self.symbol_to_inst_id(symbol)
                    has_algo = False
                    try:
                        resp = self.exchange.privateGetTradeOrdersAlgoPending({'instType': 'SWAP', 'instId': inst_id})
                        pend = resp.get('data') if isinstance(resp, dict) else resp
                        for it in (pend or []):
                            if (it.get('algoId') or it.get('algoID') or it.get('id')):
                                has_algo = True
                                break
                    except Exception:
                        has_algo = False
                    if has_algo:
                        continue
                    entry0 = float(pos.get('entry_price', 0) or 0)
                    if entry0 <= 0:
                        try:
                            df_last = self.get_klines(symbol, 20)
                            if df_last is not None and not df_last.empty:
                                entry0 = float(df_last['close'].values[-1])
                        except Exception:
                            pass
                        if entry0 <= 0:
                            try:
                                inst_id2 = self.symbol_to_inst_id(symbol)
                                tkr2 = self.exchange.publicGetMarketTicker({'instId': inst_id2})
                                d2 = tkr2.get('data') if isinstance(tkr2, dict) else tkr2
                                if isinstance(d2, list) and d2:
                                    entry0 = float(d2[0].get('last') or d2[0].get('lastPx') or 0.0)
                            except Exception:
                                entry0 = 0.0
                        if entry0 <= 0:
                            continue
                    try:
                        kl = self.get_klines(symbol, 50)
                        if kl is not None and not kl.empty:
                            ps = self.per_symbol_params.get(symbol, {})
                            atr_p = int(ps.get('atr_period', 14))
                            atr_val = calculate_atr(kl, atr_p)['atr'].iloc[-1]
                        else:
                            atr_val = 0.0
                    except Exception:
                        atr_val = 0.0
                    st0 = self.sl_tp_state.get(symbol)
                    if not st0 and atr_val > 0 and entry0 > 0:
                        self._set_initial_sl_tp(symbol, entry0, atr_val, pos.get('side', 'long'))
                    ok = self.place_okx_tp_sl(symbol, entry0, pos.get('side', 'long'), atr_val)
                    if ok:
                        logger.info(f"📌 守护补挂TP/SL成功 {symbol}")
                    else:
                        logger.warning(f"⚠️ 守护补挂TP/SL失败 {symbol}")
                except Exception as _e:
                    logger.debug(f"🔧 守护检查异常 {symbol}: {_e}")
        except Exception as e:
            logger.warning(f"⚠️ 守护执行异常: {e}")

    def _track_position_stats(self) -> None:
        """监听持仓变化:若从>0变为0,补记一笔成交统计"""
        try:
            for symbol in self.symbols:
                pos = self.get_position(symbol, force_refresh=True)
                cur_size = float((pos or {}).get('size', 0) or 0)
                cur_side = str((pos or {}).get('side', '') or '')
                cur_entry = float((pos or {}).get('entry_price', 0) or 0)
                prev = self.prev_positions.get(symbol, {'size': 0.0, 'side': '', 'entry': 0.0})
                prev_size = float(prev.get('size', 0) or 0)
                prev_side = str(prev.get('side', '') or '')
                prev_entry = float(prev.get('entry', 0) or 0)

                if prev_size > 0 and cur_size <= 0:
                    close_px = 0.0
                    try:
                        inst_id = self.symbol_to_inst_id(symbol)
                        tkr = self.exchange.publicGetMarketTicker({'instId': inst_id})
                        d = tkr.get('data') if isinstance(tkr, dict) else tkr
                        if isinstance(d, list) and d:
                            close_px = float(d[0].get('last') or d[0].get('lastPx') or 0.0)
                    except Exception:
                        close_px = 0.0
                    if close_px <= 0:
                        try:
                            df_last = self.get_klines(symbol, 20)
                            if isinstance(df_last, pd.DataFrame) and not df_last.empty:
                                close_px = float(df_last['close'].values[-1])
                        except Exception:
                            close_px = 0.0

                    entry_px = prev_entry if prev_entry > 0 else cur_entry
                    side_use = prev_side if prev_side else (cur_side or 'long')
                    pnl_u = 0.0
                    pnl_pct = 0.0
                    if entry_px > 0 and close_px > 0:
                        if side_use == 'long':
                            pnl_u = (close_px - entry_px) * prev_size
                            pnl_pct = (close_px - entry_px) / entry_px * 100.0
                        else:
                            pnl_u = (entry_px - close_px) * prev_size
                            pnl_pct = (entry_px - close_px) / entry_px * 100.0

                    try:
                        self.stats.add_trade(symbol, side_use, float(pnl_u))
                    except Exception:
                        pass
                    try:
                        if hasattr(self, 'update_learning_state'):
                            self.update_learning_state(symbol, float(pnl_pct))
                    except Exception:
                        pass

                if cur_size > 0:
                    self.prev_positions[symbol] = {'size': cur_size, 'side': cur_side, 'entry': cur_entry}
                else:
                    if symbol in self.prev_positions:
                        del self.prev_positions[symbol]
        except Exception as e:
            logger.debug(f"🔧 持仓统计监听异常: {e}")

    def manage_ranging_exits(self, symbol: str, pos: dict, market_state: str, levels: dict) -> None:
        """震荡市止盈管理"""
        try:
            if str(market_state) != 'ranging':
                return
            size = float(pos.get('size', 0) or 0.0)
            if size <= 0:
                return
            side = str(pos.get('side', '')).lower()
            entry = float(pos.get('entry_price', 0) or 0.0)
            tick_sz = float(self.get_tick_size(symbol) or 0.0)
            if entry <= 0 or tick_sz <= 0:
                return
            last = 0.0
            try:
                inst_id = self.symbol_to_inst_id(symbol)
                tkr = self.exchange.publicGetMarketTicker({'instId': inst_id})
                d = tkr.get('data') if isinstance(tkr, dict) else tkr
                if isinstance(d, list) and d:
                    last = float(d[0].get('last') or d[0].get('lastPx') or 0.0)
            except Exception:
                pass
            if last <= 0:
                try:
                    df = self.get_klines(symbol, 10)
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        last = float(df['close'].values[-1])
                except Exception:
                    return
            sup = levels.get('support', []) or []
            res = levels.get('resistance', []) or []
            T1 = None
            if side == 'long':
                ups = [x for x in res if x and float(x) > entry]
                T1 = min(ups) if ups else None
            elif side == 'short':
                dns = [x for x in sup if x and float(x) < entry]
                T1 = max(dns) if dns else None
            sl_state = self.sl_tp_state.get(symbol, {})
            sl_init = float(sl_state.get('sl', 0) or 0.0)
            R = abs(entry - sl_init) if sl_init > 0 else max(entry * 0.005, 10.0 * tick_sz)
            st = self.range_pt_state.setdefault(symbol, {'partial_done': False, 'breakeven_active': False, 'trail_anchor': entry})
            reach_T1 = False
            if T1 is not None:
                tol = max(0.002 * entry, 5.0 * tick_sz)
                reach_T1 = (side == 'long' and last >= (float(T1) - tol)) or (side == 'short' and last <= (float(T1) + tol))
            reach_07R = (side == 'long' and (last - entry) >= 0.7 * R) or (side == 'short' and (entry - last) >= 0.7 * R)
            if not st['partial_done'] and (reach_T1 or reach_07R):
                mi = self.markets_info.get(symbol, {}) if hasattr(self, 'markets_info') else {}
                lot_sz = float(mi.get('lot_size') or mi.get('lotSz') or 0.0)
                min_sz = float(mi.get('min_size') or mi.get('minSz') or 0.0)
                raw_part = float(size * 0.6)
                part_sz = float(int(raw_part / lot_sz) * lot_sz) if lot_sz > 0 else round(raw_part, 8)
                part_sz = min(part_sz, size)
                if part_sz <= 0 or (min_sz > 0 and part_sz < min_sz):
                    pass
                else:
                    try:
                        sell_buy = 'sell' if side == 'long' else 'buy'
                        inst_id = self.symbol_to_inst_id(symbol)
                        td_mode = 'isolated' if str(pos.get('margin_mode','cross')).lower() == 'isolated' else 'cross'
                        params_okx = {
                            'instId': inst_id,
                            'tdMode': td_mode,
                            'side': sell_buy,
                            'sz': str(part_sz),
                            'ordType': 'market',
                            'reduceOnly': True,
                        }
                        if self.get_position_mode() == 'hedge':
                            params_okx['posSide'] = ('long' if side == 'short' else 'short') if sell_buy == 'buy' else side
                        self.exchange.privatePostTradeOrder(params_okx)
                        logger.info(f"🎯 震荡分仓止盈60% {symbol}: side={side} size={part_sz}")
                        st['partial_done'] = True
                        st['breakeven_active'] = True
                        st['trail_anchor'] = last
                    except Exception as e:
                        logger.warning(f"⚠️ 分仓止盈失败 {symbol}: {str(e)}")
            if st['partial_done']:
                rem_sz = float(pos.get('size', 0) or 0.0)
                if rem_sz > 0 and st['breakeven_active']:
                    need_exit = (side == 'long' and last <= entry) or (side == 'short' and last >= entry)
                    if need_exit:
                        try:
                            sell_buy = 'sell' if side == 'long' else 'buy'
                            inst_id = self.symbol_to_inst_id(symbol)
                            td_mode = 'isolated' if str(pos.get('margin_mode','cross')).lower() == 'isolated' else 'cross'
                            mi = self.markets_info.get(symbol, {}) if hasattr(self, 'markets_info') else {}
                            lot_sz = float(mi.get('lot_size') or mi.get('lotSz') or 0.0)
                            min_sz = float(mi.get('min_size') or mi.get('minSz') or 0.0)
                            aligned_sz = float(rem_sz)
                            if lot_sz > 0:
                                aligned_sz = float(int(rem_sz / lot_sz) * lot_sz)
                            aligned_sz = min(aligned_sz, rem_sz)
                            if aligned_sz <= 0 or (min_sz > 0 and aligned_sz < min_sz):
                                raise Exception("余仓数量未达最小下单单位,跳过保本退出")
                            params_okx = {
                                'instId': inst_id,
                                'tdMode': td_mode,
                                'side': sell_buy,
                                'sz': str(aligned_sz),
                                'ordType': 'market',
                                'reduceOnly': True,
                            }
                            if self.get_position_mode() == 'hedge':
                                params_okx['posSide'] = ('long' if side == 'short' else 'short') if sell_buy == 'buy' else side
                            self.exchange.privatePostTradeOrder(params_okx)
                            logger.info(f"🛡️ 保本退出余仓 {symbol}: side={side} size={rem_sz}")
                            st['breakeven_active'] = False
                        except Exception as e:
                            logger.warning(f"⚠️ 保本退出失败 {symbol}: {str(e)}")
                anchor = float(st.get('trail_anchor', entry) or entry)
                if side == 'long' and last > anchor:
                    st['trail_anchor'] = last
                elif side == 'short' and last < anchor:
                    st['trail_anchor'] = last
                trail_tol = max(0.003 * entry, 10.0 * tick_sz)
                rem_sz2 = float(pos.get('size', 0) or 0.0)
                if rem_sz2 > 0:
                    recoil = (side == 'long' and (st['trail_anchor'] - last) >= trail_tol) or (side == 'short' and (last - st['trail_anchor']) >= trail_tol)
                    if recoil:
                        try:
                            sell_buy = 'sell' if side == 'long' else 'buy'
                            inst_id = self.symbol_to_inst_id(symbol)
                            td_mode = 'isolated' if str(pos.get('margin_mode','cross')).lower() == 'isolated' else 'cross'
                            mi = self.markets_info.get(symbol, {}) if hasattr(self, 'markets_info') else {}
                            lot_sz = float(mi.get('lot_size') or mi.get('lotSz') or 0.0)
                            min_sz = float(mi.get('min_size') or mi.get('minSz') or 0.0)
                            aligned_sz2 = float(rem_sz2)
                            if lot_sz > 0:
                                aligned_sz2 = float(int(rem_sz2 / lot_sz) * lot_sz)
                            aligned_sz2 = min(aligned_sz2, rem_sz2)
                            if aligned_sz2 <= 0 or (min_sz > 0 and aligned_sz2 < min_sz):
                                raise Exception("余仓数量未达最小下单单位,跳过温和跟踪退出")
                            params_okx = {
                                'instId': inst_id,
                                'tdMode': td_mode,
                                'side': sell_buy,
                                'sz': str(aligned_sz2),
                                'ordType': 'market',
                                'reduceOnly': True,
                            }
                            if self.get_position_mode() == 'hedge':
                                params_okx['posSide'] = ('long' if side == 'short' else 'short') if sell_buy == 'buy' else side
                            self.exchange.privatePostTradeOrder(params_okx)
                            logger.info(f"📉 温和跟踪触发退出 {symbol}: side={side} size={rem_sz2}")
                        except Exception as e:
                            logger.warning(f"⚠️ 跟踪退出失败 {symbol}: {str(e)}")
        except Exception as e:
            logger.warning(f"⚠️ 震荡止盈管理异常 {symbol}: {str(e)}")

    def close_position_market(self, symbol: str) -> bool:
        """🔧 修复1: 新增兜底市价平仓方法"""
        try:
            pos = self.get_position(symbol, force_refresh=True)
            if not pos or float(pos.get('size', 0) or 0) <= 0:
                logger.info(f"ℹ️ 无持仓需要平仓 {symbol}")
                return True
            
            size = float(pos.get('size', 0) or 0)
            side = str(pos.get('side', '')).lower()
            
            # 市价平仓方向
            close_side = 'sell' if side == 'long' else 'buy'
            inst_id = self.symbol_to_inst_id(symbol)
            
            # OKX原生API平仓
            td_mode = 'isolated' if str(pos.get('margin_mode', 'cross')).lower() == 'isolated' else 'cross'
            params_close = {
                'instId': inst_id,
                'tdMode': td_mode,
                'side': close_side,
                'sz': str(size),
                'ordType': 'market',
                'reduceOnly': True
            }
            
            if self.get_position_mode() == 'hedge':
                params_close['posSide'] = side
            
            resp = self.exchange.privatePostTradeOrder(params_close)
            data = resp.get('data') if isinstance(resp, dict) else resp
            
            if data and isinstance(data, list) and data[0]:
                order_id = data[0].get('ordId')
                logger.info(f"✅ 兜底平仓成功 {symbol}: {close_side} {size:.8f} order_id={order_id}")
                return True
            else:
                logger.error(f"❌ 兜底平仓失败 {symbol}: 无有效订单ID")
                return False
                
        except Exception as e:
            logger.error(f"❌ 兜底平仓异常 {symbol}: {e}")
            return False
        
    def execute_strategy(self):
        """执行策略"""
        logger.info("=" * 70)
        logger.info(f"🚀 开始执行MACD+RSI策略 (11个币种,{self.timeframe} 周期)")
        logger.info("=" * 70)
        
        try:
            self.check_sync_needed()
            try:
                self._track_position_stats()
            except Exception as _estat:
                logger.debug(f"🔧 统计监听异常: {_estat}")
            try:
                self.ensure_tpsl_guard()
            except Exception as _e_guard:
                logger.debug(f"🔧 守护执行异常: {_e_guard}")
            try:
                for symbol in self.symbols:
                    pos = self.get_position(symbol, force_refresh=False) or {}
                    cache = self.key_levels_cache.get(symbol, {})
                    sup = [float(x['price']) for x in (cache.get('supports') or [])]
                    res = [float(x['price']) for x in (cache.get('resistances') or [])]
                    df_ex = self.get_klines(symbol, 120)
                    if df_ex is None or df_ex.empty:
                        continue
                    df_ex = self.calculate_indicators(df_ex, symbol)
                    ms_ex = self.assess_market_state(df_ex).get('state', 'unclear')
                    self.manage_ranging_exits(symbol, pos, ms_ex, {'support': sup, 'resistance': res})
            except Exception as _e_range:
                logger.debug(f"🔧 震荡止盈管理异常: {_e_range}")
            
            balance = self.get_account_balance()
            logger.info(f"💰 当前账户余额: {balance:.2f} USDT")
            
            logger.info(self.stats.get_summary())
            
            self.display_current_positions()
            
            self.manage_positions()
            
            logger.info("🔍 分析交易信号...")
            logger.info("-" * 70)
            
            signals = {}
            for symbol in self.symbols:
                signals[symbol] = self.analyze_symbol_adaptive(symbol)
                position = self.get_position(symbol, force_refresh=False)
                open_orders = self.get_open_orders(symbol)
                
                status_line = f"📊 {symbol}: 信号={signals[symbol]['signal']}, 原因={signals[symbol]['reason']}"
                if open_orders:
                    status_line += f", 挂单={len(open_orders)}个"
                
                logger.info(status_line)
                try:
                    time.sleep(self.symbol_loop_delay)
                except Exception:
                    time.sleep(0.2)
            
            logger.info("-" * 70)
            logger.info("⚡ 执行交易操作...")
            logger.info("")
            
            for symbol, signal_info in signals.items():
                signal = signal_info['signal']
                reason = signal_info['reason']
                
                current_position = self.get_position(symbol, force_refresh=True)
                
                # 🔧 修复1: TP/SL触发后的兜底平仓逻辑
                try:
                    kl = self.get_klines(symbol, 50)
                    if not kl.empty:
                        close_price = float(kl.iloc[-1]['close'])
                        ps = self.per_symbol_params.get(symbol, {})
                        atr_p = int(ps.get('atr_period', 14))
                        atr_val = calculate_atr(kl, atr_p)['atr'].iloc[-1]
                        if current_position['size'] > 0 and atr_val > 0:
                            st0 = self.sl_tp_state.get(symbol)
                            if not st0:
                                try:
                                    entry0 = float(current_position.get('entry_price', 0) or 0)
                                    if entry0 > 0:
                                        self._set_initial_sl_tp(symbol, entry0,atr_val, current_position.get('side', 'long'))
                                        okx_ok = self.place_okx_tp_sl(symbol, entry0, current_position.get('side', 'long'), atr_val)
                                        if okx_ok:
                                            logger.info(f"📌 手动/历史持仓兜底:已初始化并挂TP/SL {symbol}")
                                        else:
                                            logger.warning(f"⚠️ 手动/历史持仓兜底挂单失败 {symbol}")
                                except Exception as _e0:
                                    logger.warning(f"⚠️ 兜底初始化SL/TP异常 {symbol}: {_e0}")
                            side_now = current_position.get('side', 'long')
                            self._update_trailing_stop(symbol, close_price, atr_val, side_now)
                            
                            # 🔧 修复1: 检查是否触发SL/TP
                            if self._check_hard_stop(symbol, close_price, side_now):
                                logger.warning(f"🚨 检测到价格触达SL/TP阈值 {symbol},等待交易所OCO执行...")
                                time.sleep(2)  # 等待OCO订单执行
                                
                                # 重新获取持仓
                                current_position = self.get_position(symbol, force_refresh=True)
                                
                                # 如果OCO未执行(持仓仍存在),执行兜底平仓
                                if current_position['size'] > 0:
                                    logger.error(f"🚨 OCO未执行!执行兜底市价平仓 {symbol}")
                                    close_success = self.close_position_market(symbol)
                                    if close_success:
                                        logger.info(f"✅ 兜底平仓成功 {symbol}")
                                    else:
                                        logger.error(f"❌ 兜底平仓失败 {symbol} - 请立即手动检查!")
                                else:
                                    logger.info(f"✅ OCO已执行平仓 {symbol}")
                                
                                # 刷新持仓并跳过本轮
                                current_position = self.get_position(symbol, force_refresh=True)
                                continue
                            
                            self._maybe_partial_take_profit(symbol, close_price, atr_val, side_now)
                            st = self.sl_tp_state.get(symbol)
                            if st:
                                try:
                                    entry_px = float(st.get('entry', 0) or 0)
                                    if entry_px > 0 and atr_val > 0:
                                        profit = (close_price - entry_px) if side_now == 'long' else (entry_px - close_price)
                                        if profit >= 2.5 * atr_val:
                                            st['sl'] = max(st['sl'], close_price - 1.0 * atr_val) if side_now == 'long' else min(st['sl'], close_price + 1.0 * atr_val)
                                        elif profit >= 1.5 * atr_val:
                                            st['sl'] = max(st['sl'], close_price - 1.2 * atr_val) if side_now == 'long' else min(st['sl'], close_price + 1.2 * atr_val)
                                except Exception:
                                    pass
                except Exception as e:
                    logger.debug(f"🔧 持仓管理异常 {symbol}: {e}")
                
                if signal == 'buy':
                    if current_position['size'] > 0 and current_position['side'] == 'long':
                        logger.info(f"ℹ️ {symbol}已有多头持仓,跳过重复开仓")
                        continue
                    
                    amount = self.calculate_order_amount(symbol)
                    if amount > 0:
                        if self.create_order(symbol, 'buy', amount):
                            logger.info(f"🚀 开多{symbol}成功 - {reason}")
                            self.last_position_state[symbol] = 'long'
                
                elif signal == 'sell':
                    if current_position['size'] > 0 and current_position['side'] == 'short':
                        logger.info(f"ℹ️ {symbol}已有空头持仓,跳过重复开仓")
                        continue
                    
                    amount = self.calculate_order_amount(symbol)
                    if amount > 0:
                        if self.create_order(symbol, 'sell', amount):
                            logger.info(f"📉 开空{symbol}成功 - {reason}")
                            self.last_position_state[symbol] = 'short'
                
                elif signal == 'close':
                    _pp = self.per_symbol_params.get(symbol, {})
                    allow_reverse = bool(_pp.get('allow_reverse', True)) if isinstance(_pp, dict) else True
                    logger.info(f"⛔ 触发平仓检查: {symbol} allow_reverse={allow_reverse} - {reason}")
            
            logger.info("=" * 70)
                        
        except Exception as e:
            logger.error(f"❌ 执行策略失败: {e}")
            logger.error(traceback.format_exc())
    
    def run_continuous(self, interval: int = 60):
        """连续运行策略"""
        logger.info("=" * 70)
        logger.info("🚀 MACD+RSI策略启动 - RAILWAY平台版 (11个币种)")
        logger.info("=" * 70)
        logger.info(f"📈 MACD参数: 快线={self.fast_period}, 慢线={self.slow_period}, 信号线={self.signal_period}")
        logger.info(f"📊 全局默认周期: {self.timeframe}")
        tf_desc = ', '.join([f"{s.split('/')[0]}={self.timeframe_map.get(s, self.timeframe)}" for s in self.symbols])
        logger.info(f"🗺️ 分币种周期: {tf_desc}")
        lev_desc = ', '.join([f"{s.split('/')[0]}={self.symbol_leverage.get(s, 20)}x" for s in self.symbols])
        logger.info(f"💪 杠杆倍数: {lev_desc}")
        logger.info("⏰ 刷新方式: 实时巡检(每interval秒执行一次,可用环境变量 SCAN_INTERVAL 调整,默认1秒)")
        logger.info(f"📄 状态同步: 每{self.sync_interval}秒")
        logger.info(f"📊 监控币种: {', '.join(self.symbols)}")
        logger.info(f"💡 11个币种特性: 支持0.1U起的小额交易,平均分配资金")
        logger.info(self.stats.get_summary())
        logger.info("=" * 70)
        logger.info("")
        logger.info("🔧 已修复的关键问题:")
        logger.info("  1. ✅ TP/SL触发后OCO未执行时的兜底平仓机制")
        logger.info("  2. ✅ 追踪止损可条件性更新到交易所(优化SL时撤旧挂新)")
        logger.info("  3. ✅ 开仓后立即设置TP/SL保护(无保护状态持续<2秒)")
        logger.info("=" * 70)

        china_tz = pytz.timezone('Asia/Shanghai')

        while True:
            try:
                start_ts = time.time()

                self.check_sync_needed()

                self.execute_strategy()

                elapsed = time.time() - start_ts
                sleep_sec = max(1, int(interval - elapsed)) if interval > 0 else 1
                logger.info(f"⏳ 休眠 {sleep_sec} 秒后继续实时巡检...")
                time.sleep(sleep_sec)

            except KeyboardInterrupt:
                logger.info("⛔ 用户中断,策略停止")
                break
            except Exception as e:
                logger.error(f"❌ 策略运行异常: {e}")
                logger.error(traceback.format_exc())
                logger.info("🔄 60秒后重试...")
                time.sleep(60)

def main():
    """主函数"""
    logger.info("=" * 70)
    logger.info("🎯 MACD+RSI策略程序启动中... (11个币种版本)")
    logger.info("=" * 70)
    
    okx_api_key = os.environ.get('OKX_API_KEY', '')
    okx_secret_key = os.environ.get('OKX_SECRET_KEY', '')
    okx_passphrase = os.environ.get('OKX_PASSPHRASE', '')
    
    missing_vars = []
    if not okx_api_key:
        missing_vars.append('OKX_API_KEY')
    if not okx_secret_key:
        missing_vars.append('OKX_SECRET_KEY')
    if not okx_passphrase:
        missing_vars.append('OKX_PASSPHRASE')
    
    if missing_vars:
        logger.error(f"❌ 缺少环境变量: {', '.join(missing_vars)}")
        logger.error("💡 请在RAILWAY平台上设置这些环境变量")
        return
    
    logger.info("✅ 环境变量检查通过")
    
    try:
        strategy = MACDStrategy(
            api_key=okx_api_key,
            secret_key=okx_secret_key,
            passphrase=okx_passphrase
        )
        
        logger.info("✅ 策略初始化成功")

        def _get(k, default=''):
            v = os.environ.get(k, '')
            return v if (v is not None and str(v).strip() != '') else default
        logger.info(f"🔧 变量: SCAN_INTERVAL={_get('SCAN_INTERVAL','2')} OKX_API_MIN_INTERVAL={_get('OKX_API_MIN_INTERVAL','0.2')} SYMBOL_LOOP_DELAY={_get('SYMBOL_LOOP_DELAY','0.3')} SET_LEVERAGE_ON_START={_get('SET_LEVERAGE_ON_START','false')}")
        logger.info(f"🔧 变量: MAX_RETRIES={_get('MAX_RETRIES','3')} BACKOFF_BASE={_get('BACKOFF_BASE','0.8')} BACKOFF_MAX={_get('BACKOFF_MAX','3.0')} TP_SL_REFRESH_INTERVAL={_get('TP_SL_REFRESH_INTERVAL','300')}")

        try:
            scan_interval_env = os.environ.get('SCAN_INTERVAL', '').strip()
            scan_interval = int(scan_interval_env) if scan_interval_env else 2
            if scan_interval <= 0:
                scan_interval = 1
        except Exception:
            scan_interval = 1
        logger.info(f"🛠️ 扫描间隔设置: {scan_interval} 秒(可用环境变量 SCAN_INTERVAL 覆盖)")
        strategy.run_continuous(interval=scan_interval)
        
    except Exception as e:
        logger.error(f"❌ 策略初始化或运行失败: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()