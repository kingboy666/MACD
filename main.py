#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MACD+RSI策略实现 - RAILWAY平台版本
扩展到11个币种，包含BTC/ETH/SOL/DOGE/XRP/PEPE/ARB
25倍杠杆，无限制交易，带挂单识别和状态同步
增加胜率统计和盈亏显示
进一步优化版：增强模块化(1)、性能(2)、错误处理(3)、日志(5)、其他(9)；TP/SL&BB验证无问题，但添加更多日志和dry-run模拟
新增：布林带开口过滤（>0.8*mean保留信号） 与 动态止盈调节（趋势强时放宽TP距离）
修复：检测posMode并调整posSide参数，避免one-way模式错误
集成RSI.txt中的优化MACD+RSI策略和参数
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
logger.propagate = False  # 防止重复日志

# 手动RSI计算函数（因为ta库不可用）
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
                logger.info(f"✅ 加载历史统计数据：总交易{self.stats['total_trades']}笔")
        except Exception as e:
            logger.warning(f"⚠️ 加载统计数据失败: {str(e)} - {traceback.format_exc()}，使用新数据")
    
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
        
        # 添加交易历史 - 使用北京时间
        china_tz = pytz.timezone('Asia/Shanghai')
        trade_record = {
            'timestamp': datetime.datetime.now(china_tz).strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': symbol,
            'side': side,
            'pnl': round(pnl, 4)
        }
        self.stats['trades_history'].append(trade_record)
        
        # 只保留最近100条记录
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
    # 仅对特定交易对的出场行为做覆盖（不依赖环境变量）
    # 键名说明：
    # - TRAIL_ACTIVATE_PCT：追踪止损的百分比激活阈值（替代全局 self.trail_activate_pct）
    # - trail_pct：追踪止损的跟随步长（等效于 cfg['trail_pct']）
    # - INITIAL_SL_FLOOR_PCT：初始SL的最小亏损比例地板（长: entry*(1-地板)；短: entry*(1+地板)）
    # - INITIAL_TP_TARGET_PCT：初始TP的最小盈利比例目标（长: >= entry*(1+目标)；短: <= entry*(1-目标)）
    # - PARTIAL_TP_TIERS：分批止盈阶梯字符串；空字符串表示关闭分批
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
        # SAR 结果缓存：key=(tag, len, last_ts, af_start, af_max) -> last_sar
        self._sar_cache: Dict[tuple, float] = {}
        # K线缓存：per symbol, {timestamp: klines}
        self._klines_cache: Dict[str, Dict[float, List[Dict]]] = {}
        self._klines_ttl = 60  # 秒

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
        
        # OKX统一参数
        self.okx_params = {'instType': 'SWAP'}

        # 将统一交易对转为OKX instId
        def _symbol_to_inst_id(sym: str) -> str:
            try:
                base = sym.split('/')[0]
                return f"{base}-USDT-SWAP"
            except Exception:
                return ''
        self.symbol_to_inst_id = _symbol_to_inst_id
        
        # ===== 交易对配置 - 扩展到11个币种 =====
        self.symbols = [
            # 原有4个小币种
            'FIL/USDT:USDT',
            'ZRO/USDT:USDT',
            'WIF/USDT:USDT',
            'WLD/USDT:USDT',
            # 新增7个币种
            'BTC/USDT:USDT',    # 比特币
            'ETH/USDT:USDT',    # 以太坊
            'SOL/USDT:USDT',    # Solana
            'DOGE/USDT:USDT',   # 狗狗币
            'XRP/USDT:USDT',    # 瑞波币
            'PEPE/USDT:USDT',   # 佩佩蛙
            'ARB/USDT:USDT'     # Arbitrum
        ]
        
        # 时间周期 - 15分钟
        self.timeframe = '15m'
        # 按币种指定周期：BTC/ETH/FIL/WLD 用 15m，其余使用全局 timeframe（可扩展 DOGE/XRP 为 10m）
        self.timeframe_map = {
            # 15m：波动惯性强的主流币
            'BTC/USDT:USDT': '15m',
            'ETH/USDT:USDT': '15m',
            'FIL/USDT:USDT': '15m',
            'WLD/USDT:USDT': '15m',
            # 5m：高频波动，短周期更有效
            'SOL/USDT:USDT': '15m',
            'WIF/USDT:USDT': '15m',
            'ZRO/USDT:USDT': '15m',
            'ARB/USDT:USDT': '15m',
            'PEPE/USDT:USDT': '15m',
            # 10m：中等波动
            'DOGE/USDT:USDT': '15m',
            'XRP/USDT:USDT': '15m',
        }
        
        # === 币种分类 ===
        self.coin_categories = {
            'blue_chip': ['BTC/USDT:USDT', 'ETH/USDT:USDT'],
            'mainnet': ['SOL/USDT:USDT', 'XRP/USDT:USDT', 'ARB/USDT:USDT'],
            'infrastructure': ['FIL/USDT:USDT'],
            'emerging': ['ZRO/USDT:USDT', 'WLD/USDT:USDT'],
            'meme': ['DOGE/USDT:USDT', 'WIF/USDT:USDT', 'PEPE/USDT:USDT']
        }
        
        # === 优化后的MACD参数 ===
        self.macd_params = {
            'BTC/USDT:USDT': {'fast': 8, 'slow': 17, 'signal': 9},
            'ETH/USDT:USDT': {'fast': 8, 'slow': 17, 'signal': 9},
            'SOL/USDT:USDT': {'fast': 6, 'slow': 16, 'signal': 9},
            'XRP/USDT:USDT': {'fast': 7, 'slow': 17, 'signal': 9},
            'ARB/USDT:USDT': {'fast': 10, 'slow': 26, 'signal': 9},
            'FIL/USDT:USDT': {'fast': 6, 'slow': 16, 'signal': 9},
            'ZRO/USDT:USDT': {'fast': 6, 'slow': 16, 'signal': 9},
            'WLD/USDT:USDT': {'fast': 5, 'slow': 13, 'signal': 9},
            'DOGE/USDT:USDT': {'fast': 9, 'slow': 25, 'signal': 9},
            'WIF/USDT:USDT': {'fast': 8, 'slow': 21, 'signal': 9},
            'PEPE/USDT:USDT': {'fast': 9, 'slow': 23, 'signal': 9}
        }
        
        # === 优化后的RSI参数 ===
        self.rsi_params = {
            'BTC/USDT:USDT': 14,
            'ETH/USDT:USDT': 14,
            'SOL/USDT:USDT': 11,
            'XRP/USDT:USDT': 12,
            'ARB/USDT:USDT': 10,
            'FIL/USDT:USDT': 9,
            'ZRO/USDT:USDT': 14,
            'WLD/USDT:USDT': 9,
            'DOGE/USDT:USDT': 10,
            'WIF/USDT:USDT': 9,
            'PEPE/USDT:USDT': 9
        }
        
        # === 动态超买超卖阈值 ===
        self.rsi_thresholds = {
            'BTC/USDT:USDT': {'overbought': 70, 'oversold': 30},
            'ETH/USDT:USDT': {'overbought': 70, 'oversold': 30},
            'SOL/USDT:USDT': {'overbought': 72, 'oversold': 28},
            'XRP/USDT:USDT': {'overbought': 70, 'oversold': 30},
            'ARB/USDT:USDT': {'overbought': 75, 'oversold': 25},
            'FIL/USDT:USDT': {'overbought': 73, 'oversold': 27},
            'ZRO/USDT:USDT': {'overbought': 75, 'oversold': 25},
            'WLD/USDT:USDT': {'overbought': 75, 'oversold': 25},
            'DOGE/USDT:USDT': {'overbought': 75, 'oversold': 25},
            'WIF/USDT:USDT': {'overbought': 80, 'oversold': 20},
            'PEPE/USDT:USDT': {'overbought': 80, 'oversold': 20}
        }
        
        # === 每币种严格策略模式（来自 rsi.txt） ===
        self.strategy_mode_map = {
            # 你指定的四个币种的专属模式
            'FIL/USDT:USDT': 'combo',         # 金叉死叉 + 背离
            'ZRO/USDT:USDT': 'zero_cross',    # 零轴突破
            'WLD/USDT:USDT': 'divergence',    # 背离为主（波动大）
            'WIF/USDT:USDT': 'golden_cross',  # 金叉死叉（快进快出）
            # 其余币种统一使用综合模式 combo
            'BTC/USDT:USDT': 'combo',
            'ETH/USDT:USDT': 'combo',
            'SOL/USDT:USDT': 'combo',
            'DOGE/USDT:USDT': 'combo',
            'XRP/USDT:USDT': 'combo',
            'PEPE/USDT:USDT': 'combo',
            'ARB/USDT:USDT': 'combo',
        }
        
        # === 优化后的止损止盈 ===
        self.stop_loss = {
            'BTC/USDT:USDT': 2.0,
            'ETH/USDT:USDT': 2.0,
            'SOL/USDT:USDT': 2.5,
            'XRP/USDT:USDT': 2.3,
            'ARB/USDT:USDT': 0.8,
            'FIL/USDT:USDT': 2.8,
            'ZRO/USDT:USDT': 3.0,
            'WLD/USDT:USDT': 3.5,
            'DOGE/USDT:USDT': 0.7,
            'WIF/USDT:USDT': 0.6,
            'PEPE/USDT:USDT': 0.6
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
        
        # === 仓位权重（根据币种稳定性）===
        self.position_weights = {
            'BTC/USDT:USDT': 1.2,  # 蓝筹可以稍微加仓
            'ETH/USDT:USDT': 1.2,
            'SOL/USDT:USDT': 1.0,
            'XRP/USDT:USDT': 1.0,
            'ARB/USDT:USDT': 0.9,
            'FIL/USDT:USDT': 0.9,
            'ZRO/USDT:USDT': 0.8,
            'WLD/USDT:USDT': 0.7,
            'DOGE/USDT:USDT': 0.6,  # MEME币减仓
            'WIF/USDT:USDT': 0.5,
            'PEPE/USDT:USDT': 0.4
        }
        
        self.positions = {}
        self.strategy_mode = 'combo'
        
        # === 统计数据 ===
        self.trade_stats = {symbol: {'wins': 0, 'losses': 0, 'total_pnl': 0} for symbol in self.symbols}
        
        self._sar_cache: Dict[tuple, float] = {}
        self._klines_cache: Dict[str, Dict[float, List[Dict]]] = {}
        self._klines_ttl = 60  # 秒
        
        # OKX统一参数
        self.okx_params = {'instType': 'SWAP'}

        # 将统一交易对转为OKX instId
        def _symbol_to_inst_id(sym: str) -> str:
            try:
                base = sym.split('/')[0]
                return f"{base}-USDT-SWAP"
            except Exception:
                return ''
        self.symbol_to_inst_id = _symbol_to_inst_id
        
        # 时间周期 - 15分钟
        self.timeframe = '15m'
        # 按币种指定周期：BTC/ETH/FIL/WLD 用 15m，其余使用全局 timeframe（可扩展 DOGE/XRP 为 10m）
        self.timeframe_map = {
            # 15m：波动惯性强的主流币
            'BTC/USDT:USDT': '15m',
            'ETH/USDT:USDT': '15m',
            'FIL/USDT:USDT': '15m',
            'WLD/USDT:USDT': '15m',
            # 5m：高频波动，短周期更有效
            'SOL/USDT:USDT': '15m',
            'WIF/USDT:USDT': '5m',
            'ZRO/USDT:USDT': '15m',
            'ARB/USDT:USDT': '5m',
            'PEPE/USDT:USDT': '5m',
            # 10m：中等波动
            'DOGE/USDT:USDT': '5m',
            'XRP/USDT:USDT': '15m',
        }
        
        # MACD参数
        self.fast_period = 10
        self.slow_period = 40
        self.signal_period = 15
        
        # ===== 杠杆配置 - 根据币种风险分级 =====
        self.symbol_leverage: Dict[str, int] = {
            # 原有小币种
            'FIL/USDT:USDT': 25,   # 降低(原30)
            'WIF/USDT:USDT': 20,   # 降低(原25)
            'WLD/USDT:USDT': 25,   # 降低(原30)
            'ZRO/USDT:USDT': 20,
            # 主流币 - 较高杠杆
            'BTC/USDT:USDT': 30,
            'ETH/USDT:USDT': 30,
            'SOL/USDT:USDT': 25,
            'XRP/USDT:USDT': 25,
            # Meme币 - 低杠杆
            'DOGE/USDT:USDT': 20,
            'PEPE/USDT:USDT': 15,
            # L2币
            'ARB/USDT:USDT': 25,
        }
        
        # ===== 分币种参数 - 精细调优 =====
        # 分组精细化策略参数（运行时与 per_symbol_params 合并覆盖）
        self.strategy_params: Dict[str, Dict[str, Any]] = {
            'BTC/USDT:USDT': {'strategy': 'macd_sar', 'bb_period': 20, 'bb_k': 2.0, 'sar_af_start': 0.02, 'sar_af_max': 0.20},
            'ETH/USDT:USDT': {'strategy': 'macd_sar', 'bb_period': 20, 'bb_k': 2.0, 'sar_af_start': 0.02, 'sar_af_max': 0.20},
            'SOL/USDT:USDT': {'strategy': 'macd_sar', 'bb_period': 20, 'bb_k': 2.0, 'sar_af_start': 0.02, 'sar_af_max': 0.20},
            'WIF/USDT:USDT': {'strategy': 'bb_sar',  'bb_period': 20, 'bb_k': 2.5, 'sar_af_start': 0.01, 'sar_af_max': 0.10},
            'PEPE/USDT:USDT':{'strategy': 'bb_sar',  'bb_period': 20, 'bb_k': 2.5, 'sar_af_start': 0.01, 'sar_af_max': 0.10},
            'DOGE/USDT:USDT':{'strategy': 'bb_sar',  'bb_period': 20, 'bb_k': 2.5, 'sar_af_start': 0.01, 'sar_af_max': 0.10},
            'ZRO/USDT:USDT': {'strategy': 'hybrid',  'bb_period': 20, 'bb_k': 2.2, 'sar_af_start': 0.03, 'sar_af_max': 0.25},
            'WLD/USDT:USDT': {'strategy': 'hybrid',  'bb_period': 20, 'bb_k': 2.2, 'sar_af_start': 0.03, 'sar_af_max': 0.25},
            'FIL/USDT:USDT': {'strategy': 'hybrid',  'bb_period': 20, 'bb_k': 2.2, 'sar_af_start': 0.03, 'sar_af_max': 0.25},
            'XRP/USDT:USDT': {'strategy': 'bb_sar',  'bb_period': 20, 'bb_k': 1.8, 'sar_af_start': 0.02, 'sar_af_max': 0.15},
            'ARB/USDT:USDT': {'strategy': 'bb_sar',  'bb_period': 20, 'bb_k': 1.8, 'sar_af_start': 0.02, 'sar_af_max': 0.15},
        }
        self.per_symbol_params: Dict[str, Dict[str, Any]] = {
            # 原有小币种
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
            
            # 新增主流币
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
            
            # 新增Meme币
            'DOGE/USDT:USDT': {
                'macd': (5, 13, 9), 'atr_period': 18, 'adx_period': 12,
                'adx_min_trend': 24, 'sl_n': 2.7, 'tp_m': 5.5, 'allow_reverse': True
            },
            'PEPE/USDT:USDT': {
                'macd': (5, 13, 9), 'atr_period': 18, 'adx_period': 10,
                'adx_min_trend': 24, 'sl_n': 3.2, 'tp_m': 6.5, 'allow_reverse': True
            },
            
            # 新增L2币
            'ARB/USDT:USDT': {
                'macd': (6, 18, 9), 'atr_period': 18, 'adx_period': 12,
                'adx_min_trend': 24, 'sl_n': 2.4, 'tp_m': 4.3, 'allow_reverse': True
            }
        }
        
        # 仓位配置 - 使用100%资金
        self.position_percentage = 1.0
        
        # 持仓和挂单缓存
        self.positions_cache: Dict[str, Dict[str, Any]] = {}
        self.open_orders_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.last_sync_time: float = 0
        self.sync_interval: int = 60
        
        # 市场信息缓存
        self.markets_info: Dict[str, Dict[str, Any]] = {}
        
        # API 速率限制
        self._last_api_ts: float = 0.0
        self._min_api_interval: float = 0.2

        # 每币种微延时，降低瞬时调用密度
        self.symbol_loop_delay = 0.3
        # 风险百分比（每笔占用余额百分比），默认0.5%，可用环境变量 RISK_PERCENT 覆盖
        try:
            rp_str = (os.environ.get('RISK_PERCENT') or '0.5').strip()
            self.risk_percent = max(0.0, float(rp_str))
        except Exception:
            self.risk_percent = 0.5
        # 启动时是否逐币设置杠杆（可设为 false 减少启动阶段私有接口调用）
        self.set_leverage_on_start = False
        
        # 交易统计
        self.stats = TradingStats()

        # ===== 策略分组与BB/SAR参数（第一阶段以轻量映射接入）=====
        self.strategy_by_symbol: Dict[str, str] = {
            # 主流：macd_sar
            'BTC/USDT:USDT': 'macd_sar',
            'ETH/USDT:USDT': 'macd_sar',
            'SOL/USDT:USDT': 'macd_sar',
            # 高波动：bb_sar
            'WIF/USDT:USDT': 'bb_sar',
            'PEPE/USDT:USDT': 'bb_sar',
            'DOGE/USDT:USDT': 'bb_sar',
            # 中波动：hybrid
            'ZRO/USDT:USDT': 'hybrid',
            'WLD/USDT:USDT': 'hybrid',
            'FIL/USDT:USDT': 'hybrid',
            # 震荡：bb_sar
            'XRP/USDT:USDT': 'bb_sar',
            'ARB/USDT:USDT': 'bb_sar',
        }
        self.bb_tp_offset = 0.003
        self.bb_sl_offset = 0.002
        
        # 启动基线余额与风控参数
        try:
            self.starting_balance = float(self.get_account_balance() or 0.0)
        except Exception:
            self.starting_balance = 0.0
        self.hard_sl_max_loss_pct = 0.03
        self.account_dd_limit_pct = 0.20
        self.cb_close_all = True
        # 强制彻底关闭账户熔断
        self.cb_enabled = False
        self.circuit_breaker_triggered = False
        self.partial_tp_done: Dict[str, set] = {}
        # 撤单/标记 安全控制
        self.allow_cancel_pending = True
        self.safe_cancel_only_our_tpsl = True
        self.tpsl_cl_prefix = 'MACD_TPSL_'
        
        # ATR 止盈止损参数
        self.atr_sl_n = 1.8
        self.atr_tp_m = 2.2
        
        # SL/TP 状态缓存
        self.sl_tp_state: Dict[str, Dict[str, float]] = {}
        self.okx_tp_sl_placed: Dict[str, bool] = {}
        # 1H多头时TP放大倍数(默认1.0)
        self.tp_boost_map: Dict[str, float] = {s: 1.0 for s in self.symbols}
        # TP/SL重挂冷却与阈值
        self.tp_sl_last_placed: Dict[str, float] = {}
        self.tp_sl_refresh_interval = 300
        self.tp_sl_min_delta_ticks = 2
        
        # ===== 每币种配置(用于追踪止损) =====
        self.symbol_cfg: Dict[str, Dict[str, float | str]] = {
            # 原有币种
            "ZRO/USDT:USDT": {"period": 14, "n": 2.2, "m": 3.0, "trigger_pct": 0.010, "trail_pct": 0.006, "update_basis": "high"},
            "WIF/USDT:USDT": {"period": 14, "n": 2.5, "m": 4.0, "trigger_pct": 0.017, "trail_pct": 0.008, "update_basis": "high"},
            "WLD/USDT:USDT": {"period": 14, "n": 2.0, "m": 3.5, "trigger_pct": 0.010, "trail_pct": 0.006, "update_basis": "close"},
            "FIL/USDT:USDT": {"period": 14, "n": 2.0, "m": 3.5, "trigger_pct": 0.010, "trail_pct": 0.006, "update_basis": "high"},
            
            # 新增主流币
            "BTC/USDT:USDT": {"period": 20, "n": 1.5, "m": 3.0, "trigger_pct": 0.008, "trail_pct": 0.004, "update_basis": "high"},
            "ETH/USDT:USDT": {"period": 18, "n": 1.8, "m": 3.5, "trigger_pct": 0.008, "trail_pct": 0.005, "update_basis": "high"},
            "SOL/USDT:USDT": {"period": 16, "n": 2.0, "m": 4.0, "trigger_pct": 0.012, "trail_pct": 0.007, "update_basis": "high"},
            "XRP/USDT:USDT": {"period": 16, "n": 1.8, "m": 3.5, "trigger_pct": 0.010, "trail_pct": 0.006, "update_basis": "close"},
            
            # 新增Meme币
            "DOGE/USDT:USDT": {"period": 16, "n": 2.5, "m": 5.0, "trigger_pct": 0.017, "trail_pct": 0.008, "update_basis": "high"},
            "PEPE/USDT:USDT": {"period": 14, "n": 3.0, "m": 6.0, "trigger_pct": 0.022, "trail_pct": 0.010, "update_basis": "high"},
            
            # 新增L2币
            "ARB/USDT:USDT": {"period": 15, "n": 2.2, "m": 3.8, "trigger_pct": 0.014, "trail_pct": 0.006, "update_basis": "high"}
        }
        
        # 跟踪峰值/谷值
        self.trailing_peak: Dict[str, float] = {}
        self.trailing_trough: Dict[str, float] = {}
        # 交易执行冷却与阶段追踪状态
        self.last_trade_candle_index: Dict[str, int] = {}
        self.stage3_done: Dict[str, bool] = {}
        # 信号增强配置（可用环境变量覆盖）
        self.ma_type = os.environ.get('MA_TYPE', 'sma').strip().lower() or 'sma'  # sma|ema
        self.ma_fast = int(os.environ.get('MA_FAST', '5'))
        self.ma_slow = int(os.environ.get('MA_SLOW', '20'))
        self.vol_ma_period = int(os.environ.get('VOL_MA_PERIOD', '20'))
        self.vol_boost = float(os.environ.get('VOL_BOOST', '1.2'))
        self.long_body_pct = float(os.environ.get('LONG_BODY_PCT', '0.6'))
        self.cooldown_candles = int(os.environ.get('COOLDOWN_CANDLES', '3'))
        # 三阶段追踪与最小阈值
        self.trail_stage_1 = float(os.environ.get('TRAIL_STAGE_1', '1.0'))
        self.trail_stage_2 = float(os.environ.get('TRAIL_STAGE_2', '1.75'))
        self.trail_stage_3 = float(os.environ.get('TRAIL_STAGE_3', '2.5'))
        self.trail_stage2_offset = float(os.environ.get('TRAIL_STAGE2_OFFSET', '0.8'))
        self.trail_sl_min_delta_atr = float(os.environ.get('TRAIL_SL_MIN_DELTA_ATR', '0.2'))
        self.partial_tp_ratio_stage3 = float(os.environ.get('PARTIAL_TP_RATIO_STAGE3', '0.3'))
        self.allow_strong_pa_override = (os.environ.get('ALLOW_STRONG_PA_OVERRIDE', 'true').lower() in ('1','true','yes'))
        
        # 记录上次持仓状态
        self.last_position_state: Dict[str, str] = {}
        
        # 初始化交易所
        self._setup_exchange()
        
        # 加载市场信息
        self._load_markets()
        
        # 首次同步状态
        self.sync_all_status()
        
        # 处理启动前已有的持仓和挂单
        self.handle_existing_positions_and_orders()
    
    # ===== 限频节流与退避封装 =====
    def _sleep_with_throttle(self):
        """满足最小调用间隔，加入轻微抖动"""
        try:
            now = time.time()
            delta = now - float(self._last_api_ts or 0.0)
            min_int = float(self._min_api_interval or 0.2)
            if delta < min_int:
                jitter = float(np.random.uniform(0, min_int * 0.1))
                time.sleep(min_int - delta + jitter)
            self._last_api_ts = time.time()
        except Exception:
            # 回退：固定最小sleep
            time.sleep(float(self._min_api_interval or 0.2))

    def _safe_call(self, func, *args, **kwargs):
        """
        包装API调用：先节流；遇到50011(Too Many Requests)执行指数退避重试。
        可通过环境变量调整：MAX_RETRIES, BACKOFF_BASE, BACKOFF_MAX
        """
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
                # 扩展瞬时/限频错误的重试判断范围
                is_rate = any(s in msg for s in (
                    '50011',             # Too Many Requests
                    'Too Many Requests',
                    'rate limit',
                    'ETIMEDOUT',
                    'timeout',
                    'NetworkError',
                    'ConnectionReset',
                    'ECONNRESET'
                ))
                if not is_rate or i >= retries:
                    raise
                wait = min(max_wait, base * (2 ** i)) + float(np.random.uniform(0, 0.2))
                logger.warning(f"⏳ 限频(50011) 第{i+1}次重试，等待 {wait:.2f}s")
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
                logger.warning(f"⚠️ 预加载市场数据失败，将使用安全回退: {e}")
            
            # 按交易对设置杠杆（可选）
            if self.set_leverage_on_start:
                for symbol in self.symbols:
                    try:
                        # Cancel existing TP/SL algo orders first
                        self.cancel_symbol_tp_sl(symbol)
                        time.sleep(0.5)  # Short delay to avoid rate limits
                        
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
                        logger.warning(f"⚠️ 设置{symbol}杠杆失败（可能已设置）: {e}")
            
            try:
                self.exchange.set_position_mode(True)
                logger.info("✅ 设置为双向持仓模式（多空分开）")
            except Exception as e:
                logger.warning(f"⚠️ 设置持仓模式失败（当前可能有持仓，跳过设置）")
                logger.info("ℹ️ 程序将继续运行，使用当前持仓模式")
            
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
                logger.warning(f"⚠️ 时间差较大: {time_diff}ms，可能影响交易")
            
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
        """撤销该交易对在OKX侧已挂的TP/SL（算法单）。仅撤本程序挂的单（clOrdId前缀），携带 instId，按 ordType 分组撤销。"""
        try:
            inst_id = self.symbol_to_inst_id(symbol)
            if not inst_id:
                return True
            resp = self.exchange.privateGetTradeOrdersAlgoPending({'instType': 'SWAP', 'instId': inst_id})
            data = resp.get('data') if isinstance(resp, dict) else resp
            groups: Dict[str, List[Dict[str, str]]] = {}
            for it in (data or []):
                try:
                    ord_type = str(it.get('ordType') or '').lower()
                    if not ord_type:
                        continue
                    clid = str(it.get('clOrdId') or '')
                    if self.safe_cancel_only_our_tpsl and self.tpsl_cl_prefix and (not clid.startswith(self.tpsl_cl_prefix)):
                        continue
                    aid = it.get('algoId') or it.get('algoID') or it.get('id')
                    if aid:
                        groups.setdefault(ord_type, []).append({'algoId': str(aid), 'clOrdId': clid})
                except Exception:
                    continue
            if not groups:
                return True
            total = 0
            for ord_type, items in groups.items():
                ids = [x['algoId'] for x in items]
                payload_obj = {'algoIds': [{'algoId': x} for x in ids], 'instId': inst_id}
                payload_arr = {'algoIds': ids, 'instId': inst_id}
                ok_this = False
                try:
                    self.exchange.privatePostTradeCancelAlgos(payload_obj)
                    ok_this = True
                except Exception:
                    try:
                        self.exchange.privatePostTradeCancelAlgos(payload_arr)
                        ok_this = True
                    except Exception:
                        for aid in ids:
                            try:
                                self.exchange.privatePostTradeCancelAlgos({'algoId': aid, 'instId': inst_id})
                                ok_this = True
                            except Exception:
                                continue
                if ok_this:
                    total += len(ids)
                else:
                    logger.warning(f"⚠️ 撤销 {symbol} 条件单失败：ordType={ord_type}")
            if total > 0:
                logger.info(f"✅ 撤销 {symbol} 条件单数量: {total}")
                return True
            logger.warning(f"⚠️ 撤销 {symbol} 条件单失败：未知原因")
            return False
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
        logger.info(f"💡 11个币种交易：支持0.1U起的小额交易")
        
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
            logger.info("   1. 已有持仓: 程序会根据MACD信号管理，出现反向信号时平仓")
            logger.info("   2. 已有挂单: 程序会在下次交易前自动取消")
            logger.info("   3. 程序会继续运行并根据信号执行交易")
            logger.info("=" * 70)
            logger.info("⚠️ 如果需要立即平仓所有持仓，请手动操作或重启程序前先手动平仓")
            logger.info("=" * 70)
        else:
            logger.info("✅ 启动前无持仓和挂单，可以正常运行")
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
                    side = 'long' if p.get('posSide') == 'long' else 'short'
                    entry_price = float(p.get('avgPx', 0) or 0)
                    leverage = float(p.get('lever', 0) or 0)
                    unreal = float(p.get('upl', 0) or 0)
                    pos_data = {
                        'size': size,
                        'side': side,
                        'entry_price': entry_price,
                        'unrealized_pnl': unreal,
                        'leverage': leverage,
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
        """计算下单金额 - 按信号逐币分配，不做全体平均；余额/保证金不足则跳过"""
        try:
            balance = self.get_account_balance()
            if balance <= 0:
                logger.warning(f"⚠️ 余额不足，无法为 {symbol} 分配资金 (余额:{balance:.4f}U)")
                return 0.0

            # 1) 固定目标名义金额（最高优先）
            target_str = os.environ.get('TARGET_NOTIONAL_USDT', '').strip()
            if target_str:
                try:
                    target = max(0.0, float(target_str))
                    logger.info(f"💵 使用固定目标名义金额: {target:.4f}U")
                except Exception:
                    logger.warning(f"⚠️ TARGET_NOTIONAL_USDT 无效: {target_str}")
                    target = 0.0
            else:
                # 2) 默认每笔订单名义金额（不平均，全额用于当前有信号的币）
                try:
                    target = max(0.0, float((os.environ.get('DEFAULT_ORDER_USDT') or '1.0').strip()))
                except Exception:
                    target = 1.0

            # 3) 放大因子
            try:
                factor = max(1.0, float((os.environ.get('ORDER_NOTIONAL_FACTOR') or '1').strip()))
            except Exception:
                factor = 1.0
            target *= factor

            # 4) 下限/上限
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
                logger.warning(f"⚠️ {symbol} 目标金额为0，跳过")
                return 0.0

            # 5) 保证金充足性检查（不足则跳过，避免 51008/下单失败）
            try:
                lev = float(self.symbol_leverage.get(symbol, 20) or 20)
                required_margin = target / max(1.0, lev)
                # 预留 2% 安全系数
                if balance < required_margin * 1.02:
                    logger.warning(f"⚠️ 保证金不足，跳过 {symbol}: 余额={balance:.4f}U 需保证金≈{required_margin:.4f}U (lev={lev:.1f}x, 目标={target:.4f}U)")
                    return 0.0
            except Exception:
                # 若估算失败，不强下单
                logger.warning(f"⚠️ 保证金估算失败，谨慎起见跳过 {symbol}")
                return 0.0

            logger.info(f"💵 单币分配: 模式=逐币下单, 余额={balance:.4f}U, 因子={factor:.2f}, 本币目标={target:.4f}U")
            return target

        except Exception as e:
            logger.error(f"❌ 计算{symbol}下单金额失败: {e}")
            return 0.0
    
    def create_order(self, symbol: str, side: str, amount: float) -> bool:
        """创建订单"""
        try:
            if self.has_open_orders(symbol):
                logger.warning(f"⚠️ {symbol}存在未成交订单，先取消")
                self.cancel_all_orders(symbol)
                time.sleep(1)

            if amount <= 0:
                logger.warning(f"⚠️ {symbol}下单金额为0，跳过")
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
                logger.error(f"❌ 无法获取{symbol}有效价格，跳过下单")
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

            # 预估保证金并预缩量：减少 51008 重试
            try:
                lev = float(self.symbol_leverage.get(symbol, 20) or 20)
                est_cost0 = float(contract_size * current_price)
                est_margin0 = est_cost0 / max(1.0, lev)
                avail = float(self.get_account_balance() or 0.0)
                # 预留一点安全系数（98%）
                if avail > 0 and est_margin0 > avail * 0.98:
                    ratio = (avail * 0.98 * lev) / max(1e-12, est_cost0)
                    # 按比例缩数量
                    contract_size = max(0.0, contract_size * max(0.1, min(1.0, ratio)))
                    # 对齐步进与精度
                    if lot_sz:
                        try:
                            step_pre = float(lot_sz)
                            if step_pre and step_pre > 0:
                                contract_size = math.ceil(contract_size / step_pre) * step_pre
                        except Exception:
                            pass
                    contract_size = round(contract_size, amount_precision)
                    # 不低于 minSz
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

            logger.info(f"📝 准备下单: {symbol} {side} 金额:{amount:.4f}U 价格:{current_price:.4f} 数量:{contract_size:.8f}")
            try:
                est_cost = contract_size * current_price
                logger.info(f"🧮 下单成本对齐: 分配金额={amount:.4f}U | 预计成本={est_cost:.4f}U | 数量={contract_size:.8f} | minSz={min_amount} | lotSz={lot_sz}")
            except Exception:
                pass

            pos_side = 'long' if side == 'buy' else 'short'
            order_id = None
            last_err = None

            native_only = (os.environ.get('USE_OKX_NATIVE_ONLY', '').strip().lower() in ('1', 'true', 'yes'))

            if not native_only:
                # CCXT方式
                try:
                    params = {'type': 'market', 'reduceOnly': False, 'posSide': pos_side}
                    order = self.exchange.create_order(symbol, 'market', side, contract_size, params=params)
                    order_id = order.get('id')
                except Exception as e:
                    last_err = e
                    logger.warning(f"⚠️ CCXT下单失败: {str(e)} - 尝试OKX原生API")
            
            if order_id is None:
                # OKX原生方式
                try:
                    pos_mode = self.get_position_mode()
                    if pos_mode == 'hedge':
                        td_mode = 'cross'
                        pos_side_okx = pos_side
                    else:
                        td_mode = 'cross'
                        pos_side_okx = 'net'
                    
                    params_okx = {
                        'instId': inst_id,
                        'tdMode': td_mode,
                        'side': side,
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
            return True
        except Exception as e:
            logger.error(f"❌ 创建订单失败 {symbol}: {str(e)}")
            return False
    
    def _set_initial_sl_tp(self, symbol: str, entry: float, atr: float, side: str) -> bool:
        """初始化 SL/TP（基于 ATR 与每币参数 n/m），写入 sl_tp_state"""
        try:
            cfg = self.symbol_cfg.get(symbol, {})
            n = float(cfg.get('n', 2.0))
            m = float(cfg.get('m', 3.0))
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
        """基于峰值/谷值与每币参数动态推进追踪止损（只更新内存态，重挂由冷却机制执行）"""
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

            # 达到激活阈值后才开始追踪
            activated = False
            if side == 'long':
                activated = (price >= entry * (1 + trigger_pct))
                # 维护峰值
                prev_peak = float(self.trailing_peak.get(symbol, entry) or entry)
                now_basis = price if basis == 'close' else price  # 简化：无高低价时用 close
                peak = max(prev_peak, now_basis)
                self.trailing_peak[symbol] = peak
                if activated:
                    # 新SL跟随峰值下方 trail_pct
                    new_sl = peak * (1 - trail_pct)
                    # 仅在提高SL（更接近当前价）时更新
                    if new_sl > float(st.get('sl', 0) or 0):
                        st['sl'] = new_sl
            else:  # short
                activated = (price <= entry * (1 - trigger_pct))
                # 维护谷值
                prev_trough = float(self.trailing_trough.get(symbol, entry) or entry)
                now_basis = price if basis == 'close' else price
                trough = min(prev_trough, now_basis)
                self.trailing_trough[symbol] = trough
                if activated:
                    # 新SL（空头）跟随谷值上方 trail_pct
                    new_sl = trough * (1 + trail_pct)
                    # 仅在降低SL（更接近当前价方向）时更新
                    cur_sl = float(st.get('sl', 0) or 0)
                    if cur_sl == 0 or new_sl < cur_sl:
                        st['sl'] = new_sl
        except Exception as e:
            logger.debug(f"🔧 追踪止损更新异常 {symbol}: {e}")

    def _check_hard_stop(self, symbol: str, price: float, side: str) -> bool:
        """硬止损/止盈校验（只返回布尔结果与日志，不直接平仓）"""
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
        """占位：程序内分批止盈（交易所侧当前为全仓TP）；如需交易所分批，需改为多档条件单"""
        try:
            # 可在达到 >m*ATR 时，将 TP 适度前移以提高触发概率（示例，不强制执行）
            st = self.sl_tp_state.get(symbol)
            if not st or atr <= 0:
                return
            entry = float(st.get('entry', 0) or 0)
            profit = (price - entry) if side == 'long' else (entry - price)
            # 轻微前移TP示例：盈利>2.0*ATR时，把TP向当前价靠近10%
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

    def place_okx_tp_sl(self, symbol: str, entry: float, side: str, atr: float = 0.0) -> bool:
        """挂OKX侧TP/SL条件单"""
        try:
            inst_id = self.symbol_to_inst_id(symbol)
            if not inst_id:
                return False
            st = self.sl_tp_state.get(symbol, {})
            sl = st.get('sl', 0.0)
            tp = st.get('tp', 0.0)
            if sl <= 0 or tp <= 0:
                return False
            
            sl_ticks = self.tp_sl_min_delta_ticks
            px_prec = self.markets_info.get(symbol, {}).get('price_precision', 4)
            tick_sz = 10 ** (-px_prec)
            # 应用 1H 多头 TP 放大倍数（仅多头适用）
            boost = float(self.tp_boost_map.get(symbol, 1.0) or 1.0)
            if side == 'long' and boost > 1.0:
                try:
                    tp *= boost
                except Exception:
                    pass
            sl = round(sl, px_prec)
            tp = round(tp, px_prec)
            
            cl_prefix = self.tpsl_cl_prefix or 'TPSL_'
            clid_sl = f"{cl_prefix}SL_{random.randint(1000,9999)}"
            clid_tp = f"{cl_prefix}TP_{random.randint(1000,9999)}"
            
            params_oco = {
                'instId': inst_id,
                'ordType': 'oco',
                'side': 'sell' if side == 'long' else 'buy',
                'posSide': side,
                'tdMode': 'cross',
                'tpTriggerPx': str(tp),
                'tpOrdPx': '-1',  # 市价
                'slTriggerPx': str(sl),
                'slOrdPx': '-1',  # 市价
                'closeFraction': '1',  # 全仓触发
            }
            try:
                resp_oco = self.exchange.privatePostTradeOrderAlgo(params_oco)
                data_oco = resp_oco.get('data', [])[0] if resp_oco.get('data') else {}
                if data_oco.get('sCode', '1') != '0':
                    logger.warning(f"⚠️ 挂OCO失败 {symbol}: {data_oco.get('sMsg', '')}")
                    return False
            except Exception as e:
                logger.warning(f"⚠️ 挂OCO异常 {symbol}: {str(e)}")
                return False
            self.okx_tp_sl_placed[symbol] = True
            self.tp_sl_last_placed[symbol] = time.time()
            logger.info(f"✅ 挂OCO成功 {symbol}: SL={sl:.6f} TP={tp:.6f}")
            return True
        except Exception as e:
            logger.error(f"❌ 挂TP/SL失败 {symbol}: {str(e)}")
            return False
    
    def calculate_volatility(self, df):
        """计算波动率（用于动态调整参数）"""
        returns = df['close'].pct_change()
        volatility = returns.std() * 100  # 百分比
        return volatility
    
    def calculate_indicators(self, df, symbol):
        """计算技术指标"""
        macd_p = self.macd_params.get(symbol, {'fast': 6, 'slow': 16, 'signal': 9})
        rsi_p = self.rsi_params.get(symbol, 9)
        
        # MACD
        ema_fast = df['close'].ewm(span=macd_p['fast'], adjust=False).mean()
        ema_slow = df['close'].ewm(span=macd_p['slow'], adjust=False).mean()
        df['macd_diff'] = ema_fast - ema_slow
        df['macd_dea'] = df['macd_diff'].ewm(span=macd_p['signal'], adjust=False).mean()
        df['macd_histogram'] = df['macd_diff'] - df['macd_dea']
        
        # RSI
        df = calculate_rsi(df, rsi_p)
        
        # 成交量
        df['volume_ma'] = df['volume'].rolling(window=5).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # 波动率
        df['volatility'] = df['close'].pct_change().rolling(window=20).std() * 100
        
        # ATR（真实波动幅度）
        df = calculate_atr(df, 14)
        
        # EMA均线族
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        return df
    
    def detect_divergence(self, df, lookback=25):
        """增强版背离检测"""
        if len(df) < lookback:
            return {'bullish': False, 'bearish': False, 'strength': 0}
        
        recent_df = df.tail(lookback)
        
        price_lows = []
        macd_lows = []
        price_highs = []
        macd_highs = []
        
        for i in range(3, len(recent_df) - 3):
            # 更严格的极值检测（前后3根K线）
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
        
        # 底背离
        bullish_div = False
        div_strength = 0
        if len(price_lows) >= 2 and len(macd_lows) >= 2:
            last_price_low = price_lows[-1][1]
            prev_price_low = price_lows[-2][1]
            last_macd_low = macd_lows[-1][1]
            prev_macd_low = macd_lows[-2][1]
            
            if last_price_low < prev_price_low and last_macd_low > prev_macd_low:
                bullish_div = True
                # 计算背离强度
                price_change = (prev_price_low - last_price_low) / prev_price_low
                macd_change = (last_macd_low - prev_macd_low) / abs(prev_macd_low)
                div_strength = (price_change + macd_change) * 100
        
        # 顶背离
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
        """趋势识别系统"""
        latest = df.iloc[-1]
        
        # 多重均线趋势
        ema_trend = 'up' if latest['ema_9'] > latest['ema_20'] > latest['ema_50'] else \
                   ('down' if latest['ema_9'] < latest['ema_20'] < latest['ema_50'] else 'neutral')
        
        # MACD趋势
        macd_trend = 'up' if latest['macd_diff'] > 0 and latest['macd_histogram'] > 0 else \
                    ('down' if latest['macd_diff'] < 0 and latest['macd_histogram'] < 0 else 'neutral')
        
        # 价格相对均线位置
        price_position = 'above' if latest['close'] > latest['ema_20'] else 'below'
        
        return {
            'ema_trend': ema_trend,
            'macd_trend': macd_trend,
            'price_position': price_position,
            'strong_trend': ema_trend == macd_trend and ema_trend != 'neutral'
        }
    
    def get_category(self, symbol: str) -> str:
        """返回币种分类（blue_chip/mainnet/infrastructure/emerging/meme），默认 unknown"""
        try:
            for cat, lst in (self.coin_categories or {}).items():
                if symbol in lst:
                    return cat
        except Exception:
            pass
        return 'unknown'

    def check_long_signal(self, df, symbol):
        """优化版做多信号检测"""
        if len(df) < 5:
            return False, "数据不足", 0
        
        # 指标列存在性校验，避免 KeyError
        required_cols = ['macd_diff','macd_dea','macd_histogram','rsi','ema_20','volume','volume_ma','volume_ratio']
        for col in required_cols:
            if col not in df.columns:
                return False, "指标缺失", 0
        
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        
        thresholds = self.rsi_thresholds.get(symbol, {'overbought': 70, 'oversold': 30})
        divergence = self.detect_divergence(df)
        trend = self.check_trend(df)
        category = self.get_category(symbol)
        
        signal_strength = 0  # 信号强度评分（0-100）
        
        # === 策略1: 底背离（最强信号）===
        if divergence['bullish']:
            if (latest['rsi'] < thresholds['oversold'] + 10 and 
                latest['rsi'] > previous['rsi'] and
                latest['macd_histogram'] > previous['macd_histogram']):
                signal_strength = 90 + min(divergence['strength'], 10)
                return True, f"🔥底背离(强度{divergence['strength']:.1f})", signal_strength
        
        # === 策略2: MACD金叉 + RSI确认 ===
        golden_cross = (
            previous['macd_diff'] <= previous['macd_dea'] and
            latest['macd_diff'] > latest['macd_dea'] and
            latest['macd_histogram'] > 0
        )
        
        if golden_cross:
            # 零轴下方金叉（抄底）
            if latest['macd_diff'] < 0:
                if (latest['rsi'] > thresholds['oversold'] and 
                    latest['rsi'] < 50 and
                    latest['volume_ratio'] > 1.2):
                    signal_strength = 75
                    return True, "MACD零轴下金叉（抄底）", signal_strength
            
            # 零轴上方金叉（趋势确认）
            elif trend['strong_trend'] and trend['ema_trend'] == 'up':
                if latest['rsi'] > 50:
                    signal_strength = 80
                    return True, "MACD零轴上金叉（趋势）", signal_strength
        
        # === 策略3: 零轴突破 ===
        if (previous['macd_diff'] < 0 and latest['macd_diff'] > 0 and
            latest['rsi'] > 50 and trend['price_position'] == 'above'):
            signal_strength = 70
            return True, "MACD零轴突破", signal_strength
        
        # === 策略4: RSI超卖反弹（针对MEME币）===
        if category == 'meme':
            if (latest['rsi'] < thresholds['oversold'] and
                latest['rsi'] > previous['rsi'] and
                latest['macd_histogram'] > previous['macd_histogram'] and
                latest['volume_ratio'] > 1.5):
                signal_strength = 65
                return True, "RSI超卖反弹（MEME）", signal_strength
        
        return False, "", 0
    
    def check_short_signal(self, df, symbol):
        """优化版做空信号检测"""
        if len(df) < 5:
            return False, "数据不足", 0
        
        # 指标列存在性校验，避免 KeyError
        required_cols = ['macd_diff','macd_dea','macd_histogram','rsi','ema_20','volume','volume_ma','volume_ratio']
        for col in required_cols:
            if col not in df.columns:
                return False, "指标缺失", 0
        
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        
        thresholds = self.rsi_thresholds.get(symbol, {'overbought': 70, 'oversold': 30})
        divergence = self.detect_divergence(df)
        trend = self.check_trend(df)
        category = self.get_category(symbol)
        
        signal_strength = 0
        
        # === 策略1: 顶背离 ===
        if divergence['bearish']:
            if (latest['rsi'] > thresholds['overbought'] - 10 and
                latest['rsi'] < previous['rsi'] and
                latest['macd_histogram'] < previous['macd_histogram']):
                signal_strength = 90 + min(divergence['strength'], 10)
                return True, f"🔥顶背离(强度{divergence['strength']:.1f})", signal_strength
        
        # === 策略2: MACD死叉 ===
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
                    return True, "MACD零轴上死叉（逃顶）", signal_strength
            
            elif trend['strong_trend'] and trend['ema_trend'] == 'down':
                if latest['rsi'] < 50:
                    signal_strength = 80
                    return True, "MACD零轴下死叉（趋势）", signal_strength
        
        # === 策略3: 零轴下破 ===
        if (previous['macd_diff'] > 0 and latest['macd_diff'] < 0 and
            latest['rsi'] < 50 and trend['price_position'] == 'below'):
            signal_strength = 70
            return True, "MACD零轴下破", signal_strength
        
        # === 策略4: RSI超买反弹（针对MEME币）===
        if category == 'meme':
            if (latest['rsi'] > thresholds['overbought'] and
                latest['rsi'] < previous['rsi'] and
                latest['macd_histogram'] < previous['macd_histogram'] and
                latest['volume_ratio'] > 1.5):
                signal_strength = 65
                return True, "RSI超买反弹（MEME）", signal_strength
        
        return False, "", 0
    
    def calculate_position_size(self, symbol, entry_price, signal_strength):
        """动态仓位计算（根据信号强度和币种权重）"""
        try:
            balance = self.get_account_balance()
            usdt_balance = balance
            
            # 基础风险金额
            base_risk = usdt_balance * (self.risk_percent / 100)
            
            # 币种权重调整
            weight = self.position_weights.get(symbol, 1.0)
            
            # 信号强度调整（60-100分对应0.8-1.2倍）
            strength_multiplier = 0.8 + (signal_strength - 60) / 100
            strength_multiplier = max(0.8, min(1.2, strength_multiplier))
            
            # 最终风险金额
            adjusted_risk = base_risk * weight * strength_multiplier
            
            # 计算仓位
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
                logger.warning(f"⚠️ 仓位计算错误，跳过 {symbol}")
                return
            
            # 模拟下单（实盘时取消注释）
            # order = self.exchange.create_market_order(symbol, side, position_size)
            
            # 计算止损止盈
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
            
            # 记录持仓
            self.positions[symbol] = {
                'side': side,
                'entry_price': entry_price,
                'size': position_size,
                'stop_loss': stop_loss_price,
                'take_profits': take_profit_prices,
                'tp_filled': [False, False, False],
                'entry_time': datetime.now(),
                'entry_reason': reason,
                'signal_strength': signal_strength,
                'category': category,
                'macd_diff': latest['macd_diff'],
                'rsi': latest['rsi']
            }
            
            # 显示开仓信息
            emoji = '📈' if side == 'buy' else '📉'
            category_emoji = {'blue_chip': '💎', 'mainnet': '⛓️', 'infrastructure': '🏗️', 
                            'emerging': '🌱', 'meme': '🐸'}.get(category, '❓')
            
            logger.info(f"\n{'='*70}")
            logger.info(f"✅ {category_emoji} 开仓成功！")
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
                # 实盘时使用
                # ticker = self.exchange.fetch_ticker(symbol)
                # current_price = ticker['last']
                
                # 模拟价格变动（实盘时删除）
                import random
                current_price = pos['entry_price'] * (1 + random.uniform(-0.03, 0.03))
                
                pnl_percent = 0
                if pos['side'] == 'buy':
                    pnl_percent = (current_price - pos['entry_price']) / pos['entry_price'] * 100
                    
                    # 止损
                    if current_price <= pos['stop_loss']:
                        self.close_position(symbol, f"止损 ({pnl_percent:.2f}%)", pnl_percent)
                        continue
                    
                    # 移动止损（盈利超过第一止盈点后，移动止损到成本价）
                    if current_price >= pos['take_profits'][0] and pos['stop_loss'] < pos['entry_price']:
                        pos['stop_loss'] = pos['entry_price'] * 0.998  # 保本+0.2%
                        logger.info(f"📌 移动止损: {symbol} 止损移至保本价 ${pos['stop_loss']:.6f}")
                    
                    # 分批止盈
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
            
            # 实盘时取消注释
            # self.exchange.create_market_order(symbol, side, close_size)
            
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
            
            # 实盘时取消注释
            # self.exchange.create_market_order(symbol, side, pos['size'])
            
            if pnl_percent < 0:
                self.trade_stats[symbol]['losses'] += 1
            else:
                self.trade_stats[symbol]['wins'] += 1
            
            self.trade_stats[symbol]['total_pnl'] += pnl_percent
            
            del self.positions[symbol]
            
            emoji = "🔴" if pnl_percent < 0 else "🟢"
            logger.info(f"{emoji} 平仓: {symbol} - {reason}")
            
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
        now = datetime.now()
        hour = now.hour
        
        # 避开时段
        avoid_hours = list(range(0, 2)) + list(range(8, 10))
        
        # 周末流动性差（针对小币种）
        if now.weekday() >= 5:  # 周六日
            return False
        
        return hour not in avoid_hours
    
    def adaptive_parameter_adjustment(self, symbol, df):
        """自适应参数调整（高级功能）"""
        volatility = self.calculate_volatility(df)
        
        # 如果波动率突然增加50%以上，临时放宽止损
        avg_volatility = df['volatility'].tail(50).mean()
        if volatility > avg_volatility * 1.5:
            adjusted_sl = self.stop_loss[symbol] * 1.3
            logger.info(f"⚠️ {symbol} 波动率异常 ({volatility:.2f}% vs {avg_volatility:.2f}%), 止损放宽至 {adjusted_sl:.1f}%")
            return adjusted_sl
        
        return self.stop_loss[symbol]
    
    def check_correlation(self):
        """检查币种相关性（防止过度集中）"""
        if len(self.positions) < 2:
            return True
        
        # 检查是否所有持仓都是MEME币（高风险）
        meme_count = sum(1 for pos in self.positions.values() if pos['category'] == 'meme')
        if meme_count >= 3:
            logger.info(f"⚠️ MEME币持仓过多 ({meme_count}/3)，暂停新的MEME币交易")
            return False
        
        return True
    
    def analyze_symbol(self, symbol: str) -> Dict[str, str]:
        """分析符号信号（加入 1H 趋势门控与严格策略模式过滤）"""
        try:
            df = self.get_klines(symbol, 150)
            if df.empty or len(df) < 50:
                return {'signal': 'hold', 'reason': '数据不足'}
            
            df = self.calculate_indicators(df, symbol)
            # 边界保护：去除初期 NaN 行，确保指标完整
            df = df.dropna()
            if df.empty or len(df) < 5:
                return {'signal': 'hold', 'reason': '数据不足'}
            current_position = self.get_position(symbol, force_refresh=False)
            
            # 1H 趋势门控：计算 1小时 MACD 与 RSI
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
                tp_boost_hint = False
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
                    latest_diff_1h = float(macd_diff_1h.iloc[-1])
                    latest_dea_1h = float(macd_dea_1h.iloc[-1])
                    latest_rsi_1h = float(rsi1h.iloc[-1])
                    bullish_1h = (latest_diff_1h > latest_dea_1h and latest_rsi_1h > 50)
                    bearish_1h = (latest_diff_1h < latest_dea_1h and latest_rsi_1h < 50)
                    if bullish_1h:
                        min_long_strength = 60   # 降低做多阈值
                        self.tp_boost_map[symbol] = 1.5  # 1H多头：TP放大1.5x
                    elif bearish_1h:
                        allow_long = False       # 暂停做多，仅允许做空
                        self.tp_boost_map[symbol] = 1.0
                    else:
                        self.tp_boost_map[symbol] = 1.0
                else:
                    allow_long = True
                    allow_short = True
                    min_long_strength = 65
                    min_short_strength = 65
                    tp_boost_hint = False
            except Exception:
                allow_long = True
                allow_short = True
                min_long_strength = 65
                min_short_strength = 65
                tp_boost_hint = False
            
            # 评估信号
            if current_position['size'] == 0:
                long_signal, long_reason, long_strength = self.check_long_signal(df, symbol)
                short_signal, short_reason, short_strength = self.check_short_signal(df, symbol)
                
                # 严格策略模式过滤
                mode = self.strategy_mode_map.get(symbol, 'combo')
                def _is_mode_ok(reason: str, side: str) -> bool:
                    r = reason or ''
                    if mode == 'zero_cross':
                        return ('零轴突破' in r) if side == 'buy' else ('零轴下破' in r)
                    if mode == 'divergence':
                        return ('背离' in r)
                    if mode == 'golden_cross':
                        return ('金叉' in r) if side == 'buy' else ('死叉' in r)
                    return True  # combo
                if long_signal and allow_long and long_strength >= min_long_strength and _is_mode_ok(long_reason, 'buy'):
                    if float(self.tp_boost_map.get(symbol, 1.0) or 1.0) > 1.0:
                        logger.info(f"🌟 1H多头趋势：{symbol} 做多TP目标已放大至 1.5x")
                    return {'signal': 'buy', 'reason': long_reason}
                if short_signal and allow_short and short_strength >= min_short_strength and _is_mode_ok(short_reason, 'sell'):
                    return {'signal': 'sell', 'reason': short_reason}
                return {'signal': 'hold', 'reason': '无信号'}
            else:
                # 对于持仓，检查反向平仓信号（模式不过滤平仓）
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
    
    def execute_strategy(self):
        """执行策略"""
        logger.info("=" * 70)
        logger.info(f"🚀 开始执行MACD+RSI策略 (11个币种，{self.timeframe} 周期)")
        logger.info("=" * 70)
        
        try:
            self.check_sync_needed()
            
            balance = self.get_account_balance()
            logger.info(f"💰 当前账户余额: {balance:.2f} USDT")
            # 熔断机制已移除
            
            logger.info(self.stats.get_summary())
            
            self.display_current_positions()
            
            self.manage_positions()
            
            logger.info("🔍 分析交易信号...")
            logger.info("-" * 70)
            
            signals = {}
            for symbol in self.symbols:
                signals[symbol] = self.analyze_symbol(symbol)
                position = self.get_position(symbol, force_refresh=False)
                open_orders = self.get_open_orders(symbol)
                
                status_line = f"📊 {symbol}: 信号={signals[symbol]['signal']}, 原因={signals[symbol]['reason']}"
                if open_orders:
                    status_line += f", 挂单={len(open_orders)}个"
                
                logger.info(status_line)
                # 每币种之间加入微延时，降低瞬时并发
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
                
                try:
                    kl = self.get_klines(symbol, 50)
                    if not kl.empty:
                        close_price = float(kl.iloc[-1]['close'])
                        ps = self.per_symbol_params.get(symbol, {})
                        atr_p = int(ps.get('atr_period', 14))
                        atr_val = calculate_atr(kl, atr_p)['atr'].iloc[-1]
                        if current_position['size'] > 0 and atr_val > 0:
                            # 若是手动持仓或尚未初始化SL/TP，这里兜底初始化并在OKX侧挂出TP/SL
                            st0 = self.sl_tp_state.get(symbol)
                            if not st0:
                                try:
                                    entry0 = float(current_position.get('entry_price', 0) or 0)
                                    if entry0 > 0:
                                        self._set_initial_sl_tp(symbol, entry0, atr_val, current_position.get('side', 'long'))
                                        okx_ok = self.place_okx_tp_sl(symbol, entry0, current_position.get('side', 'long'), atr_val)
                                        if okx_ok:
                                            logger.info(f"📌 手动/历史持仓兜底：已初始化并挂TP/SL {symbol}")
                                        else:
                                            logger.warning(f"⚠️ 手动/历史持仓兜底挂单失败 {symbol}")
                                except Exception as _e0:
                                    logger.warning(f"⚠️ 兜底初始化SL/TP异常 {symbol}: {_e0}")
                            side_now = current_position.get('side', 'long')
                            self._update_trailing_stop(symbol, close_price, atr_val, side_now)
                            # 硬止损兜底
                            if self._check_hard_stop(symbol, close_price, side_now):
                                current_position = self.get_position(symbol, force_refresh=True)
                                continue
                            # 分批止盈
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
                                try:
                                    # 仅在超过冷却时间时重挂TP/SL，避免频繁撤销/重挂
                                    last_ts = self.tp_sl_last_placed.get(symbol, 0.0)
                                    if (time.time() - last_ts) >= float(self.tp_sl_refresh_interval):
                                        try:
                                            self.cancel_symbol_tp_sl(symbol)
                                        except Exception:
                                            pass
                                        entry_px2 = float(self.sl_tp_state.get(symbol, {}).get('entry', 0) or 0)
                                        okx_ok = False
                                        if entry_px2 > 0:
                                            okx_ok = self.place_okx_tp_sl(symbol, entry_px2, side_now, atr_val)
                                        if okx_ok:
                                            logger.info(f"🔄 更新追踪止盈：冷却达到，已重挂 {symbol}")
                                        else:
                                            logger.warning(f"⚠️ 更新追踪止盈重挂失败 {symbol}")
                                    else:
                                        logger.debug(f"⏳ 距上次挂单未达冷却({self.tp_sl_refresh_interval}s)，跳过重挂 {symbol}")
                                except Exception as _e:
                                    logger.warning(f"⚠️ 更新追踪止盈重挂失败 {symbol}: {_e}")
                                if side_now == 'long':
                                    if close_price <= st['sl'] or close_price >= st['tp']:
                                        logger.info(f"⛔ 触发SL/TP多头 {symbol}: 价={close_price:.6f} SL={st['sl']:.6f} TP={st['tp']:.6f}")
                                        self.close_position(symbol, open_reverse=False)
                                        current_position = self.get_position(symbol, force_refresh=True)
                                        continue
                                else:
                                    if close_price >= st['sl'] or close_price <= st['tp']:
                                        logger.info(f"⛔ 触发SL/TP空头 {symbol}: 价={close_price:.6f} SL={st['sl']:.6f} TP={st['tp']:.6f}")
                                        self.close_position(symbol, open_reverse=False)
                                        current_position = self.get_position(symbol, force_refresh=True)
                                        continue
                except Exception:
                    pass
                
                if signal == 'buy':
                    if current_position['size'] > 0 and current_position['side'] == 'long':
                        logger.info(f"ℹ️ {symbol}已有多头持仓，跳过重复开仓")
                        continue
                    
                    amount = self.calculate_order_amount(symbol)
                    if amount > 0:
                        if self.create_order(symbol, 'buy', amount):
                            logger.info(f"🚀 开多{symbol}成功 - {reason}")
                            self.last_position_state[symbol] = 'long'
                
                elif signal == 'sell':
                    if current_position['size'] > 0 and current_position['side'] == 'short':
                        logger.info(f"ℹ️ {symbol}已有空头持仓，跳过重复开仓")
                        continue
                    
                    amount = self.calculate_order_amount(symbol)
                    if amount > 0:
                        if self.create_order(symbol, 'sell', amount):
                            logger.info(f"📉 开空{symbol}成功 - {reason}")
                            self.last_position_state[symbol] = 'short'
                
                elif signal == 'close':
                    _pp = self.per_symbol_params.get(symbol, {})
                    allow_reverse = bool(_pp.get('allow_reverse', True)) if isinstance(_pp, dict) else True
                    if self.close_position(symbol, open_reverse=allow_reverse):
                        if allow_reverse:
                            logger.info(f"✅ 平仓并反手开仓 {symbol} 成功 - {reason}")
                        else:
                            logger.info(f"✅ 平仓完成（不反手） {symbol} - {reason}")
            
            logger.info("=" * 70)
                        
        except Exception as e:
            logger.error(f"❌ 执行策略失败: {e}")
    
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
        logger.info("⏰ 刷新方式: 实时巡检（每interval秒执行一次，可用环境变量 SCAN_INTERVAL 调整，默认1秒）")
        logger.info(f"🔄 状态同步: 每{self.sync_interval}秒")
        logger.info(f"📊 监控币种: {', '.join(self.symbols)}")
        logger.info(f"💡 11个币种特性: 支持0.1U起的小额交易，平均分配资金")
        logger.info(self.stats.get_summary())
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
                logger.info("⛔ 用户中断，策略停止")
                break
            except Exception as e:
                logger.error(f"❌ 策略运行异常: {e}")
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

        # 环境变量生效情况打印
        def _get(k, default=''):
            v = os.environ.get(k, '')
            return v if (v is not None and str(v).strip() != '') else default
        logger.info(f"🔧 变量: SCAN_INTERVAL={_get('SCAN_INTERVAL','2')} OKX_API_MIN_INTERVAL={_get('OKX_API_MIN_INTERVAL','0.2')} SYMBOL_LOOP_DELAY={_get('SYMBOL_LOOP_DELAY','0.3')} SET_LEVERAGE_ON_START={_get('SET_LEVERAGE_ON_START','true')}")
        logger.info(f"🔧 变量: MAX_RETRIES={_get('MAX_RETRIES','3')} BACKOFF_BASE={_get('BACKOFF_BASE','0.8')} BACKOFF_MAX={_get('BACKOFF_MAX','3.0')} TP_SL_REFRESH_INTERVAL={_get('TP_SL_REFRESH_INTERVAL','300')}")

        try:
            scan_interval_env = os.environ.get('SCAN_INTERVAL', '').strip()
            scan_interval = int(scan_interval_env) if scan_interval_env else 2
            if scan_interval <= 0:
                scan_interval = 1
        except Exception:
            scan_interval = 1
        logger.info(f"🛠 扫描间隔设置: {scan_interval} 秒（可用环境变量 SCAN_INTERVAL 覆盖）")
        strategy.run_continuous(interval=scan_interval)
        
    except Exception as e:
        logger.error(f"❌ 策略初始化或运行失败: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
