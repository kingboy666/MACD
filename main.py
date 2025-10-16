#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MACD策略实现 - RAILWAY平台版本
扩展到11个币种，包含BTC/ETH/SOL/DOGE/XRP/PEPE/ARB
25倍杠杆，无限制交易，带挂单识别和状态同步
增加胜率统计和盈亏显示
进一步优化版：增强模块化(1)、性能(2)、错误处理(3)、日志(5)、其他(9)；TP/SL&BB验证无问题，但添加更多日志和dry-run模拟
新增：布林带开口过滤（>0.8*mean保留信号） 与 动态止盈调节（趋势强时放宽TP距离）
修复：检测posMode并调整posSide参数，避免one-way模式错误
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

# 工具函数：从env获取值并转换
def _get_env_str(key: str, default: str = '') -> str:
    return os.environ.get(key, default).strip()

def _get_env_float(key: str, default: float = 0.0) -> float:
    try:
        return float(_get_env_str(key, str(default)))
    except ValueError:
        return default

def _get_env_int(key: str, default: int = 0) -> int:
    try:
        return int(_get_env_str(key, str(default)))
    except ValueError:
        return default

def _get_env_bool(key: str, default: bool = False) -> bool:
    val = _get_env_str(key, '').lower()
    return val in ('1', 'true', 'yes') if val else default

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
    """MACD策略类 - 扩展到11个币种"""
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
        self.timeframe = '5m'
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
        self.sync_interval: int = _get_env_int('SYNC_INTERVAL', 60)
        
        # 市场信息缓存
        self.markets_info: Dict[str, Dict[str, Any]] = {}
        
        # API 速率限制
        self._last_api_ts: float = 0.0
        self._min_api_interval: float = _get_env_float('OKX_API_MIN_INTERVAL', 0.2)
        # 下单安全系数（控制名义额度占可用保证金的比例），默认0.80
        self.order_safety_factor: float = _get_env_float('ORDER_SAFETY_FACTOR', 0.80)

        # 每币种微延时，降低瞬时调用密度
        self.symbol_loop_delay = _get_env_float('SYMBOL_LOOP_DELAY', 0.3)
        # 启动时是否逐币设置杠杆（默认 False，避免 59669 导致启动失败；需要统一杠杆时可临时设为 True）
        self.set_leverage_on_start = _get_env_bool('SET_LEVERAGE_ON_START', False)
        
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
        self.bb_tp_offset = _get_env_float('BB_TP_OFFSET', 0.003)
        self.bb_sl_offset = _get_env_float('BB_SL_OFFSET', 0.002)
        # 止损参数（布林优先 + ATR 兜底）
        self.min_sl_pct = _get_env_float('MIN_SL_PCT', 0.12)  # 兜底扩大为12%
        # 追踪止损激活门槛：盈利达到这两者较大值才开始推进SL
        self.trail_activate_atr = _get_env_float('TRAIL_ACTIVATE_BY_ATR', 1.8)   # ≥1.8×ATR（更晚启动）
        self.trail_activate_pct = _get_env_float('TRAIL_ACTIVATE_PCT', 0.02)     # 或 ≥2.0%
        # 动态地板参数（用于震荡行情自适应加宽初始SL）
        self.sl_floor_k_atr = _get_env_float('SL_FLOOR_ATR_K', 1.6)
        self.sl_floor_c_bw = _get_env_float('SL_FLOOR_BW_C', 0.8)
        self.base_sl_pct_main = _get_env_float('BASE_SL_PCT_MAIN', 0.09)
        self.base_sl_pct_mid = _get_env_float('BASE_SL_PCT_MID', 0.12)
        self.base_sl_pct_high = _get_env_float('BASE_SL_PCT_HIGH', 0.15)
        # SAR掉头平仓参数
        self.use_sar_flip_exit = _get_env_bool('USE_SAR_FLIP_EXIT', True)
        self.sar_confirm_bars = _get_env_int('SAR_CONFIRM_BARS', 1)
        self.sar_min_cross_pct = _get_env_float('SAR_MIN_CROSS_PCT', 0.003)
        # 波动分级（可按需扩充/调整）
        self.symbol_vol_tier: Dict[str, str] = {
            'BTC/USDT:USDT': 'main',
            'ETH/USDT:USDT': 'main',
            'SOL/USDT:USDT': 'main',
            'XRP/USDT:USDT': 'mid',
            'ARB/USDT:USDT': 'mid',
            'ZRO/USDT:USDT': 'mid',
            'WLD/USDT:USDT': 'mid',
            'FIL/USDT:USDT': 'mid',
            'WIF/USDT:USDT': 'high',
            'DOGE/USDT:USDT': 'high',
            'PEPE/USDT:USDT': 'high',
        }

        # 统一以XRP模板应用到所有交易对（运行期覆盖，不改动原参数存储）
        self.apply_xrp_template_all = _get_env_bool('APPLY_XRP_FOR_ALL', True)
        self.xrp_symbol = 'XRP/USDT:USDT'
        
        # 启动基线余额与风控参数
        self.starting_balance = self.get_account_balance() or 0.0
        # 暂时关闭硬止损（可用环境变量覆盖重新开启）
        self.hard_sl_max_loss_pct = _get_env_float('HARD_SL_MAX_LOSS_PCT', 0.0)
        self.account_dd_limit_pct = _get_env_float('ACCOUNT_DD_LIMIT_PCT', 0.20)  # 20%
        self.cb_close_all = _get_env_bool('CB_CLOSE_ALL', True)
        # 强制彻底关闭账户熔断
        self.cb_enabled = False
        self.circuit_breaker_triggered = False
        self.partial_tp_done: Dict[str, set] = {}
        # 撤单/标记 安全控制
        self.allow_cancel_pending = _get_env_bool('ALLOW_CANCEL_PENDING', True)
        self.safe_cancel_only_our_tpsl = _get_env_bool('SAFE_CANCEL_ONLY_OUR_TPSL', True)
        self.tpsl_cl_prefix = _get_env_str('TPSL_CL_PREFIX', 'MACD_TPSL_')
        # 是否为算法单携带 algoClOrdId（默认关闭以规避部分账户 51000 报错）
        self.use_algo_client_id = _get_env_bool('USE_ALGO_CLIENT_ID', False)
        
        # ATR 止盈止损参数
        self.atr_sl_n = _get_env_float('ATR_SL_N', 1.8)
        self.atr_tp_m = _get_env_float('ATR_TP_M', 2.2)
        
        # SL/TP 状态缓存
        self.sl_tp_state: Dict[str, Dict[str, float]] = {}
        self.okx_tp_sl_placed: Dict[str, bool] = {}
        # TP/SL重挂冷却与阈值
        self.tp_sl_last_placed: Dict[str, float] = {}
        self.tp_sl_refresh_interval = _get_env_int('TP_SL_REFRESH_INTERVAL', 300)
        self.tp_sl_min_delta_ticks = _get_env_int('TP_SL_MIN_DELTA_TICKS', 2)
        
        # ===== 每币种配置(用于追踪止损) =====
        self.symbol_cfg: Dict[str, Dict[str, float | str]] = {
            # 原有币种
            "ZRO/USDT:USDT": {"period": 14, "n": 3.8, "m": 3.5, "trigger_pct": 0.020, "trail_pct": 0.0025, "update_basis": "high"},
            "WIF/USDT:USDT": {"period": 14, "n": 4.2, "m": 4.5, "trigger_pct": 0.027, "trail_pct": 0.003, "update_basis": "high"},
            "WLD/USDT:USDT": {"period": 14, "n": 3.8, "m": 3.5, "trigger_pct": 0.020, "trail_pct": 0.0025, "update_basis": "close"},
            "FIL/USDT:USDT": {"period": 14, "n": 3.8, "m": 3.5, "trigger_pct": 0.020, "trail_pct": 0.0025, "update_basis": "high"},
            
            # 新增主流币
            "BTC/USDT:USDT": {"period": 20, "n": 3.0, "m": 3.5, "trigger_pct": 0.016, "trail_pct": 0.002, "update_basis": "high"},
            "ETH/USDT:USDT": {"period": 18, "n": 3.2, "m": 4.0, "trigger_pct": 0.018, "trail_pct": 0.0025, "update_basis": "high"},
            "SOL/USDT:USDT": {"period": 16, "n": 3.2, "m": 4.0, "trigger_pct": 0.018, "trail_pct": 0.0025, "update_basis": "high"},
            "XRP/USDT:USDT": {"period": 16, "n": 3.8, "m": 3.5, "trigger_pct": 0.020, "trail_pct": 0.0025, "update_basis": "close"},
            
            # 新增Meme币
            "DOGE/USDT:USDT": {"period": 16, "n": 4.2, "m": 4.5, "trigger_pct": 0.027, "trail_pct": 0.003, "update_basis": "high"},
            "PEPE/USDT:USDT": {"period": 14, "n": 4.2, "m": 6.0, "trigger_pct": 0.032, "trail_pct": 0.005, "update_basis": "high"},
            
            # 新增L2币
            "ARB/USDT:USDT": {"period": 15, "n": 3.2, "m": 3.8, "trigger_pct": 0.022, "trail_pct": 0.003, "update_basis": "high"}
        }
        
        # 跟踪峰值/谷值
        self.trailing_peak: Dict[str, float] = {}
        self.trailing_trough: Dict[str, float] = {}
        # 交易执行冷却与阶段追踪状态
        self.last_trade_candle_index: Dict[str, int] = {}
        self.stage3_done: Dict[str, bool] = {}
        # 信号增强配置（可用环境变量覆盖）
        self.ma_type = _get_env_str('MA_TYPE', 'sma').lower()
        self.ma_fast = _get_env_int('MA_FAST', 5)
        self.ma_slow = _get_env_int('MA_SLOW', 20)
        self.vol_ma_period = _get_env_int('VOL_MA_PERIOD', 20)
        self.vol_boost = _get_env_float('VOL_BOOST', 1.2)
        self.long_body_pct = _get_env_float('LONG_BODY_PCT', 0.6)
        self.cooldown_candles = _get_env_int('COOLDOWN_CANDLES', 3)
        # 三阶段追踪与最小阈值
        self.trail_stage_1 = _get_env_float('TRAIL_STAGE_1', 1.0)
        self.trail_stage_2 = _get_env_float('TRAIL_STAGE_2', 1.75)
        self.trail_stage_3 = _get_env_float('TRAIL_STAGE_3', 2.5)
        self.trail_stage2_offset = _get_env_float('TRAIL_STAGE2_OFFSET', 0.8)
        self.trail_sl_min_delta_atr = _get_env_float('TRAIL_SL_MIN_DELTA_ATR', 0.2)
        self.partial_tp_ratio_stage3 = _get_env_float('PARTIAL_TP_RATIO_STAGE3', 0.3)
        self.allow_strong_pa_override = _get_env_bool('ALLOW_STRONG_PA_OVERRIDE', True)
        # —— 滚仓配置（强趋势开口时金字塔加仓）——
        self.pyramid_max_adds = _get_env_int('PYRAMID_MAX_ADDS', 3)
        self.pyramid_step_atr = _get_env_float('PYRAMID_STEP_ATR', 0.8)      # 与上次加仓价的最小ATR增幅
        self.pyramid_min_gap_pct = _get_env_float('PYRAMID_MIN_GAP_PCT', 0.006)  # 与上次加仓价的最小百分比间距
        self.pyramid_cooldown_s = _get_env_int('PYRAMID_COOLDOWN_S', 120)
        # 规模因子（逐次递减）
        self.pyramid_size_factors = json.loads(_get_env_str('PYRAMID_SIZE_FACTORS', '[0.5,0.35,0.25]'))
        # 滚仓状态
        self.pyramid_count: Dict[str, int] = {}
        self.pyramid_last_add_px: Dict[str, float] = {}
        self.pyramid_last_add_ts: Dict[str, float] = {}
        
        # 记录上次持仓状态
        self.last_position_state: Dict[str, str] = {}
        
        # Dry-run模式
        self.dry_run = _get_env_bool('DRY_RUN', False)
        if self.dry_run:
            logger.warning("⚠️ DRY_RUN模式启用：模拟交易，不实际下单")
        
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
        now = time.time()
        delta = now - self._last_api_ts
        if delta < self._min_api_interval:
            jitter = np.random.uniform(0, self._min_api_interval * 0.1)
            time.sleep(self._min_api_interval - delta + jitter)
        self._last_api_ts = time.time()

    def _safe_call(self, func, *args, **kwargs):
        """
        包装API调用：先节流；遇到50011(Too Many Requests)执行指数退避重试。
        """
        max_retries = _get_env_int('MAX_RETRIES', 3)
        backoff_base = _get_env_float('BACKOFF_BASE', 0.8)
        backoff_max = _get_env_float('BACKOFF_MAX', 3.0)

        for i in range(max_retries + 1):
            try:
                self._sleep_with_throttle()
                return func(*args, **kwargs)
            except Exception as e:
                msg = str(e)
                is_rate = ('50011' in msg) or ('Too Many Requests' in msg)
                if not is_rate or i >= max_retries:
                    logger.error(f"❌ API调用失败: {msg}\n{traceback.format_exc()}")
                    raise
                wait = min(backoff_max, backoff_base * (2 ** i)) + np.random.uniform(0, 0.2)
                logger.warning(f"⏳ 限频(50011) 第{i+1}次重试，等待 {wait:.2f}s")
                time.sleep(wait)
        return None

    def _setup_exchange(self):
        """设置交易所配置"""
        try:
            self.exchange.check_required_credentials()
            self.exchange.version = 'v5'
            opts = self.exchange.options or {}
            opts.update({'defaultType': 'swap', 'defaultSettle': 'USDT', 'version': 'v5'})
            self.exchange.options = opts
            logger.info("✅ API连接验证成功")
            
            self.sync_exchange_time()
            
            self.exchange.load_markets(True, {'type': 'swap'})
            logger.info("✅ 预加载市场数据完成 (swap)")
            
            # 尝试设置hedge模式
            try:
                self.exchange.set_position_mode(True)
                logger.info("✅ 设置为双向持仓模式（多空分开）")
            except Exception as e:
                logger.warning(f"⚠️ 设置持仓模式失败（可能有持仓或已设置）： {str(e)}")
            
            # 无论是否设置成功，都获取当前模式
            config = self._safe_call(self.exchange.privateGetAccountConfig)
            if config:
                pos_mode = config.get('data', [{}])[0].get('posMode', '')
                self.is_hedge_mode = (pos_mode == 'long_short_mode')
                logger.info(f"ℹ️ 当前持仓模式: {'hedge (long_short_mode)' if self.is_hedge_mode else 'one-way (net_mode)'}")
            else:
                self.is_hedge_mode = False
                logger.warning("⚠️ 无法获取持仓模式，假设one-way模式")
            
            # 按交易对设置杠杆（仅在与目标不一致时设置；失败仅告警跳过，避免 59669 终止初始化）
            if self.set_leverage_on_start:
                for symbol in self.symbols:
                    try:
                        target_lev = float(self.symbol_leverage.get(symbol, 20))
                        inst_id = self.symbol_to_inst_id(symbol)
                        cur = self.get_current_leverage(symbol)
                        if self.is_hedge_mode:
                            need_set_long = cur.get('long') is None or abs(cur.get('long', 0.0) - target_lev) > 1e-9
                            need_set_short = cur.get('short') is None or abs(cur.get('short', 0.0) - target_lev) > 1e-9
                            if not need_set_long and not need_set_short:
                                logger.info(f"ℹ️ 杠杆一致(hedge) 跳过 {symbol}: long={cur.get('long')} short={cur.get('short')} 目标={target_lev}")
                                continue
                            leverage_params = {'instId': inst_id, 'lever': f"{target_lev}", 'mgnMode': 'cross'}
                            if need_set_long:
                                try:
                                    self._safe_call(self.exchange.privatePostAccountSetLeverage, {**leverage_params, 'posSide': 'long'})
                                    logger.info(f"✅ 已设置{symbol} long 杠杆为{target_lev}倍")
                                except Exception as eL:
                                    emsg = str(eL)
                                    if '59669' in emsg:
                                        logger.warning(f"⚠️ 跳过设置杠杆 {symbol} long: 59669（交叉保证金条件单/追踪/TP/SL/机器人）保持现状")
                                    else:
                                        logger.warning(f"⚠️ 跳过设置杠杆 {symbol} long: {emsg}")
                            if need_set_short:
                                try:
                                    self._safe_call(self.exchange.privatePostAccountSetLeverage, {**leverage_params, 'posSide': 'short'})
                                    logger.info(f"✅ 已设置{symbol} short 杠杆为{target_lev}倍")
                                except Exception as eS:
                                    emsg = str(eS)
                                    if '59669' in emsg:
                                        logger.warning(f"⚠️ 跳过设置杠杆 {symbol} short: 59669（交叉保证金条件单/追踪/TP/SL/机器人）保持现状")
                                    else:
                                        logger.warning(f"⚠️ 跳过设置杠杆 {symbol} short: {emsg}")
                        else:
                            cur_any = cur.get('any')
                            if cur_any is not None and abs(cur_any - target_lev) <= 1e-9:
                                logger.info(f"ℹ️ 杠杆一致(one-way) 跳过 {symbol}: 当前={cur_any} 目标={target_lev}")
                                continue
                            leverage_params = {'instId': inst_id, 'lever': f"{target_lev}", 'mgnMode': 'cross'}
                            try:
                                self._safe_call(self.exchange.privatePostAccountSetLeverage, leverage_params)
                                logger.info(f"✅ 已设置{symbol} 杠杆为{target_lev}倍（one-way）")
                            except Exception as eO:
                                emsg = str(eO)
                                if '59669' in emsg:
                                    logger.warning(f"⚠️ 跳过设置杠杆 {symbol}: 59669（交叉保证金条件单/追踪/TP/SL/机器人）保持现状")
                                else:
                                    logger.warning(f"⚠️ 跳过设置杠杆 {symbol}: {emsg}")
                    except Exception as e_loop:
                        logger.warning(f"⚠️ 设置杠杆环节异常（已跳过）{symbol}: {str(e_loop)}")
                        continue
            
        except Exception as e:
            logger.error(f"❌ 交易所设置失败: {str(e)} - {traceback.format_exc()}")
            raise

    def get_current_leverage(self, symbol: str) -> Dict[str, Optional[float]]:
        """
        查询OKX当前杠杆信息：
        - 对冲模式：分别返回 long / short 的杠杆
        - 单向模式：返回 any（同一个数）
        """
        try:
            inst_id = self.symbol_to_inst_id(symbol)
            if not inst_id:
                return {'long': None, 'short': None, 'any': None}
            resp = self._safe_call(self.exchange.privateGetAccountLeverageInfo, {'instId': inst_id, 'mgnMode': 'cross'})
            data = (resp or {}).get('data', [])
            cur_long: Optional[float] = None
            cur_short: Optional[float] = None
            cur_any: Optional[float] = None
            for it in data:
                if it.get('instId') != inst_id:
                    continue
                ps = str(it.get('posSide') or '').lower()
                lev_val = None
                for v in (it.get('lever'), it.get('leverLong'), it.get('leverShort')):
                    try:
                        if v is not None:
                            lev_val = float(v)
                            break
                    except Exception:
                        continue
                if ps == 'long':
                    cur_long = lev_val
                elif ps == 'short':
                    cur_short = lev_val
                else:
                    cur_any = lev_val
            return {'long': cur_long, 'short': cur_short, 'any': cur_any}
        except Exception as e:
            logger.warning(f"⚠️ 查询当前杠杆失败 {symbol}: {str(e)}")
            return {'long': None, 'short': None, 'any': None}

    def _load_markets(self):
        """加载市场信息"""
        try:
            logger.info("📄 加载市场信息...")
            resp = self._safe_call(self.exchange.publicGetPublicInstruments, {'instType': 'SWAP'})
            data = resp.get('data', [])
            spec_map = {it['instId']: it for it in data if it.get('settleCcy') == 'USDT'}
            for symbol in self.symbols:
                inst_id = self.symbol_to_inst_id(symbol)
                it = spec_map.get(inst_id, {})
                min_sz = float(it.get('minSz', 0)) or 0.000001
                lot_sz = float(it.get('lotSz', 0)) or None
                tick_sz = float(it.get('tickSz', 0)) or 0.0001
                amt_prec = len(str(lot_sz).split('.')[-1]) if lot_sz and '.' in str(lot_sz) else 8
                px_prec = len(str(tick_sz).split('.')[-1]) if '.' in str(tick_sz) else 4
                self.markets_info[symbol] = {
                    'min_amount': min_sz,
                    'min_cost': 0.0,
                    'amount_precision': amt_prec,
                    'price_precision': px_prec,
                    'lot_size': lot_sz,
                    'max_market_size': (float(it.get('maxMktSz', 0)) if it.get('maxMktSz') is not None else 0.0) or None,
                }
                logger.info(f"📊 {symbol} - 最小数量:{min_sz:.8f} 步进:{(lot_sz or 0):.8f} Tick:{tick_sz:.8f}")
            logger.info("✅ 市场信息加载完成")
        except Exception as e:
            logger.error(f"❌ 加载市场信息失败: {str(e)} - {traceback.format_exc()}")
            for symbol in self.symbols:
                self.markets_info[symbol] = {
                    'min_amount': 0.000001,
                    'min_cost': 0.1,
                    'amount_precision': 8,
                    'price_precision': 4,
                    'lot_size': None,
                    'max_market_size': None,
                }
    
    def sync_exchange_time(self):
        """同步交易所时间"""
        try:
            server_time = int(self._safe_call(self.exchange.fetch_time) or 0)
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
            logger.error(f"❌ 同步时间失败: {str(e)} - {traceback.format_exc()}")
            return 0
    
    def get_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        """获取未成交订单"""
        try:
            inst_id = self.symbol_to_inst_id(symbol)
            resp = self._safe_call(self.exchange.privateGetTradeOrdersPending, {'instType': 'SWAP', 'instId': inst_id})
            data = resp.get('data', [])
            results = []
            for o in data:
                results.append({
                    'id': o.get('ordId') or o.get('clOrdId'),
                    'side': 'buy' if o.get('side') == 'buy' else 'sell',
                    'amount': float(o.get('sz', 0)),
                    'price': float(o.get('px', 0)) if o.get('px') else None,
                })
            return results
        except Exception as e:
            logger.error(f"❌ 获取{symbol}挂单失败: {str(e)} - {traceback.format_exc()}")
            return []
    
    def cancel_all_orders(self, symbol: str) -> bool:
        """取消所有未成交订单"""
        try:
            orders = self.get_open_orders(symbol)
            if not orders:
                return True
            
            for order in orders:
                self._safe_call(self.exchange.cancel_order, order['id'], symbol)
                logger.info(f"✅ 取消订单: {symbol} {order['id']}")
            
            return True
        except Exception as e:
            logger.error(f"❌ 批量取消订单失败: {str(e)} - {traceback.format_exc()}")
            return False

    def cancel_symbol_tp_sl(self, symbol: str) -> bool:
        """撤销该交易对在OKX侧已挂的TP/SL（算法单）。仅撤本程序挂的单（clOrdId前缀），携带 instId，按 ordType 分组撤销。"""
        try:
            inst_id = self.symbol_to_inst_id(symbol)
            if not inst_id:
                return True
            # OKX v5 要求 ordType 必填；为兼容策略中使用的 oco/trigger（以及部分场景的 conditional），循环查询合并
            data = []
            for _ord in ('oco', 'trigger', 'conditional'):
                try:
                    resp = self._safe_call(
                        self.exchange.privateGetTradeOrdersAlgoPending,
                        {'instType': 'SWAP', 'instId': inst_id, 'ordType': _ord}
                    )
                    data.extend(resp.get('data', []))
                except Exception as _e:
                    # 若某 ordType 不支持或无数据，忽略即可
                    continue
            groups: Dict[str, List[Dict[str, str]]] = {}
            # 使用与下单一致的“清洗前缀”进行匹配（仅[A-Za-z0-9_-]）
            safe_prefix = re.sub('[^A-Za-z0-9_-]', '', self.tpsl_cl_prefix or '')
            # 若为对冲模式，且当前有持仓，则仅撤对应posSide的条件单；否则不按posSide过滤
            desired_pos_side = None
            if self.is_hedge_mode:
                pos_now = self.get_position(symbol, force_refresh=True)
                if pos_now.get('size', 0) > 0:
                    desired_pos_side = pos_now.get('side')  # 'long' or 'short'
            for it in data:
                ord_type = str(it.get('ordType', '')).lower()
                if not ord_type:
                    continue
                clid = str(it.get('algoClOrdId') or it.get('clOrdId', ''))
                if self.safe_cancel_only_our_tpsl and self.use_algo_client_id and safe_prefix and not clid.startswith(safe_prefix):
                    continue
                its_pos_side = str(it.get('posSide') or '').lower()
                if desired_pos_side and its_pos_side and its_pos_side != desired_pos_side:
                    continue
                aid = it.get('algoId') or it.get('algoID') or it.get('id')
                if aid:
                    groups.setdefault(ord_type, []).append({'algoId': str(aid), 'clOrdId': clid})
            if not groups:
                return True
            total = 0
            # 逐个 algoId 撤销，避免批量 JSON 结构导致 50002
            for ord_type, items in groups.items():
                for obj in items:
                    aid = obj['algoId']
                    try:
                        mapped = ('oco' if ord_type in ('tp','sl','oco') else ('trigger' if ord_type == 'trigger' else ('move_order_stop' if ord_type in ('trailing','move_order_stop','move_stop') else 'conditional')))
                        payload_okx = {'algoIds': [{'algoId': str(aid)}], 'ordType': mapped, 'instId': inst_id}
                        self._safe_call(self.exchange.privatePostTradeCancelAlgos, payload_okx)
                        total += 1
                    except Exception as _e:
                        logger.warning(f"⚠️ 撤销失败 {symbol}: ordType={mapped} algoId={aid} err={_e}")
                        continue
            if total > 0:
                logger.info(f"✅ 撤销 {symbol} 条件单数量: {total}")
                return True
            return False
        except Exception as e:
            logger.warning(f"⚠️ 撤销 {symbol} 条件单失败: {str(e)} - {traceback.format_exc()}")
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
                    kl = self.get_klines(symbol, 50)
                    ps = self.per_symbol_params.get(symbol, {})
                    atr_p = ps.get('atr_period', 14)
                    atr_val = self.calculate_atr(kl, atr_p) if kl else 0.0
                    entry = position.get('entry_price', 0)
                    if atr_val > 0 and entry > 0:
                        okx_ok = self.place_okx_tp_sl(symbol, entry, position.get('side', 'long'), atr_val)
                        if okx_ok:
                            logger.info(f"📌 已为已有持仓补挂TP/SL {symbol}")
                        else:
                            logger.warning(f"⚠️ 补挂交易所侧TP/SL失败 {symbol}")
                    has_positions = True
                
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
            logger.error(f"❌ 同步状态失败: {str(e)} - {traceback.format_exc()}")
    
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
            resp = self._safe_call(self.exchange.privateGetAccountBalance, {})
            data = resp.get('data', [])
            avail = 0.0
            for acc in data:
                for d in acc.get('details', []):
                    if d.get('ccy') == 'USDT':
                        # 优先使用合约账户可用权益(更贴近可用保证金)，回退到余额字段
                        v = d.get('availEq') or d.get('availBal') or d.get('cashBal') or '0'
                        avail = float(v)
                        break
            return avail
        except Exception as e:
            logger.error(f"❌ 获取账户余额失败: {str(e)} - {traceback.format_exc()}")
            return 0.0
    
    def get_klines(self, symbol: str, limit: int = 100) -> List[Dict]:
        """获取K线数据，使用缓存"""
        try:
            now = time.time()
            cache = self._klines_cache.get(symbol, {})
            if cache and now - list(cache.keys())[0] < self._klines_ttl:
                return cache[list(cache.keys())[0]]
            
            inst_id = self.symbol_to_inst_id(symbol)
            tf = self.timeframe_map.get(symbol, self.timeframe)
            params = {'instId': inst_id, 'bar': tf, 'limit': str(limit)}
            resp = self._safe_call(self.exchange.publicGetMarketCandles, params)
            rows = resp.get('data', [])
            result: List[Dict] = []
            for r in rows:
                ts = int(r[0])
                o = float(r[1]); h = float(r[2]); l = float(r[3]); c = float(r[4]); v = float(r[5])
                result.append({
                    'timestamp': pd.to_datetime(ts, unit='ms'),
                    'open': o, 'high': h, 'low': l, 'close': c, 'volume': v
                })
            result.sort(key=lambda x: x['timestamp'])
            self._klines_cache[symbol] = {now: result}
            return result
        except Exception as e:
            logger.error(f"❌ 获取{symbol}K线数据失败: {str(e)} - {traceback.format_exc()}")
            return []
    
    def get_position(self, symbol: str, force_refresh: bool = False) -> Dict[str, Any]:
        """获取当前持仓"""
        try:
            if not force_refresh and symbol in self.positions_cache:
                return self.positions_cache[symbol]
            
            inst_id = self.symbol_to_inst_id(symbol)
            resp = self._safe_call(self.exchange.privateGetAccountPositions, {'instType': 'SWAP', 'instId': inst_id})
            data = resp.get('data', [])
            for p in data:
                if p.get('instId') == inst_id:
                    pos = float(p.get('pos', 0) or 0)
                    if pos == 0:
                        continue
                    size = abs(pos)
                    if self.is_hedge_mode:
                        side = 'long' if p.get('posSide') == 'long' else 'short'
                    else:
                        # 单向(net)模式：依据仓位正负判断方向
                        side = 'long' if pos > 0 else 'short'
                    entry_price = float(p.get('avgPx', 0))
                    leverage = float(p.get('lever', 0))
                    unreal = float(p.get('upl', 0))
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
            logger.error(f"❌ 获取{symbol}持仓失败: {str(e)} - {traceback.format_exc()}")
            return self.positions_cache.get(symbol, {'size': 0, 'side': 'none', 'entry_price': 0, 'unrealized_pnl': 0, 'leverage': 0})
    
    def has_open_orders(self, symbol: str) -> bool:
        """检查是否有未成交订单"""
        try:
            orders = self.get_open_orders(symbol)
            has_orders = len(orders) > 0
            if has_orders:
                logger.info(f"⚠️ {symbol} 存在{len(orders)}个未成交订单")
            return has_orders
        except Exception as e:
            logger.error(f"❌ 检查挂单失败: {str(e)} - {traceback.format_exc()}")
            return False
    
    def check_margin_sufficiency(self, symbol: str, amount: float) -> bool:
        """检查保证金是否足够，避免51008错误"""
        try:
            # 获取当前价格
            inst_id = self.symbol_to_inst_id(symbol)
            tkr = self._safe_call(self.exchange.publicGetMarketTicker, {'instId': inst_id})
            d = tkr.get('data', [{}])[0]
            current_price = float(d.get('last') or d.get('lastPx') or 0.0)
            
            if current_price <= 0:
                logger.error(f"❌ 无法获取{symbol}有效价格，无法检查保证金")
                return False
            
            # 获取杠杆倍数
            leverage = self.symbol_leverage.get(symbol, 20)
            
            # 计算所需保证金
            required_margin = amount / leverage
            
            # 获取可用余额
            available_balance = self.get_account_balance()
            
            # 保留额外20%作为缓冲
            safe_margin = available_balance * 0.8
            
            # 检查保证金是否足够
            if required_margin > safe_margin:
                logger.warning(f"⚠️ {symbol}保证金不足: 需要{required_margin:.4f}U, 可用{safe_margin:.4f}U (总余额:{available_balance:.4f}U)")
                return False
            
            # 检查最小保证金要求（调整为0.05U以适应小额账户）
            min_margin = max(0.05, _get_env_float('MIN_MARGIN_USDT', 0.05))
            if required_margin < min_margin:
                logger.warning(f"⚠️ {symbol}保证金低于最小要求{min_margin}U: {required_margin:.4f}U")
                return False
            
            logger.info(f"✅ {symbol}保证金检查通过: 需要{required_margin:.4f}U, 可用{safe_margin:.4f}U")
            return True
            
        except Exception as e:
            logger.error(f"❌ 检查{symbol}保证金失败: {str(e)} - {traceback.format_exc()}")
            return False

    def calculate_order_amount(self, symbol: str, active_count: Optional[int] = None) -> float:
        """计算下单金额（不均分）：基于“实时余额×安全系数×单笔比例”，并保证名义≥0.5U×杠杆。"""
        try:
            # 1) 固定目标名义金额（最高优先）
            target_str = _get_env_str('TARGET_NOTIONAL_USDT')
            if target_str:
                target = max(0.1, float(target_str))
                logger.info(f"💵 使用固定目标名义金额: {target:.4f}U")
                return target

            # 2) 实时可用余额
            balance = self.get_account_balance()
            if balance <= 0:
                logger.warning(f"⚠️ 余额不足，无法为 {symbol} 分配资金 (余额:{balance:.4f}U)")
                return 0.0

            # 3) 安全系数与单笔占用比例（串行下单，默认单笔使用“可用×安全系数×50%”）
            safety = float(getattr(self, 'order_safety_factor', 0.80) or 0.80)
            per_order_frac = _get_env_float('PER_ORDER_FRACTION', 0.50)
            per_order_frac = min(max(per_order_frac, 0.05), 1.0)  # 限定 5%~100%

            base_budget = balance * safety
            allocated_amount = base_budget * per_order_frac

            # 4) 名义地板：≥ 0.5U × 杠杆
            lev = float(self.symbol_leverage.get(symbol, 20))
            min_target_usdt = 0.5 * lev
            if allocated_amount < min_target_usdt:
                allocated_amount = min_target_usdt

            # 5) 上限保护（可选 env）
            max_cap = max(0.0, _get_env_float('MAX_PER_SYMBOL_USDT', 0.0))
            if max_cap > 0 and allocated_amount > max_cap:
                allocated_amount = max_cap

            logger.info(f"💵 分配(不均分): 余额={balance:.4f}U 安全系数={safety:.2f} 单笔比例={per_order_frac:.2f} → 目标名义={allocated_amount:.4f}U (地板={min_target_usdt:.4f}U)")
            return float(max(allocated_amount, 0.0))
        except Exception as e:
            logger.error(f"❌ 计算{symbol}下单金额失败: {str(e)} - {traceback.format_exc()}")
            return 0.0
    
    def create_order(self, symbol: str, side: str, amount: float) -> bool:
        """创建订单"""
        try:
            if self.dry_run:
                logger.info(f"🧪 [DRY_RUN] 模拟下单: {symbol} {side} 金额:{amount:.4f}U")
                return True
            
            # 1. 预检查保证金是否足够
            if not self.check_margin_sufficiency(symbol, amount):
                logger.error(f"❌ 可用保证金不足以满足{amount:.4f}U或minSz，放弃下单 {symbol}")
                return False
            
            if self.has_open_orders(symbol):
                logger.warning(f"⚠️ {symbol}存在未成交订单，先取消")
                self.cancel_all_orders(symbol)
                time.sleep(1)

            if amount <= 0:
                logger.warning(f"⚠️ {symbol}下单金额为0，跳过")
                return False

            market_info = self.markets_info.get(symbol, {})
            min_amount = market_info.get('min_amount', 0.001)
            amount_precision = market_info.get('amount_precision', 8)
            lot_sz = market_info.get('lot_size')
            # 修复: 确保 step 在后续任何分支使用前已定义，避免 UnboundLocalError
            step = float(lot_sz) if lot_sz else 0.0

            inst_id = self.symbol_to_inst_id(symbol)
            tkr = self._safe_call(self.exchange.publicGetMarketTicker, {'instId': inst_id})
            d = tkr.get('data', [{}])[0]
            current_price = float(d.get('last') or d.get('lastPx') or 0.0)

            if current_price <= 0:
                logger.error(f"❌ 无法获取{symbol}有效价格，跳过下单")
                return False

            contract_size = amount / current_price

            if contract_size < min_amount:
                contract_size = min_amount

            if lot_sz:
                step = float(lot_sz)
                if step > 0:
                    contract_size = math.ceil(contract_size / step) * step
            contract_size = round(contract_size, amount_precision)

            if contract_size <= 0 or contract_size < min_amount:
                contract_size = max(min_amount, 10 ** (-amount_precision))
                if lot_sz and step > 0:
                    contract_size = math.ceil(contract_size / step) * step
                contract_size = round(contract_size, amount_precision)

            used_usdt = contract_size * current_price
            if used_usdt < amount:
                need_qty = (amount - used_usdt) / current_price
                incr_step = step if step > 0 else (10 ** (-amount_precision))
                add_qty = math.ceil(need_qty / incr_step) * incr_step
                contract_size = round(contract_size + add_qty, amount_precision)
                if contract_size < min_amount:
                    contract_size = max(min_amount, 10 ** (-amount_precision))
                    if lot_sz and step > 0:
                        contract_size = math.ceil(contract_size / step) * step
                    contract_size = round(contract_size, amount_precision)

            # 最低保证金阈值（调整为0.05U以适应小额账户）：确保名义金额>=阈值*杠杆
            lev = float(self.symbol_leverage.get(symbol, 20))
            min_margin_usdt = max(0.0, _get_env_float('MIN_MARGIN_USDT', 0.05))
            min_target_usdt = min_margin_usdt * lev
            base_target_usdt = max(amount, min_target_usdt)
            used_usdt = contract_size * current_price
            if used_usdt < base_target_usdt:
                need_qty = (base_target_usdt - used_usdt) / current_price
                incr_step = step if step > 0 else (10 ** (-amount_precision))
                add_qty = math.ceil(need_qty / incr_step) * incr_step
                contract_size = round(contract_size + add_qty, amount_precision)
                # 再次保证不低于交易所最小数量
                if contract_size < min_amount:
                    contract_size = max(min_amount, 10 ** (-amount_precision))
                    if lot_sz and step > 0:
                        contract_size = math.ceil(contract_size / step) * step
                    contract_size = round(contract_size, amount_precision)

            # 预估保证金并预缩量
            lev = self.symbol_leverage.get(symbol, 20)
            est_cost0 = contract_size * current_price
            est_margin0 = est_cost0 / max(1.0, lev)
            avail = self.get_account_balance()
            # 以可用保证金做硬上限：cap_qty = floor(((avail*0.80)*lev)/price, 到 lotSz 步进)
            cap_usdt = max(0.0, (avail * 0.80))
            cap_qty_raw = (cap_usdt * lev) / max(current_price, 1e-12)
            cap_qty = cap_qty_raw
            if step > 0:
                cap_qty = math.floor(cap_qty_raw / step) * step
            cap_qty = round(cap_qty, amount_precision)
            if cap_qty <= 0:
                logger.warning(f"⚠️ 可用保证金不足：avail={avail:.4f}U lev={lev} price={current_price:.6f} → 最大数量=0，跳过下单 {symbol}")
                return False
            if contract_size > cap_qty:
                logger.info(f"🔧 按可用保证金限额收缩数量: 原={contract_size:.8f} → 上限={cap_qty:.8f} (avail={avail:.4f}U lev={lev}x)")
                contract_size = cap_qty

            # 单笔市价单最大数量（maxMktSz）限幅
            max_mkt = self.markets_info.get(symbol, {}).get('max_market_size')
            if max_mkt and max_mkt > 0:
                if contract_size > max_mkt:
                    logger.info(f"🔧 按交易所单笔上限收缩数量: 原={contract_size:.8f} → 上限={max_mkt:.8f}")
                    contract_size = max_mkt
                    if step > 0:
                        contract_size = math.floor(contract_size / step) * step
                    contract_size = round(contract_size, amount_precision)

            # 兜底：不低于交易所最小数量
            if contract_size < min_amount:
                contract_size = min_amount
                if step > 0:
                    contract_size = math.ceil(contract_size / step) * step
                contract_size = round(contract_size, amount_precision)

            if contract_size <= 0:
                logger.warning(f"⚠️ {symbol}最终数量无效: {contract_size}")
                return False

            # 发单前的保证金硬校验（更保守，避免 51008）：若名义占用 > avail*0.60，则按比例收缩数量
            lev = float(self.symbol_leverage.get(symbol, 20))
            avail = self.get_account_balance()
            est_margin_check = (contract_size * current_price) / max(1.0, lev)
            margin_cap = max(0.0, avail * 0.60)
            if est_margin_check > margin_cap and contract_size > 0:
                shrink_ratio = margin_cap / max(est_margin_check, 1e-12)
                new_qty = contract_size * max(min(shrink_ratio, 1.0), 0.0)
                if step > 0:
                    new_qty = math.floor(new_qty / step) * step
                new_qty = round(new_qty, amount_precision)
                if new_qty < min_amount or new_qty <= 0:
                    logger.warning(f"⚠️ 保证金不足，收缩后低于最小数量，放弃下单 {symbol} (avail={avail:.4f}U)")
                    return False
                # 二次校验：确保每笔保证金不低于0.5U
                est_margin_after = (new_qty * current_price) / max(1.0, lev)
                if est_margin_after < 0.5:
                    logger.warning(f"⚠️ 收缩后保证金仍低于0.5U，放弃下单 {symbol} (est_margin={est_margin_after:.4f}U)")
                    return False
                logger.info(f"🔧 按保证金硬上限收缩数量: 原={contract_size:.8f} → {new_qty:.8f} (avail={avail:.4f}U lev={lev}x)")
                contract_size = new_qty

            # 3. 最终保证金检查（防止边界情况）
            lev = float(self.symbol_leverage.get(symbol, 20))
            final_margin = (contract_size * current_price) / max(1.0, lev)
            final_balance = self.get_account_balance()
            if final_margin > final_balance * 0.5:  # 使用50%作为安全阈值
                logger.warning(f"⚠️ 最终保证金检查失败: 需要{final_margin:.4f}U, 可用{final_balance:.4f}U, 放弃下单 {symbol}")
                return False

            logger.info(f"📝 准备下单: {symbol} {side} 金额:{amount:.4f}U 价格:{current_price:.4f} 数量:{contract_size:.8f}")
            est_cost = contract_size * current_price
            logger.info(f"🧮 下单成本对齐: 分配金额={amount:.4f}U | 预计成本={est_cost:.4f}U | 数量={contract_size:.8f} | minSz={min_amount} | lotSz={lot_sz}")

            pos_side = 'long' if side == 'buy' else 'short'
            native_only = True  # 强制走OKX原生接口，避免ccxt抽象差异

            if not native_only:
                # 使用ccxt下单
                order_type = 'market'
                params = {
                    'marginMode': 'cross',
                    'leverage': self.symbol_leverage.get(symbol, 20),
                }
                if self.is_hedge_mode:
                    params['positionSide'] = pos_side
                order = self._safe_call(self.exchange.create_order, symbol, order_type, side, contract_size, None, params)
                if order:
                    logger.info(f"✅ 下单成功: {symbol} {side} {contract_size:.8f} @{order_type}")
                    return True
                else:
                    logger.error(f"❌ 下单失败: {symbol} {side}")
                    return False
            else:
                # 使用OKX原生API下单
                payload = {
                    'instId': inst_id,
                    'tdMode': 'cross',
                    'side': side,
                    'ordType': 'market',
                    'sz': str(contract_size),
                    'lever': str(self.symbol_leverage.get(symbol, 20)),
                }
                if self.is_hedge_mode:
                    payload['posSide'] = pos_side
                # 原生下单 + 51008降规模重试（最多2次，每次减半数量，直到不低于minSz）
                def _try_place(qty: float) -> Optional[dict]:
                    pp = dict(payload)
                    pp['sz'] = f"{max(qty, min_amount)}"
                    return self._safe_call(self.exchange.privatePostTradeOrder, pp)
                attempt = 0
                qty = contract_size
                while attempt <= 2:
                    resp = None
                    try:
                        resp = _try_place(qty)
                        if resp and str(resp.get('code','')) == '0':
                            logger.info(f"✅ 原生下单成功: {symbol} {side} {qty:.8f}")
                            # 下单成功后立即尝试挂交易所侧TP/SL，避免等待下轮巡检
                            try:
                                pos_now = self.get_position(symbol, force_refresh=True)
                                if pos_now.get('size', 0) > 0:
                                    kl = self.get_klines(symbol, 50)
                                    ps = self.per_symbol_params.get(symbol, {})
                                    atr_p = ps.get('atr_period', 14)
                                    atr_val = self.calculate_atr(kl, atr_p) if kl else 0.0
                                    entry_px = pos_now.get('entry_price', 0.0)
                                    side_now = pos_now.get('side', 'long')
                                    if entry_px > 0 and atr_val > 0:
                                        # 初始化本地SL/TP状态
                                        self._set_initial_sl_tp(symbol, entry_px, atr_val, side_now)
                                        # 挂交易所侧TP/SL
                                        okx_ok = self.place_okx_tp_sl(symbol, entry_px, side_now, atr_val)
                                        if okx_ok:
                                            logger.info(f"📌 开仓即挂交易所侧TP/SL成功 {symbol}")
                                        else:
                                            logger.warning(f"⚠️ 开仓后挂交易所侧TP/SL失败 {symbol}")
                                    else:
                                        logger.debug(f"ℹ️ 开仓后TP/SL跳过：entry={entry_px} ATR={atr_val} {symbol}")
                            except Exception as _e:
                                logger.warning(f"⚠️ 开仓后挂TP/SL异常 {symbol}: {str(_e)}")
                            return True
                        else:
                            # 如果返回体包含 data.sCode=51008，也按不足处理
                            data = (resp or {}).get('data', []) if isinstance(resp, dict) else []
                            scode = str(data[0].get('sCode','')) if data else ''
                            if scode == '51008':
                                raise ccxt.InsufficientFunds('Insufficient margin')
                            logger.error(f"❌ 原生下单失败: {symbol} {side} - {resp}")
                            return False
                    except Exception as e:
                        emsg = str(e)
                        # 51008: 保证金不足；51202: 市价单数量超过最大值
                        if (('InsufficientFunds' in emsg or '51008' in emsg) or ('51202' in emsg)) and attempt < 2:
                            # 一次性计算可承载的安全数量（按 avail*0.60），并满足0.5U地板与minSz
                            avail_now = self.get_account_balance()
                            safe_cap_usdt = max(0.0, avail_now * 0.60)
                            # 单笔上限保护
                            max_qty_by_avail = (safe_cap_usdt * lev) / max(current_price, 1e-12)
                            if max_mkt and max_mkt > 0:
                                max_qty_by_avail = min(max_qty_by_avail, max_mkt)
                            if step > 0:
                                max_qty_by_avail = math.floor(max_qty_by_avail / step) * step
                            max_qty_by_avail = round(max_qty_by_avail, amount_precision)
                            # 满足0.5U保证金所需的最小数量
                            min_qty_for_floor = (0.5 * lev) / max(current_price, 1e-12)
                            if step > 0:
                                min_qty_for_floor = math.ceil(min_qty_for_floor / step) * step
                            min_qty_for_floor = max(min_amount, round(min_qty_for_floor, amount_precision))
                            if max_qty_by_avail < min_qty_for_floor or max_qty_by_avail <= 0:
                                logger.error(f"❌ 可用保证金不足以满足0.5U地板或minSz，放弃下单 {symbol} (avail={avail_now:.4f}U)")
                                return False
                            new_qty = min(qty, max_qty_by_avail)
                            # 对齐步进与精度
                            if step > 0:
                                new_qty = math.floor(new_qty / step) * step
                            new_qty = round(new_qty, amount_precision)
                            if new_qty < min_qty_for_floor:
                                new_qty = min_qty_for_floor
                            if new_qty < min_amount or new_qty <= 0:
                                logger.error(f"❌ 重新计算后的数量仍低于minSz，放弃下单 {symbol}")
                                return False
                            logger.warning(f"⚠️ {'51202上限' if '51202' in emsg else '51008保证金'}，按可用保证金一次性收缩重试: {qty:.8f} → {new_qty:.8f}")
                            qty = new_qty
                            attempt += 1
                            continue
                        logger.error(f"❌ 原生下单异常: {symbol} {side}: {emsg}")
                        return False

        except Exception as e:
            logger.error(f"❌ 下单失败 {symbol} {side}: {str(e)} - {traceback.format_exc()}")
            return False
    
    def close_position(self, symbol: str, open_reverse: bool = False) -> bool:
        """平仓"""
        try:
            if self.dry_run:
                logger.info(f"🧪 [DRY_RUN] 模拟平仓: {symbol} open_reverse={open_reverse}")
                return True
            
            position = self.get_position(symbol, force_refresh=True)
            if position['size'] == 0:
                logger.info(f"ℹ️ {symbol} 无持仓，跳过平仓")
                return True
            
            side = 'sell' if position['side'] == 'long' else 'buy'
            amount = position['size']
            
            if self.has_open_orders(symbol):
                self.cancel_all_orders(symbol)
                time.sleep(1)
            
            # 平仓（改为OKX原生接口，严格reduceOnly）
            inst_id = self.symbol_to_inst_id(symbol)
            payload = {
                'instId': inst_id,
                'tdMode': 'cross',
                'side': side,
                'ordType': 'market',
                'sz': f"{amount}",
                'reduceOnly': True,
            }
            if self.is_hedge_mode:
                payload['posSide'] = position['side']
            resp = self._safe_call(self.exchange.privatePostTradeOrder, payload)
            ok = isinstance(resp, dict) and str(resp.get('code', '')) == '0'
            if ok:
                pnl = position['unrealized_pnl']
                self.stats.add_trade(symbol, position['side'], pnl)
                logger.info(f"✅ 平仓成功: {symbol} {side} {amount:.6f} PNL:{pnl:.2f}U")
                
                # 清理缓存
                self.positions_cache[symbol] = {'size': 0, 'side': 'none', 'entry_price': 0, 'unrealized_pnl': 0, 'leverage': 0}
                self.sl_tp_state.pop(symbol, None)
                self.okx_tp_sl_placed.pop(symbol, None)
                self.tp_sl_last_placed.pop(symbol, None)
                self.trailing_peak.pop(symbol, None)
                self.trailing_trough.pop(symbol, None)
                self.stage3_done.pop(symbol, None)
                self.partial_tp_done.pop(symbol, None)
                self.last_position_state[symbol] = 'none'
                
                if open_reverse:
                    reverse_side = 'buy' if side == 'sell' else 'sell'
                    alloc_amount = self.calculate_order_amount(symbol)
                    if alloc_amount > 0:
                        self.create_order(symbol, reverse_side, alloc_amount)
                return True
            else:
                logger.error(f"❌ 平仓失败(原生): {symbol} - {resp}")
                return False
            
        except Exception as e:
            logger.error(f"❌ 平仓失败 {symbol}: {str(e)} - {traceback.format_exc()}")
            return False
    
    def reduce_only_market(self, symbol: str, side: str, qty: float, pos_side: str) -> bool:
        """减仓市价单"""
        try:
            if self.dry_run:
                logger.info(f"🧪 [DRY_RUN] 模拟减仓: {symbol} {side} {qty:.6f}")
                return True
            
            inst_id = self.symbol_to_inst_id(symbol)
            payload = {
                'instId': inst_id,
                'tdMode': 'cross',
                'side': side,
                'ordType': 'market',
                'sz': str(qty),
                'reduceOnly': True,
            }
            if self.is_hedge_mode:
                payload['posSide'] = pos_side
            resp = self._safe_call(self.exchange.privatePostTradeOrder, payload)
            if resp and resp.get('code') == '0':
                logger.info(f"✅ 减仓成功: {symbol} {side} {qty:.6f}")
                return True
            else:
                logger.error(f"❌ 减仓失败: {symbol} - {resp}")
                return False
        except Exception as e:
            logger.error(f"❌ 减仓失败 {symbol}: {str(e)} - {traceback.format_exc()}")
            return False
    
    def _set_initial_sl_tp(self, symbol: str, entry_price: float, atr_val: float, side: str):
        """设置初始SL/TP"""
        try:
            logger.debug(f"📍 初始化SL/TP {symbol}: entry={entry_price:.6f} ATR={atr_val:.6f} side={side}")
            strat = self.get_strategy_for(symbol)
            sl = None
            tp = None

            # 尝试BB/SAR (if applicable)
            if strat in ('bb_sar', 'hybrid'):
                kl = self.get_klines(symbol, 50)
                closes = [k['close'] for k in kl]
                ps_b = self.get_strategy_params(symbol)
                bb_period = ps_b.get('bb_period', 20)
                bb_k = ps_b.get('bb_k', 2.5 if strat == 'bb_sar' else 2.2)
                bb = self.calculate_bollinger_bands(closes, bb_period, bb_k) or {}
                sar_val = self.calculate_sar(
                    kl,
                    ps_b.get('sar_af_start', 0.01 if strat=='bb_sar' else 0.03),
                    ps_b.get('sar_af_max', 0.10 if strat=='bb_sar' else 0.25)
                )
                if bb:
                    upper = bb['upper']; middle = bb['middle']; lower = bb['lower']
                    band_width = bb['band_width']; band_ma20 = bb['band_ma20']
                    tp_offset = 1.005 if band_width > band_ma20 * 1.2 else 1.003
                    sl_offset = 1.0 - self.bb_sl_offset if side == 'long' else 1.0 + self.bb_sl_offset
                    if side == 'long':
                        tp = upper * tp_offset
                        # 融合BB+ATR：取更保守的更远SL
                        cfg2 = self.get_symbol_cfg(symbol)
                        n2 = cfg2.get('n', self.atr_sl_n)
                        c1 = entry_price - n2 * atr_val
                        c2 = lower - 0.5 * atr_val
                        sl_bb = max(sar_val or 0, middle * sl_offset) if sar_val else middle * sl_offset
                        sl = min(sl_bb, c1, c2)
                    else:
                        tp = lower * (2.0 - tp_offset)  # symmetric for short
                        cfg2 = self.get_symbol_cfg(symbol)
                        n2 = cfg2.get('n', self.atr_sl_n)
                        c1 = entry_price + n2 * atr_val
                        c2 = upper + 0.5 * atr_val
                        sl_bb = min(sar_val or 0, middle * (2.0 - sl_offset)) if sar_val else middle * (2.0 - sl_offset)
                        sl = max(sl_bb, c1, c2)

            if sl is None or tp is None:
                # 回退至ATR
                if atr_val <= 0:
                    return
                cfg = self.get_symbol_cfg(symbol)
                n = cfg['n']; m = cfg['m']
                tp_pct = cfg.get('tp_pct')
                if side == 'long':
                    sl = entry_price - n * atr_val
                    tp = entry_price * (1 + tp_pct) if tp_pct else entry_price + m * atr_val
                else:
                    sl = entry_price + n * atr_val
                    tp = entry_price * (1 - tp_pct) if tp_pct else entry_price - m * atr_val

            # 动态地板：floor_pct = max(分级基础、ATR项、带宽项)，震荡越大地板越宽
            try:
                # 1) 分级基础百分比
                tier = self.symbol_vol_tier.get(symbol, 'mid')
                if tier == 'high':
                    base_pct = self.base_sl_pct_high
                elif tier == 'main':
                    base_pct = self.base_sl_pct_main
                else:
                    base_pct = self.base_sl_pct_mid
                base_pct = max(0.0, float(base_pct))
                # 2) ATR项（按入场价比例）
                atr_comp = 0.0
                if entry_price > 0 and atr_val > 0:
                    atr_comp = max(0.0, float(self.sl_floor_k_atr) * (atr_val / entry_price))
                # 3) 布林带带宽项（带宽/中轨），有BB时生效
                bw_comp = 0.0
                try:
                    # 优先尝试使用同周期BB；若前面已算过 upper/middle/band_width 则可重用
                    kl2 = self.get_klines(symbol, 50)
                    closes2 = [k['close'] for k in kl2] if kl2 else []
                    ps_b2 = self.get_strategy_params(symbol) if hasattr(self, 'strategy_params') else {}
                    bb_period2 = ps_b2.get('bb_period', 20)
                    bb_k2 = ps_b2.get('bb_k', 2.0)
                    bb2 = self.calculate_bollinger_bands(closes2, bb_period2, bb_k2) if closes2 else None
                    if bb2:
                        mid2 = float(bb2.get('middle') or 0.0)
                        bw2 = float(bb2.get('band_width') or 0.0)
                        if mid2 > 0:
                            bw_comp = max(0.0, float(self.sl_floor_c_bw) * (bw2 / mid2))
                except Exception:
                    bw_comp = 0.0
                floor_pct = max(base_pct, atr_comp, bw_comp)
                # 应用动态地板
                if side == 'long':
                    floor_px = entry_price * (1 - floor_pct)
                    if sl > floor_px:
                        sl = floor_px
                else:
                    floor_px = entry_price * (1 + floor_pct)
                    if sl < floor_px:
                        sl = floor_px
            except Exception:
                # 回退到旧的固定地板（保证兼容）
                min_pct = max(0.0, getattr(self, 'min_sl_pct', 0.06))
                if side == 'long':
                    floor_px = entry_price * (1 - min_pct)
                    if sl > floor_px:
                        sl = floor_px
                else:
                    floor_px = entry_price * (1 + min_pct)
                    if sl < floor_px:
                        sl = floor_px

            # 写入状态
            side_num = 1.0 if side == 'long' else -1.0
            peak_init = entry_price if side == 'long' else float('inf')
            trough_init = entry_price if side == 'short' else float('-inf')
            self.trailing_peak[symbol] = max(self.trailing_peak.get(symbol, peak_init), entry_price)
            self.trailing_trough[symbol] = min(self.trailing_trough.get(symbol, trough_init), entry_price)
            self.sl_tp_state[symbol] = {'sl': float(sl), 'tp': float(tp), 'side': side_num, 'entry': float(entry_price)}
            logger.debug(f"📍 初始化完成 {symbol}: SL={sl:.6f} TP={tp:.6f}")
        except Exception as e:
            logger.warning(f"⚠️ 初始化SL/TP异常 {symbol}: {str(e)} - {traceback.format_exc()}")

    def _update_trailing_stop(self, symbol: str, current_price: float, atr_val: float, side: str):
        """动态移动止损"""
        try:
            st = self.sl_tp_state.get(symbol)
            if not st or atr_val <= 0 or current_price <= 0 or side not in ('long', 'short'):
                return
            cfg = self.get_symbol_cfg(symbol)
            n = cfg['n']; trigger_pct = cfg['trigger_pct']; trail_pct = cfg['trail_pct']
            # 覆盖追踪步长（仅特定币）
            trail_pct = float(self.get_sym_cfg(symbol, 'trail_pct', trail_pct) or trail_pct)
            entry = st.get('entry', 0)
            if entry <= 0:
                return
            # 覆盖追踪激活阈值（仅特定币）
            trail_activate_pct_local = float(self.get_sym_cfg(symbol, 'TRAIL_ACTIVATE_PCT', self.trail_activate_pct) or self.trail_activate_pct)

            basis_price = current_price
            activated = False
            new_sl = st['sl']
            if side == 'long':
                peak = max(self.trailing_peak.get(symbol, entry), basis_price)
                self.trailing_peak[symbol] = peak
                profit_long = (basis_price - entry)
                act_need = max(self.trail_activate_atr * atr_val, trail_activate_pct_local * entry)
                activated = profit_long >= act_need
                atr_sl = basis_price - n * atr_val
                percent_sl = peak * (1 - trail_pct) if activated else st['sl']
                new_sl = max(st['sl'], atr_sl, percent_sl)
                if new_sl > st['sl'] and (new_sl - st['sl']) >= (self.trail_sl_min_delta_atr * atr_val):
                    st['sl'] = new_sl
            else:
                trough = min(self.trailing_trough.get(symbol, entry), basis_price)
                self.trailing_trough[symbol] = trough
                profit_short = (entry - basis_price)
                act_need = max(self.trail_activate_atr * atr_val, trail_activate_pct_local * entry)
                activated = profit_short >= act_need
                atr_sl = basis_price + n * atr_val
                percent_sl = trough * (1 + trail_pct) if activated else st['sl']
                new_sl = min(st['sl'], atr_sl, percent_sl)
                if new_sl < st['sl'] and (st['sl'] - new_sl) >= (self.trail_sl_min_delta_atr * atr_val):
                    st['sl'] = new_sl
            
            # 三阶段追踪
            profit = (basis_price - entry) if side == 'long' else (entry - basis_price)
            atr_mult = profit / atr_val if atr_val > 0 else 0.0
            if atr_mult >= self.trail_stage_1:
                if side == 'long':
                    st['sl'] = max(st['sl'], entry)
                else:
                    st['sl'] = min(st['sl'], entry)
            if atr_mult >= self.trail_stage_2:
                if side == 'long':
                    st['sl'] = max(st['sl'], entry + self.trail_stage2_offset * atr_val)
                else:
                    st['sl'] = min(st['sl'], entry - self.trail_stage2_offset * atr_val)
            if atr_mult >= self.trail_stage_3 and not self.stage3_done.get(symbol, False):
                pos = self.get_position(symbol, force_refresh=True)
                sz = pos.get('size', 0)
                if sz > 0 and 0 < self.partial_tp_ratio_stage3 < 1:
                    cut = min(sz, sz * self.partial_tp_ratio_stage3)
                    if cut > 0:
                        reduce_side = 'sell' if side == 'long' else 'buy'
                        if self.reduce_only_market(symbol, reduce_side, cut, side):
                            logger.info(f"✅ Stage3分批止盈 {symbol}: 减仓 {cut:.6f} ({self.partial_tp_ratio_stage3:.2f})")
                            self.stage3_done[symbol] = True
            
            self.sl_tp_state[symbol] = st
            logger.debug(f"🔄 更新追踪止损 {symbol}: 新SL={new_sl:.6f} 激活={activated}")
        except Exception as e:
            logger.warning(f"⚠️ 更新追踪止损异常 {symbol}: {str(e)} - {traceback.format_exc()}")

    def _check_hard_stop(self, symbol: str, current_price: float, side: str) -> bool:
        """硬止损：当亏损超过阈值(按入场价百分比)立即市价平仓。返回是否已执行平仓。"""
        try:
            st = self.sl_tp_state.get(symbol)
            if not st:
                return False
            entry = st.get('entry', 0)
            if entry <= 0 or current_price <= 0:
                return False
            max_loss_pct = self.hard_sl_max_loss_pct
            if max_loss_pct <= 0:
                return False
            loss_pct = (entry - current_price) / entry if side == 'long' else (current_price - entry) / entry
            if loss_pct >= max_loss_pct:
                logger.warning(f"🛑 硬止损触发 {symbol}: 亏损比例={loss_pct:.4%} ≥ 阈值={max_loss_pct:.2%}，立即平仓")
                self.close_position(symbol, open_reverse=False)
                return True
            return False
        except Exception as e:
            logger.warning(f"⚠️ 硬止损检查异常 {symbol}: {str(e)} - {traceback.format_exc()}")
            return False

    def _check_sar_flip_exit(self, symbol: str, side: str) -> bool:
        """SAR掉头平仓：按最近K线的SAR穿越与偏离阈值判断，满足则reduceOnly平仓。返回是否已平仓。"""
        try:
            if not self.use_sar_flip_exit:
                return False
            kl = self.get_klines(symbol, max(50, self.sar_confirm_bars + 2))
            if not kl or len(kl) < (self.sar_confirm_bars + 1):
                return False
            # 计算SAR（使用与BB相同的策略参数组或默认）
            ps_b = self.get_strategy_params(symbol) if hasattr(self, 'strategy_params') else {}
            sar_af_start = ps_b.get('sar_af_start', 0.02)
            sar_af_max = ps_b.get('sar_af_max', 0.2)
            sar_series = None
            sar_val_last = self.calculate_sar(kl, sar_af_start, sar_af_max)
            if sar_val_last is not None:
                sar_series = [sar_val_last] * (self.sar_confirm_bars + 1)
            if not sar_series or len(sar_series) < self.sar_confirm_bars + 1:
                return False
            # 使用收盘价进行确认
            closes = [k['close'] for k in kl]
            confirm = self.sar_confirm_bars
            min_cross = self.sar_min_cross_pct
            # 最近confirm根逐根检查穿越方向
            ok = True
            for i in range(1, confirm + 1):
                c = float(closes[-i]); s = float(sar_series[-i])
                if c <= 0 or s <= 0:
                    ok = False; break
                diff_pct = abs(c - s) / c
                if side == 'long':
                    # 多头：收盘位于SAR之下且偏离足够
                    if not (c < s and diff_pct >= min_cross):
                        ok = False; break
                else:
                    # 空头：收盘位于SAR之上且偏离足够
                    if not (c > s and diff_pct >= min_cross):
                        ok = False; break
            if not ok:
                return False
            # 满足条件，执行reduceOnly平仓
            pos = self.get_position(symbol, force_refresh=True)
            qty = pos.get('size', 0)
            if qty <= 0:
                return False
            reduce_side = 'sell' if side == 'long' else 'buy'
            if self.reduce_only_market(symbol, reduce_side, qty, side):
                logger.warning(f"🧭 SAR翻转平仓触发 {symbol}: side={side} confirm={confirm} min_cross={min_cross:.3%}")
                return True
            return False
        except Exception as e:
            logger.warning(f"⚠️ SAR翻转平仓检测异常 {symbol}: {str(e)} - {traceback.format_exc()}")
            return False

    def _maybe_partial_take_profit(self, symbol: str, current_price: float, atr_val: float, side: str):
        """分批止盈：基于 ATR 阶梯，达到阈值即按比例减仓"""
        try:
            # per-symbol 覆盖；空串表示关闭分批
            tiers_str = self.get_sym_cfg(symbol, 'PARTIAL_TP_TIERS', _get_env_str('PARTIAL_TP_TIERS'))  # e.g., "1.5:0.3,3.0:0.3"
            if tiers_str == '' or not tiers_str or atr_val <= 0:
                return
            st = self.sl_tp_state.get(symbol)
            pos = self.get_position(symbol, force_refresh=True)
            size = pos.get('size', 0)
            if size <= 0 or not st:
                return
            entry = st.get('entry', 0)
            if entry <= 0 or current_price <= 0:
                return
            profit = (current_price - entry) if side == 'long' else (entry - current_price)
            atr_mult = profit / atr_val if atr_val > 0 else 0.0
            done = self.partial_tp_done.setdefault(symbol, set())
            for seg in tiers_str.split(','):
                if ':' not in seg:
                    continue
                th_s, ratio_s = seg.split(':', 1)
                th = float(th_s); ratio = float(ratio_s)
                key = f"{th:.3f}"
                if atr_mult >= th and key not in done and 0 < ratio < 1:
                    qty = min(size * ratio, size)
                    if qty <= 0:
                        continue
                    side_reduce = 'sell' if side == 'long' else 'buy'
                    if self.reduce_only_market(symbol, side_reduce, qty, side):
                        done.add(key)
                        logger.info(f"✅ 分批止盈 {symbol}: 触发 {th}×ATR，减仓比例 {ratio:.2f}，数量 {qty:.6f}")
                        size -= qty
                        if size <= 0:
                            break
                    else:
                        logger.warning(f"⚠️ 分批止盈下单失败 {symbol}: 阶梯 {th}×ATR, 比例 {ratio:.2f}")
        except Exception as e:
            logger.warning(f"⚠️ 分批止盈异常 {symbol}: {str(e)} - {traceback.format_exc()}")

    def place_okx_tp_sl(self, symbol: str, entry_price: float, side: str, atr_val: float) -> bool:
        """在OKX侧同时挂TP/SL条件单。优先 oco，失败(51000)回退 tp_sl；严格以 sCode 判定成功。"""
        try:
            if self.okx_tp_sl_placed.get(symbol):
                return True
            inst_id = self.symbol_to_inst_id(symbol)
            if not inst_id or entry_price <= 0 or atr_val <= 0 or side not in ('long', 'short'):
                return False
            pos = self.get_position(symbol, force_refresh=True)
            size = pos.get('size', 0)
            if size <= 0:
                logger.warning(f"⚠️ 无有效持仓数量，跳过挂TP/SL {symbol}")
                return False

            # 定义本次目标posSide，避免未定义变量
            pos_side = 'long' if side == 'long' else 'short'

            # 若已存在同instId(+posSide)的未完成条件单，则直接跳过挂单，避免重复
            try:
                existing = []
                for _ord in ('oco', 'trigger', 'conditional'):
                    try:
                        _resp = self._safe_call(
                            self.exchange.privateGetTradeOrdersAlgoPending,
                            {'instType': 'SWAP', 'instId': inst_id, 'ordType': _ord}
                        )
                        existing.extend(_resp.get('data', []))
                    except Exception:
                        pass
                if existing:
                    match_found = False
                    for it in existing:
                        if it.get('instId') != inst_id:
                            continue
                        if self.is_hedge_mode:
                            if (it.get('posSide') or '').lower() != pos_side:
                                continue
                        match_found = True
                        break
                    if match_found:
                        logger.info(f"ℹ️ 已存在未完成TP/SL条件单，跳过重挂 {symbol}")
                        self.okx_tp_sl_placed[symbol] = True
                        self.tp_sl_last_placed[symbol] = time.time()
                        return True
            except Exception:
                # 查询失败不阻塞后续流程
                pass
            # 若不存在则清理旧单（在未使用algoClOrdId时也能清理干净）
            self.cancel_symbol_tp_sl(symbol)
            time.sleep(0.3)

            cfg = self.get_symbol_cfg(symbol)
            n = cfg.get('n', self.atr_sl_n); m = cfg.get('m', self.atr_tp_m)
            if side == 'long':
                sl_trigger = entry_price - n * atr_val
                tp_trigger = entry_price + m * atr_val
                ord_side = 'sell'
                pos_side = 'long'
            else:
                sl_trigger = entry_price + n * atr_val
                tp_trigger = entry_price - m * atr_val
                ord_side = 'buy'
                pos_side = 'short'
            
            tkr = self._safe_call(self.exchange.publicGetMarketTicker, {'instId': inst_id})
            d = tkr.get('data', [{}])[0]
            last_price = float(d.get('last') or d.get('lastPx') or 0.0)
            price_prec = self.markets_info.get(symbol, {}).get('price_precision', 4)
            tick = 10 ** (-price_prec)
            min_gap = max(0.001 * last_price, 5 * tick) if last_price > 0 else 5 * tick
            if last_price > 0:
                if side == 'long':
                    sl_trigger = min(sl_trigger, last_price - min_gap)
                    tp_trigger = max(tp_trigger, last_price + min_gap)
                    sl_trigger = math.floor(sl_trigger / tick) * tick
                    tp_trigger = math.ceil(tp_trigger / tick) * tick
                else:
                    sl_trigger = max(sl_trigger, last_price + min_gap)
                    tp_trigger = min(tp_trigger, last_price - min_gap)
                    sl_trigger = math.ceil(sl_trigger / tick) * tick
                    tp_trigger = math.floor(tp_trigger / tick) * tick



            # 计算布林带强趋势：强趋势仅挂SL触发单（trigger）
            kl_tmp = self.get_klines(symbol, 60)
            closes_tmp = [k['close'] for k in kl_tmp] if kl_tmp else []
            strong_trend = False
            if closes_tmp:
                ps_b = self.get_strategy_params(symbol)
                bb_period = ps_b.get('bb_period', 20)
                bb_k = ps_b.get('bb_k', 2.0)
                bbv = self.calculate_bollinger_bands(closes_tmp, bb_period, bb_k)
                if bbv and bbv.get('band_ma20', 0) and bbv.get('band_width', 0):
                    strong_trend = bbv['band_width'] > bbv['band_ma20'] * 1.0

            def _post_algo(ord_type: str):
                # 基础字段
                payload = {
                    'instId': inst_id,
                    'tdMode': 'cross',
                    'side': ord_side,
                    'ordType': ord_type,
                    'reduceOnly': True,
                    'sz': f"{size}",
                }
                # 字段按类型区分：oco 使用 tp/slTriggerPx；trigger 使用 triggerPx/orderPx
                if ord_type == 'oco':
                    payload['slTriggerPx'] = f"{sl_trigger}"
                    payload['slOrdPx'] = '-1'
                    if not strong_trend:
                        payload['tpTriggerPx'] = f"{tp_trigger}"
                        payload['tpOrdPx'] = '-1'
                else:  # trigger
                    payload['triggerPx'] = f"{sl_trigger}"
                    payload['orderPx'] = '-1'
                if self.is_hedge_mode:
                    payload['posSide'] = pos_side
                return self._safe_call(self.exchange.privatePostTradeOrderAlgo, payload)

            def _is_success(resp: Any) -> bool:
                if not isinstance(resp, dict) or str(resp.get('code', '')) not in ('0', '200'):
                    return False
                data = resp.get('data', [])
                return any(str(x.get('sCode', '')) == '0' for x in data)

            if strong_trend:
                # 强趋势：优先只挂SL触发单
                resp = _post_algo('trigger')
                if _is_success(resp):
                    logger.info(f"📌 交易所侧已挂SL触发单(强趋势) {symbol}: size={size:.6f} SL@{sl_trigger:.6f} (ordType=trigger)")
                    self.okx_tp_sl_placed[symbol] = True
                    self.tp_sl_last_placed[symbol] = time.time()
                    return True
                # 回退尝试OCO
                resp2 = _post_algo('oco')
                if _is_success(resp2):
                    logger.info(f"📌 交易所侧TP/SL已挂(回退OCO) {symbol}: size={size:.6f} TP@{tp_trigger:.6f} SL@{sl_trigger:.6f} (ordType=oco)")
                    self.okx_tp_sl_placed[symbol] = True
                    self.tp_sl_last_placed[symbol] = time.time()
                    return True
                logger.warning(f"⚠️ 交易所侧挂单失败(强趋势) {symbol}: trigger→oco 均失败: {resp2 or resp}")
                return False
            else:
                # 非强趋势：先试OCO，失败再回退trigger
                resp = _post_algo('oco')
                if _is_success(resp):
                    logger.info(f"📌 交易所侧TP/SL已挂 {symbol}: size={size:.6f} TP@{tp_trigger:.6f} SL@{sl_trigger:.6f} (ordType=oco)")
                    self.okx_tp_sl_placed[symbol] = True
                    self.tp_sl_last_placed[symbol] = time.time()
                    return True

                msg = str(resp)
                # 兼容51000/ordType异常时回退
                if '51000' in msg or 'ordType' in msg.lower():
                    resp2 = _post_algo('trigger')
                    if _is_success(resp2):
                        logger.info(f"📌 交易所侧已挂SL触发单(回退) {symbol}: size={size:.6f} SL@{sl_trigger:.6f} (ordType=trigger)")
                        self.okx_tp_sl_placed[symbol] = True
                        self.tp_sl_last_placed[symbol] = time.time()
                        return True
                    logger.warning(f"⚠️ 交易所侧挂单失败 {symbol} 回退trigger失败: {resp2}")
                    return False

                logger.warning(f"⚠️ 交易所侧TP/SL挂单失败 {symbol}: {resp}")
                return False
        except Exception as e:
            logger.warning(f"⚠️ 交易所侧TP/SL挂单异常 {symbol}: {str(e)} - {traceback.format_exc()}")
            return False

    def calculate_atr(self, klines: List[Dict], period: int = 14) -> float:
        """计算 ATR（Wilder）"""
        try:
            if len(klines) < period + 1:
                return 0.0
            highs = np.array([k['high'] for k in klines], dtype=float)
            lows = np.array([k['low'] for k in klines], dtype=float)
            closes = np.array([k['close'] for k in klines], dtype=float)
            prev_closes = np.concatenate(([closes[0]], closes[:-1]))
            tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_closes), np.abs(lows - prev_closes)))
            atr = np.zeros_like(tr)
            atr[period-1] = tr[:period].mean()
            for i in range(period, len(tr)):
                atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
            return float(atr[-1])
        except Exception as e:
            logger.debug(f"⚠️ ATR计算异常: {str(e)}")
            return 0.0

    def calculate_adx(self, klines: List[Dict], period: int = 14) -> float:
        """计算 ADX（Wilder）"""
        try:
            if len(klines) < period + 1:
                return 0.0
            highs = np.array([k['high'] for k in klines], dtype=float)
            lows = np.array([k['low'] for k in klines], dtype=float)
            closes = np.array([k['close'] for k in klines], dtype=float)

            up_move = highs[1:] - highs[:-1]
            down_move = lows[:-1] - lows[1:]
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

            prev_closes = closes[:-1]
            tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - prev_closes), np.abs(lows[1:] - prev_closes)))

            def wilder_smooth(arr):
                sm = np.zeros_like(arr)
                sm[period-1] = arr[:period].sum()
                for i in range(period, len(arr)):
                    sm[i] = sm[i-1] - (sm[i-1] / period) + arr[i]
                return sm

            plus_dm_sm = wilder_smooth(plus_dm)
            minus_dm_sm = wilder_smooth(minus_dm)
            tr_sm = wilder_smooth(tr)

            tr_sm_safe = np.where(tr_sm == 0, 1e-12, tr_sm)

            plus_di = 100.0 * (plus_dm_sm / tr_sm_safe)
            minus_di = 100.0 * (minus_dm_sm / tr_sm_safe)
            dx = 100.0 * (np.abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, 1e-12))

            adx = np.zeros_like(dx)
            adx[period-1] = dx[:period].mean()
            for i in range(period, len(dx)):
                adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period

            return float(adx[-1])
        except Exception as e:
            logger.debug(f"⚠️ ADX计算异常: {str(e)}")
            return 0.0

    def calculate_bollinger_bands(self, prices: List[float], period: int = 20, k: float = 2.0) -> Dict[str, Any]:
        """计算布林带：返回上轨/中轨/下轨及带宽和中轨斜率"""
        try:
            if len(prices) < period + 2:
                return {}
            s = pd.Series(np.array(prices, dtype=float))
            mid = s.rolling(window=period, min_periods=period).mean()
            std = s.rolling(window=period, min_periods=period).std()
            upper = mid + k * std
            lower = mid - k * std
            up_arr = np.asarray(upper)
            lo_arr = np.asarray(lower)
            mid_arr = np.asarray(mid)
            if np.isnan(up_arr[-1]) or np.isnan(lo_arr[-1]) or np.isnan(mid_arr[-1]):
                return {}
            width = up_arr[-1] - lo_arr[-1]
            prev_width = up_arr[-2] - lo_arr[-2]
            mid_slope = mid_arr[-1] - mid_arr[-2] if not np.isnan(mid_arr[-2]) else 0.0
            bw_arr = (up_arr - lo_arr) / np.where(mid_arr == 0, np.nan, mid_arr)
            band_width = bw_arr[-1] if not np.isnan(bw_arr[-1]) else 0.0
            last_n = min(20, len(bw_arr))
            band_ma20 = np.nanmean(bw_arr[-last_n:]) if last_n > 0 else 0.0
            return {
                'upper': up_arr[-1],
                'middle': mid_arr[-1],
                'lower': lo_arr[-1],
                'prev_width': prev_width,
                'width': width,
                'mid_slope': mid_slope,
                'band_width': band_width,
                'band_ma20': band_ma20
            }
        except Exception as e:
            logger.debug(f"⚠️ BB计算异常: {str(e)}")
            return {}

    def calculate_sar(self, klines: List[Dict], af_start: float = 0.02, af_max: float = 0.2) -> Optional[float]:
        """计算抛物线SAR，返回最后一个SAR值（简化实现）"""
        try:
            if len(klines) < 3:
                return None
            last_ts = int(klines[-1].get('timestamp').timestamp() * 1000)
            cache_key = ('sar_last', len(klines), last_ts, af_start, af_max)
            if cache_key in self._sar_cache:
                return self._sar_cache[cache_key]
            
            highs = [k['high'] for k in klines]
            lows = [k['low'] for k in klines]
            sar = lows[0]
            trend = 1  # 1=上升，-1=下降
            ep = highs[0]
            af = af_start
            for i in range(1, len(highs)):
                prev_sar = sar
                if trend == 1:
                    sar = prev_sar + af * (ep - prev_sar)
                    if lows[i] < sar:
                        trend = -1
                        sar = ep
                        ep = lows[i]
                        af = af_start
                    else:
                        if highs[i] > ep:
                            ep = highs[i]
                            af = min(af + af_start, af_max)
                else:
                    sar = prev_sar + af * (ep - prev_sar)
                    if highs[i] > sar:
                        trend = 1
                        sar = ep
                        ep = highs[i]
                        af = af_start
                    else:
                        if lows[i] < ep:
                            ep = lows[i]
                            af = min(af + af_start, af_max)
            self._sar_cache[cache_key] = sar
            return sar
        except Exception as e:
            logger.debug(f"⚠️ SAR计算异常: {str(e)}")
            return None

    def get_strategy_for(self, symbol: str) -> str:
        """获取币种策略类型（当启用XRP模板时，对所有币种返回XRP的策略类型）"""
        if getattr(self, 'apply_xrp_template_all', False):
            return self.strategy_by_symbol.get(getattr(self, 'xrp_symbol', 'XRP/USDT:USDT'), 'bb_sar')
        return self.strategy_by_symbol.get(symbol, 'macd_sar')

    def get_symbol_cfg(self, symbol: str) -> Dict:
        """获取币种配置（当启用XRP模板时，返回XRP的追踪/动态TP/SL配置）"""
        if getattr(self, 'apply_xrp_template_all', False):
            return self.symbol_cfg.get(getattr(self, 'xrp_symbol', 'XRP/USDT:USDT'), {})
        return self.symbol_cfg.get(symbol, {})

    def calculate_macd(self, prices: List[float]) -> Dict[str, float]:
        """计算MACD"""
        s = pd.Series(prices)
        ema_fast = s.ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = s.ewm(span=self.slow_period, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=self.signal_period, adjust=False).mean()
        hist = macd - signal
        return {'macd': macd.iloc[-1], 'signal': signal.iloc[-1], 'histogram': hist.iloc[-1]}

    def get_strategy_params(self, symbol: str) -> Dict[str, Any]:
        """获取策略参数（BB/SAR参数组）。启用XRP模板时，统一返回XRP的参数。"""
        if getattr(self, 'apply_xrp_template_all', False):
            return self.strategy_params.get(getattr(self, 'xrp_symbol', 'XRP/USDT:USDT'), {})
        return self.strategy_params.get(symbol, {})

    def calculate_macd_with_params(self, prices: List[float], fast: int, slow: int, signal: int) -> Dict[str, float]:
        """带参数MACD计算"""
        s = pd.Series(prices)
        ema_fast = s.ewm(span=fast, adjust=False).mean()
        ema_slow = s.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        sig = macd.ewm(span=signal, adjust=False).mean()
        hist = macd - sig
        return {'macd': macd.iloc[-1], 'signal': sig.iloc[-1], 'histogram': hist.iloc[-1]}

    def analyze_macd_sar(self, symbol: str, closes: List[float], klines: List[Dict], atr_val: float, adx_val: float, position: Dict, close_price: float) -> Dict[str, str]:
        """macd_sar策略分析"""
        ps = self.per_symbol_params.get(symbol, {})
        macd_params = ps.get('macd', (self.fast_period, self.slow_period, self.signal_period))
        macd_current = self.calculate_macd_with_params(closes, *macd_params)
        macd_prev = self.calculate_macd_with_params(closes[:-1], *macd_params)
        
        prev_macd = macd_prev['macd']
        prev_signal = macd_prev['signal']
        prev_hist = macd_prev['histogram']
        current_macd = macd_current['macd']
        current_signal = macd_current['signal']
        current_hist = macd_current['histogram']
        
        if position['size'] > 0:
            if position['side'] == 'long':
                if (prev_macd >= prev_signal and current_macd < current_signal) and (current_hist < 0):
                    return {'signal': 'close', 'reason': '多头双确认平仓：死叉且柱状图为负'}
                return {'signal': 'hold', 'reason': '持有多头'}
            else:
                if (prev_macd <= prev_signal and current_macd > current_signal) and (current_hist > 0):
                    return {'signal': 'close', 'reason': '空头双确认平仓：金叉且柱状图为正'}
                return {'signal': 'hold', 'reason': '持有空头'}

        buy_cross = (prev_macd <= prev_signal and current_macd > current_signal)
        buy_color = (prev_hist <= 0 and current_hist > 0)
        sell_cross = (prev_macd >= prev_signal and current_macd < current_signal)
        sell_color = (prev_hist >= 0 and current_hist < 0)

        hist_strength_pct = _get_env_float('HIST_STRENGTH_PCT', 0.0008)
        hist_abs_thresh = hist_strength_pct * close_price

        congested = False
        last_n = 30
        if len(klines) >= last_n:
            hi_max = max(k['high'] for k in klines[-last_n:])
            lo_min = min(k['low'] for k in klines[-last_n:])
            rng = hi_max - lo_min
            congested = rng < (1.8 * atr_val)

        ema_ok_long = True
        ema_ok_short = True
        inst_id = self.symbol_to_inst_id(symbol)
        resp15 = self._safe_call(self.exchange.publicGetMarketCandles, {'instId': inst_id, 'bar': '15m', 'limit': '80'})
        rows15 = resp15.get('data', [])
        closes15 = [float(r[4]) for r in rows15]
        if len(closes15) >= 50:
            ema20 = pd.Series(closes15).ewm(span=20, adjust=False).mean().values[-1]
            ema50 = pd.Series(closes15).ewm(span=50, adjust=False).mean().values[-1]
            ema_ok_long = ema20 > ema50
            ema_ok_short = ema20 < ema50

        if buy_cross and buy_color:
            if abs(prev_hist) < hist_abs_thresh:
                return {'signal': 'hold', 'reason': '柱状图强度不足'}
            if congested:
                return {'signal': 'hold', 'reason': '拥挤过滤'}
            if not ema_ok_long:
                return {'signal': 'hold', 'reason': '15m EMA不同向(多)'}
            return {'signal': 'buy', 'reason': 'MACD双确认+过滤通过'}
        elif sell_cross and sell_color:
            if abs(prev_hist) < hist_abs_thresh:
                return {'signal': 'hold', 'reason': '柱状图强度不足'}
            if congested:
                return {'signal': 'hold', 'reason': '拥挤过滤'}
            if not ema_ok_short:
                return {'signal': 'hold', 'reason': '15m EMA不同向(空)'}
            return {'signal': 'sell', 'reason': 'MACD双确认+过滤通过'}
        else:
            return {'signal': 'hold', 'reason': '等待MACD双确认'}

    def analyze_bb_sar(self, symbol: str, closes: List[float], klines: List[Dict], position: Dict, close_price: float) -> Dict[str, str]:
        """bb_sar策略分析"""
        if position['size'] > 0:
            return {'signal': 'hold', 'reason': '已有持仓，bb_sar不处理平仓'}
        
        ps_b = self.get_strategy_params(symbol)
        bb_period = ps_b.get('bb_period', 20)
        bb_k = ps_b.get('bb_k', 2.5)
        bb = self.calculate_bollinger_bands(closes, bb_period, bb_k) or {}
        if not bb:
            return {'signal': 'hold', 'reason': 'BB数据不足'}
        
        sar_val = self.calculate_sar(
            klines,
            ps_b.get('sar_af_start', 0.01),
            ps_b.get('sar_af_max', 0.10)
        )
        
        upper = bb['upper']; middle = bb['middle']; lower = bb['lower']
        bw = bb['band_width']; bw_ma20 = bb['band_ma20']
        if bw_ma20 > 0 and bw <= bw_ma20 * 0.8:
            return {'signal': 'hold', 'reason': '布林收口观望(band_width <= mean*0.8)'}

        price = close_price
        mid_slope = bb['mid_slope']
        cond_buy = (mid_slope > 0 and price > middle and (sar_val is None or price > sar_val))
        cond_sell = (mid_slope < 0 and price < middle and (sar_val is None or price < sar_val))

        if cond_buy:
            return {'signal': 'buy', 'reason': 'BB三线向上+价>中轨+SAR下方'}
        if cond_sell:
            return {'signal': 'sell', 'reason': 'BB三线向下+价<中轨+SAR上方'}
        return {'signal': 'hold', 'reason': 'BB条件未满足'}

    def analyze_hybrid(self, symbol: str, closes: List[float], klines: List[Dict], position: Dict, close_price: float) -> Dict[str, str]:
        """hybrid策略分析"""
        if position['size'] > 0:
            return {'signal': 'hold', 'reason': '已有持仓，hybrid不处理平仓'}
        
        ps_b = self.strategy_params.get(symbol, {})
        bb_period = ps_b.get('bb_period', 20)
        bb_k = ps_b.get('bb_k', 2.2)
        bb = self.calculate_bollinger_bands(closes, bb_period, bb_k) or {}
        if not bb:
            return {'signal': 'hold', 'reason': 'BB数据不足'}
        
        sar_val = self.calculate_sar(
            klines,
            ps_b.get('sar_af_start', 0.03),
            ps_b.get('sar_af_max', 0.25)
        )
        
        upper = bb['upper']; lower = bb['lower']
        bw = bb['band_width']; bw_ma20 = bb['band_ma20']
        if bw_ma20 > 0 and bw <= bw_ma20 * 0.8:
            return {'signal': 'hold', 'reason': '布林收口观望(band_width <= mean*0.8)'}

        price = close_price
        bull_break = (price > upper and (sar_val is None or price > sar_val))
        bear_break = (price < lower and (sar_val is None or price < sar_val))

        if bull_break:
            return {'signal': 'buy', 'reason': 'BB上轨突破 + SAR确认'}
        if bear_break:
            return {'signal': 'sell', 'reason': 'BB下轨跌破 + SAR确认'}
        return {'signal': 'hold', 'reason': '等待BB突破+SAR确认'}

    def analyze_symbol(self, symbol: str) -> Dict[str, str]:
        """分析单个交易对"""
        try:
            klines = self.get_klines(symbol, 100)
            if not klines:
                return {'signal': 'hold', 'reason': '数据获取失败'}
            
            closes = [k['close'] for k in klines]
            if len(closes) < 2:
                return {'signal': 'hold', 'reason': '数据不足'}

            ps = self.per_symbol_params.get(symbol, {})
            atr_period = ps.get('atr_period', 14)
            atr_ratio_thresh = _get_env_float('ATR_RATIO_THRESH', 0.008)
            adx_period = ps.get('adx_period', 14)
            adx_min_trend = _get_env_float('ADX_MIN_TREND', 25)

            close_price = closes[-1]
            atr_val = self.calculate_atr(klines, atr_period)
            adx_val = self.calculate_adx(klines, adx_period)

            if atr_val > 0 and close_price > 0:
                atr_ratio = atr_val / close_price
                if atr_ratio < atr_ratio_thresh:
                    logger.debug(f"ATR滤波提示：波动率低（ATR/收盘={atr_ratio:.4f} < {atr_ratio_thresh}），不拦截信号")

            if adx_val > 0 and adx_val < adx_min_trend:
                logger.debug(f"ADX滤波提示：趋势不足（ADX={adx_val:.1f} < {adx_min_trend}），不拦截信号")

            logger.debug(f"🔍 {symbol} ATR({atr_period})={atr_val:.6f}, ATR/Close={atr_val/close_price:.6f} | ADX({adx_period})={adx_val:.2f}")

            position = self.get_position(symbol, force_refresh=True)
            adx_th = ps.get('adx_min_trend', 0)
            if adx_th > 0 and adx_val < adx_th and position['size'] == 0:
                return {'signal': 'hold', 'reason': f'ADX不足 {adx_val:.1f} < {adx_th:.1f}'}

            strat = self.get_strategy_for(symbol)
            if strat == 'macd_sar':
                return self.analyze_macd_sar(symbol, closes, klines, atr_val, adx_val, position, close_price)
            elif strat == 'bb_sar':
                return self.analyze_bb_sar(symbol, closes, klines, position, close_price)
            else:  # hybrid
                return self.analyze_hybrid(symbol, closes, klines, position, close_price)
                        
        except Exception as e:
            logger.error(f"❌ 分析{symbol}失败: {str(e)} - {traceback.format_exc()}")
            return {'signal': 'hold', 'reason': f'分析异常: {str(e)}'}
    
    def execute_strategy(self):
        """执行策略"""
        logger.info("=" * 70)
        logger.info(f"🚀 开始执行MACD策略 (11个币种，{self.timeframe} 周期)")
        logger.info("=" * 70)
        
        try:
            self.check_sync_needed()
            
            balance = self.get_account_balance()
            logger.info(f"💰 当前账户余额: {balance:.2f} USDT")
            
            logger.info(self.stats.get_summary())
            
            self.display_current_positions()
            
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
                time.sleep(self.symbol_loop_delay)
            
            logger.info("-" * 70)
            logger.info("⚡ 执行交易操作...")
            logger.info("")
            
            for symbol, signal_info in signals.items():
                signal = signal_info['signal']
                reason = signal_info['reason']
                
                current_position = self.get_position(symbol, force_refresh=True)
                
                kl = self.get_klines(symbol, 50)
                if kl:
                    close_price = kl[-1]['close']
                    ps = self.per_symbol_params.get(symbol, {})
                    atr_p = ps.get('atr_period', 14)
                    atr_val = self.calculate_atr(kl, atr_p)
                    if current_position['size'] > 0 and atr_val > 0:
                        st0 = self.sl_tp_state.get(symbol)
                        if not st0:
                            entry0 = current_position.get('entry_price', 0)
                            if entry0 > 0:
                                self._set_initial_sl_tp(symbol, entry0, atr_val, current_position['side'])
                                okx_ok = self.place_okx_tp_sl(symbol, entry0, current_position['side'], atr_val)
                                if okx_ok:
                                    logger.info(f"📌 手动/历史持仓兜底：已初始化并挂TP/SL {symbol}")
                                else:
                                    logger.warning(f"⚠️ 手动/历史持仓兜底挂单失败 {symbol}")
                        side_now = current_position['side']
                        self._update_trailing_stop(symbol, close_price, atr_val, side_now)
                        if self._check_hard_stop(symbol, close_price, side_now):
                            current_position = self.get_position(symbol, force_refresh=True)
                            continue
                        self._maybe_partial_take_profit(symbol, close_price, atr_val, side_now)
                        st = self.sl_tp_state.get(symbol)
                        if st:
                            entry_px = st.get('entry', 0)
                            if entry_px > 0 and atr_val > 0:
                                profit = (close_price - entry_px) if side_now == 'long' else (entry_px - close_price)
                                if profit >= 2.5 * atr_val:
                                    st['sl'] = max(st['sl'], close_price - 1.0 * atr_val) if side_now == 'long' else min(st['sl'], close_price + 1.0 * atr_val)
                                elif profit >= 1.5 * atr_val:
                                    st['sl'] = max(st['sl'], close_price - 1.2 * atr_val) if side_now == 'long' else min(st['sl'], close_price + 1.2 * atr_val)
                            # —— 强趋势滚仓判定与执行 —— #
                            try:
                                # 强趋势：BB带宽 > 带宽均值，且中轨斜率向趋势方向
                                closes_tmp = [k['close'] for k in kl] if kl else []
                                strong_trend = False
                                if closes_tmp:
                                    ps_b = self.strategy_params.get(symbol, {})
                                    bb_period = ps_b.get('bb_period', 20)
                                    bb_k = ps_b.get('bb_k', 2.0)
                                    bbv = self.calculate_bollinger_bands(closes_tmp, bb_period, bb_k)
                                    if bbv and bbv.get('band_ma20', 0) and bbv.get('band_width', 0):
                                        if side_now == 'long':
                                            strong_trend = (bbv['band_width'] > bbv['band_ma20'] * 1.0) and (bbv.get('mid_slope', 0.0) > 0)
                                        else:
                                            strong_trend = (bbv['band_width'] > bbv['band_ma20'] * 1.0) and (bbv.get('mid_slope', 0.0) < 0)
                                if strong_trend and atr_val > 0 and not self.has_open_orders(symbol):
                                    cnt = self.pyramid_count.get(symbol, 0)
                                    if cnt < self.pyramid_max_adds:
                                        last_px = self.pyramid_last_add_px.get(symbol, st.get('entry', 0.0))
                                        last_ts_add = self.pyramid_last_add_ts.get(symbol, 0.0)
                                        step_ok = abs(close_price - last_px) >= max(self.pyramid_step_atr * atr_val, self.pyramid_min_gap_pct * last_px)
                                        cool_ok = (time.time() - last_ts_add) >= self.pyramid_cooldown_s
                                        # 仅在中轨之上（多头）或之下（空头）加仓，尽量跟随趋势
                                        middle_px = bbv.get('middle', 0.0) if closes_tmp else 0.0
                                        bias_ok = (close_price >= middle_px) if side_now == 'long' else (close_price <= middle_px)
                                        if step_ok and cool_ok and bias_ok:
                                            # 计算本次加仓名义金额 = 基准金额 * 因子
                                            base_amt = self.calculate_order_amount(symbol)
                                            factors = self.pyramid_size_factors if isinstance(self.pyramid_size_factors, list) else [0.5, 0.35, 0.25]
                                            factor = factors[cnt] if cnt < len(factors) else factors[-1]
                                            add_amt = max(1.0, base_amt * float(factor))
                                            side_open = 'buy' if side_now == 'long' else 'sell'
                                            if self.create_order(symbol, side_open, add_amt):
                                                # 更新滚仓状态
                                                self.pyramid_count[symbol] = cnt + 1
                                                self.pyramid_last_add_px[symbol] = close_price
                                                self.pyramid_last_add_ts[symbol] = time.time()
                                                # 同时更严SL：靠近中轨留保护
                                                if closes_tmp and middle_px > 0:
                                                    if side_now == 'long':
                                                        st['sl'] = max(st['sl'], middle_px - 0.5 * atr_val, close_price - self.get_symbol_cfg(symbol).get('n', self.atr_sl_n) * atr_val)
                                                    else:
                                                        st['sl'] = min(st['sl'], middle_px + 0.5 * atr_val, close_price + self.get_symbol_cfg(symbol).get('n', self.atr_sl_n) * atr_val)
                                                logger.info(f"➕ 滚仓加仓 {symbol}: 第{cnt+1}次, 金额={add_amt:.2f}U, 价={close_price:.6f}")
                                            else:
                                                logger.warning(f"⚠️ 滚仓加仓失败 {symbol}: 条件满足但下单失败")
                            except Exception as _e:
                                logger.debug(f"ℹ️ 滚仓判定异常 {symbol}: {_e}")
                            last_ts = self.tp_sl_last_placed.get(symbol, 0.0)
                            if (time.time() - last_ts) >= self.tp_sl_refresh_interval:
                                self.cancel_symbol_tp_sl(symbol)
                                entry_px2 = st.get('entry', 0)
                                okx_ok = self.place_okx_tp_sl(symbol, entry_px2, side_now, atr_val) if entry_px2 > 0 else False
                                if okx_ok:
                                    logger.info(f"🔄 更新追踪止盈：冷却达到，已重挂 {symbol}")
                                else:
                                    logger.warning(f"⚠️ 更新追踪止盈重挂失败 {symbol}")
                            else:
                                logger.debug(f"⏳ 距上次挂单未达冷却({self.tp_sl_refresh_interval}s)，跳过重挂 {symbol}")
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
                    ps = self.per_symbol_params.get(symbol, {})
                    allow_reverse = ps.get('allow_reverse', True)
                    if self.close_position(symbol, open_reverse=allow_reverse):
                        if allow_reverse:
                            logger.info(f"✅ 平仓并反手开仓 {symbol} 成功 - {reason}")
                        else:
                            logger.info(f"✅ 平仓完成（不反手） {symbol} - {reason}")
            
            logger.info("=" * 70)
                        
        except Exception as e:
            logger.error(f"❌ 执行策略失败: {str(e)} - {traceback.format_exc()}")

    def run_continuous(self, interval: int = 60):
        """连续运行策略"""
        logger.info("=" * 70)
        logger.info("🚀 MACD策略启动 - RAILWAY平台版 (11个币种)")
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
        logger.info(f"💡 11个币种特性: 支持0.1U起的小额交易；优先为有信号的币种分配，串行下单，智能不均分")
        logger.info(self.stats.get_summary())
        logger.info("=" * 70)

        china_tz = pytz.timezone('Asia/Shanghai')

        while True:
            try:
                start_ts = time.time()

                self.check_sync_needed()

                self.execute_strategy()

                elapsed = time.time() - start_ts
                sleep_sec = max(1, interval - elapsed)
                logger.info(f"⏳ 休眠 {sleep_sec} 秒后继续实时巡检...")
                time.sleep(sleep_sec)

            except KeyboardInterrupt:
                logger.info("⛔ 用户中断，策略停止")
                break
            except Exception as e:
                logger.error(f"❌ 策略运行异常: {str(e)} - {traceback.format_exc()}")
                logger.info("🔄 60秒后重试...")
                time.sleep(60)

def main():
    """主函数"""
    logger.info("=" * 70)
    logger.info("🎯 MACD策略程序启动中... (11个币种版本)")
    logger.info("=" * 70)
    
    okx_api_key = _get_env_str('OKX_API_KEY')
    okx_secret_key = _get_env_str('OKX_SECRET_KEY')
    okx_passphrase = _get_env_str('OKX_PASSPHRASE')
    
    missing_vars = [var for var, val in [('OKX_API_KEY', okx_api_key), ('OKX_SECRET_KEY', okx_secret_key), ('OKX_PASSPHRASE', okx_passphrase)] if not val]
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
        logger.info(f"🔧 变量: SCAN_INTERVAL={_get_env_str('SCAN_INTERVAL','2')} OKX_API_MIN_INTERVAL={_get_env_str('OKX_API_MIN_INTERVAL','0.2')} SYMBOL_LOOP_DELAY={_get_env_str('SYMBOL_LOOP_DELAY','0.3')} SET_LEVERAGE_ON_START={_get_env_str('SET_LEVERAGE_ON_START','true')}")
        logger.info(f"🔧 变量: MAX_RETRIES={_get_env_str('MAX_RETRIES','3')} BACKOFF_BASE={_get_env_str('BACKOFF_BASE','0.8')} BACKOFF_MAX={_get_env_str('BACKOFF_MAX','3.0')} TP_SL_REFRESH_INTERVAL={_get_env_str('TP_SL_REFRESH_INTERVAL','300')}")

        scan_interval = _get_env_int('SCAN_INTERVAL', 2)
        if scan_interval <= 0:
            scan_interval = 1
        logger.info(f"🛠 扫描间隔设置: {scan_interval} 秒（可用环境变量 SCAN_INTERVAL 覆盖）")
        strategy.run_continuous(interval=scan_interval)
        
    except Exception as e:
        logger.error(f"❌ 策略初始化或运行失败: {str(e)} - {traceback.format_exc()}")

if __name__ == "__main__":
    main()
