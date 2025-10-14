#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MACD策略实现 - RAILWAY平台版本
扩展到11个币种，包含BTC/ETH/SOL/DOGE/XRP/PEPE/ARB
25倍杠杆，无限制交易，带挂单识别和状态同步
增加胜率统计和盈亏显示
"""
import time
import logging
import datetime
import os
import json
from typing import Dict, Any, List, Optional, Literal, cast
import pytz

import ccxt
import pandas as pd
import numpy as np
import math

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
handler.setLevel(logging.INFO)
formatter = ChinaTimeFormatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False  # 防止重复日志

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
            logger.warning(f"⚠️ 加载统计数据失败: {e}，使用新数据")
    
    def save_stats(self):
        """保存统计数据"""
        try:
            with open(self.stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            logger.error(f"❌ 保存统计数据失败: {e}")
    
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
    def __init__(self, api_key: str, secret_key: str, passphrase: str):
        """初始化策略"""
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
        self.per_symbol_params: Dict[str, Dict[str, Any]] = {
            # 原有小币种
            'FIL/USDT:USDT': {
                'macd': (8, 32, 12), 'atr_period': 14, 'adx_period': 12,
                'adx_min_trend': 23, 'sl_n': 2.0, 'tp_m': 3.5, 'allow_reverse': True
            },
            'ZRO/USDT:USDT': {
                'macd': (12, 36, 18), 'atr_period': 14, 'adx_period': 10,
                'adx_min_trend': 30, 'sl_n': 2.2, 'tp_m': 3.0, 'allow_reverse': True
            },
            'WIF/USDT:USDT': {
                'macd': (8, 26, 12), 'atr_period': 14, 'adx_period': 10,
                'adx_min_trend': 30, 'sl_n': 1.9, 'tp_m': 4.0, 'tp_pct': 0.012, 'allow_reverse': True
            },
            'WLD/USDT:USDT': {
                'macd': (9, 34, 13), 'atr_period': 14, 'adx_period': 12,
                'adx_min_trend': 23, 'sl_n': 2.0, 'tp_m': 3.5, 'allow_reverse': True
            },
            
            # 新增主流币
            'BTC/USDT:USDT': {
                'macd': (10, 40, 15), 'atr_period': 20, 'adx_period': 14,
                'adx_min_trend': 25, 'sl_n': 1.5, 'tp_m': 3.0, 'allow_reverse': True
            },
            'ETH/USDT:USDT': {
                'macd': (9, 35, 14), 'atr_period': 18, 'adx_period': 14,
                'adx_min_trend': 25, 'sl_n': 1.8, 'tp_m': 3.5, 'allow_reverse': True
            },
            'SOL/USDT:USDT': {
                'macd': (9, 35, 14), 'atr_period': 16, 'adx_period': 12,
                'adx_min_trend': 30, 'sl_n': 1.8, 'tp_m': 4.0, 'tp_pct': 0.012, 'allow_reverse': True
            },
            'XRP/USDT:USDT': {
                'macd': (10, 36, 14), 'atr_period': 16, 'adx_period': 14,
                'adx_min_trend': 24, 'sl_n': 1.8, 'tp_m': 3.5, 'allow_reverse': True
            },
            
            # 新增Meme币
            'DOGE/USDT:USDT': {
                'macd': (6, 16, 9), 'atr_period': 16, 'adx_period': 12,
                'adx_min_trend': 22, 'sl_n': 2.5, 'tp_m': 5.0, 'allow_reverse': True
            },
            'PEPE/USDT:USDT': {
                'macd': (5, 13, 8), 'atr_period': 14, 'adx_period': 10,
                'adx_min_trend': 20, 'sl_n': 3.0, 'tp_m': 6.0, 'allow_reverse': True
            },
            
            # 新增L2币
            'ARB/USDT:USDT': {
                'macd': (7, 22, 10), 'atr_period': 15, 'adx_period': 12,
                'adx_min_trend': 23, 'sl_n': 2.2, 'tp_m': 3.8, 'allow_reverse': True
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
        try:
            self._min_api_interval: float = float((os.environ.get('OKX_API_MIN_INTERVAL') or '0.2').strip())
        except Exception:
            self._min_api_interval = 0.2

        # 每币种微延时，降低瞬时调用密度
        try:
            self.symbol_loop_delay = float((os.environ.get('SYMBOL_LOOP_DELAY') or '0.3').strip())
        except Exception:
            self.symbol_loop_delay = 0.3
        # 启动时是否逐币设置杠杆（可设为 false 减少启动阶段私有接口调用）
        try:
            self.set_leverage_on_start = (os.environ.get('SET_LEVERAGE_ON_START', 'true').strip().lower() in ('1', 'true', 'yes'))
        except Exception:
            self.set_leverage_on_start = True
        
        # 交易统计
        self.stats = TradingStats()
        
        # 启动基线余额与风控参数
        try:
            self.starting_balance = float(self.get_account_balance() or 0.0)
        except Exception:
            self.starting_balance = 0.0
        try:
            self.hard_sl_max_loss_pct = float((os.environ.get('HARD_SL_MAX_LOSS_PCT') or '0.03').strip())  # 3%
        except Exception:
            self.hard_sl_max_loss_pct = 0.03
        try:
            self.account_dd_limit_pct = float((os.environ.get('ACCOUNT_DD_LIMIT_PCT') or '0.20').strip())  # 20%
        except Exception:
            self.account_dd_limit_pct = 0.20
        try:
            self.cb_close_all = (os.environ.get('CB_CLOSE_ALL', 'true').strip().lower() in ('1', 'true', 'yes'))
        except Exception:
            self.cb_close_all = True
        # 强制彻底关闭账户熔断
        self.cb_enabled = False
        self.circuit_breaker_triggered = False
        self.partial_tp_done: Dict[str, set] = {}
        # 撤单/标记 安全控制
        try:
            self.allow_cancel_pending = (os.environ.get('ALLOW_CANCEL_PENDING', 'true').strip().lower() in ('1','true','yes'))
        except Exception:
            self.allow_cancel_pending = True
        try:
            self.safe_cancel_only_our_tpsl = (os.environ.get('SAFE_CANCEL_ONLY_OUR_TPSL', 'true').strip().lower() in ('1','true','yes'))
        except Exception:
            self.safe_cancel_only_our_tpsl = True
        self.tpsl_cl_prefix = os.environ.get('TPSL_CL_PREFIX', 'MACD_TPSL_').strip() or 'MACD_TPSL_'
        
        # ATR 止盈止损参数
        try:
            self.atr_sl_n = float((os.environ.get('ATR_SL_N') or '1.8').strip())
        except Exception:
            self.atr_sl_n = 1.8
        try:
            self.atr_tp_m = float((os.environ.get('ATR_TP_M') or '2.2').strip())
        except Exception:
            self.atr_tp_m = 2.2
        
        # SL/TP 状态缓存
        self.sl_tp_state: Dict[str, Dict[str, float]] = {}
        self.okx_tp_sl_placed: Dict[str, bool] = {}
        # TP/SL重挂冷却与阈值
        self.tp_sl_last_placed: Dict[str, float] = {}
        try:
            self.tp_sl_refresh_interval = int((os.environ.get('TP_SL_REFRESH_INTERVAL') or '300').strip())
        except Exception:
            self.tp_sl_refresh_interval = 300
        try:
            self.tp_sl_min_delta_ticks = int((os.environ.get('TP_SL_MIN_DELTA_TICKS') or '2').strip())
        except Exception:
            self.tp_sl_min_delta_ticks = 2
        
        # ===== 每币种配置(用于追踪止损) =====
        self.symbol_cfg: Dict[str, Dict[str, float | str]] = {
            # 原有币种
            "ZRO/USDT:USDT": {"period": 14, "n": 2.2, "m": 3.0, "trigger_pct": 0.010, "trail_pct": 0.006, "update_basis": "high"},
            "WIF/USDT:USDT": {"period": 14, "n": 2.5, "m": 4.0, "trigger_pct": 0.015, "trail_pct": 0.010, "update_basis": "high"},
            "WLD/USDT:USDT": {"period": 14, "n": 2.0, "m": 3.5, "trigger_pct": 0.010, "trail_pct": 0.006, "update_basis": "close"},
            "FIL/USDT:USDT": {"period": 14, "n": 2.0, "m": 3.5, "trigger_pct": 0.010, "trail_pct": 0.006, "update_basis": "high"},
            
            # 新增主流币
            "BTC/USDT:USDT": {"period": 20, "n": 1.5, "m": 3.0, "trigger_pct": 0.008, "trail_pct": 0.004, "update_basis": "high"},
            "ETH/USDT:USDT": {"period": 18, "n": 1.8, "m": 3.5, "trigger_pct": 0.008, "trail_pct": 0.005, "update_basis": "high"},
            "SOL/USDT:USDT": {"period": 16, "n": 2.0, "m": 4.0, "trigger_pct": 0.012, "trail_pct": 0.007, "update_basis": "high"},
            "XRP/USDT:USDT": {"period": 16, "n": 1.8, "m": 3.5, "trigger_pct": 0.010, "trail_pct": 0.006, "update_basis": "close"},
            
            # 新增Meme币
            "DOGE/USDT:USDT": {"period": 16, "n": 2.5, "m": 5.0, "trigger_pct": 0.015, "trail_pct": 0.010, "update_basis": "high"},
            "PEPE/USDT:USDT": {"period": 14, "n": 3.0, "m": 6.0, "trigger_pct": 0.020, "trail_pct": 0.012, "update_basis": "high"},
            
            # 新增L2币
            "ARB/USDT:USDT": {"period": 15, "n": 2.2, "m": 3.8, "trigger_pct": 0.012, "trail_pct": 0.008, "update_basis": "high"}
        }
        
        # 跟踪峰值/谷值
        self.trailing_peak: Dict[str, float] = {}
        self.trailing_trough: Dict[str, float] = {}
        # 交易执行冷却与阶段追踪状态
        self.last_trade_candle_index: Dict[str, int] = {}
        self.stage3_done: Dict[str, bool] = {}
        # 信号增强配置（可用环境变量覆盖）
        self.ma_type = (os.environ.get('MA_TYPE', 'sma').strip().lower() or 'sma')  # sma|ema
        try:
            self.ma_fast = int((os.environ.get('MA_FAST') or '5').strip())
        except Exception:
            self.ma_fast = 5
        try:
            self.ma_slow = int((os.environ.get('MA_SLOW') or '20').strip())
        except Exception:
            self.ma_slow = 20
        try:
            self.vol_ma_period = int((os.environ.get('VOL_MA_PERIOD') or '20').strip())
        except Exception:
            self.vol_ma_period = 20
        try:
            self.vol_boost = float((os.environ.get('VOL_BOOST') or '1.2').strip())
        except Exception:
            self.vol_boost = 1.2
        try:
            self.long_body_pct = float((os.environ.get('LONG_BODY_PCT') or '0.6').strip())
        except Exception:
            self.long_body_pct = 0.6
        try:
            self.cooldown_candles = int((os.environ.get('COOLDOWN_CANDLES') or '3').strip())
        except Exception:
            self.cooldown_candles = 3
        # 三阶段追踪与最小阈值
        try:
            self.trail_stage_1 = float((os.environ.get('TRAIL_STAGE_1') or '1.0').strip())
        except Exception:
            self.trail_stage_1 = 1.0
        try:
            self.trail_stage_2 = float((os.environ.get('TRAIL_STAGE_2') or '1.75').strip())
        except Exception:
            self.trail_stage_2 = 1.75
        try:
            self.trail_stage_3 = float((os.environ.get('TRAIL_STAGE_3') or '2.5').strip())
        except Exception:
            self.trail_stage_3 = 2.5
        try:
            self.trail_stage2_offset = float((os.environ.get('TRAIL_STAGE2_OFFSET') or '0.8').strip())
        except Exception:
            self.trail_stage2_offset = 0.8
        try:
            self.trail_sl_min_delta_atr = float((os.environ.get('TRAIL_SL_MIN_DELTA_ATR') or '0.2').strip())
        except Exception:
            self.trail_sl_min_delta_atr = 0.2
        try:
            self.partial_tp_ratio_stage3 = float((os.environ.get('PARTIAL_TP_RATIO_STAGE3') or '0.3').strip())
        except Exception:
            self.partial_tp_ratio_stage3 = 0.3
        try:
            self.allow_strong_pa_override = (os.environ.get('ALLOW_STRONG_PA_OVERRIDE', 'true').strip().lower() in ('1','true','yes'))
        except Exception:
            self.allow_strong_pa_override = True
        
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
                is_rate = ('50011' in msg) or ('Too Many Requests' in msg)
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
            try:
                opts = self.exchange.options or {}
                opts.update({'defaultType': 'swap', 'defaultSettle': 'USDT', 'version': 'v5'})
                self.exchange.options = opts
            except Exception:
                pass
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
                payload_obj = {'algoIds': [{'algoId': x} for x in ids], 'ordType': ord_type, 'instId': inst_id}
                payload_arr = {'algoIds': ids, 'ordType': ord_type, 'instId': inst_id}
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
                                self.exchange.privatePostTradeCancelAlgos({'algoId': aid, 'ordType': ord_type, 'instId': inst_id})
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
                        ps = getattr(self, 'per_symbol_params', {}).get(symbol, {})
                        if isinstance(ps, dict) and ps.get('atr_period'):
                            atr_p = int(ps.get('atr_period'))
                        else:
                            atr_p = int((os.environ.get('ATR_PERIOD') or '14').strip())
                        atr_val = self.calculate_atr(kl, atr_p) if kl else 0.0
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
    
    def get_klines(self, symbol: str, limit: int = 100) -> List[Dict]:
        """获取K线数据"""
        try:
            inst_id = self.symbol_to_inst_id(symbol)
            tf = self.timeframe_map.get(symbol, None) or (self.timeframe if self.timeframe else '5m')
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
            return result
        except Exception as e:
            logger.error(f"❌ 获取{symbol}K线数据失败: {e}")
            return []
    
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
                logger.info(f"⚠️ {symbol} 存在{len(orders)}个未成交订单")
            return has_orders
        except Exception as e:
            logger.error(f"❌ 检查挂单失败: {e}")
            return False
    
    def calculate_order_amount(self, symbol: str, active_count: Optional[int] = None) -> float:
        """计算下单金额 - 方案A: 平均分配"""
        try:
            # 1) 固定目标名义金额（最高优先）
            target_str = os.environ.get('TARGET_NOTIONAL_USDT', '').strip()
            if target_str:
                try:
                    target = max(0.0, float(target_str))
                    logger.info(f"💵 使用固定目标名义金额: {target:.4f}U")
                    return target
                except Exception:
                    logger.warning(f"⚠️ TARGET_NOTIONAL_USDT 无效: {target_str}")

            # 2) 基于余额分配 - 方案A: 平均分配到11个币种
            balance = self.get_account_balance()
            if balance <= 0:
                logger.warning(f"⚠️ 余额不足，无法为 {symbol} 分配资金 (余额:{balance:.4f}U)")
                return 0.0

            # 平均分配：总余额 / 11个币种
            num_symbols = len(self.symbols)  # 11个币种
            allocated_amount = balance / max(1, num_symbols)

            # 3) 放大因子
            factor_str = os.environ.get('ORDER_NOTIONAL_FACTOR', '50').strip()
            try:
                factor = max(1.0, float(factor_str or '1'))
            except Exception:
                factor = 1.0
            allocated_amount *= factor

            # 4) 下限/上限
            def _to_float(env_name: str, default: float) -> float:
                try:
                    s = os.environ.get(env_name, '').strip()
                    return float(s) if s else default
                except Exception:
                    return default

            min_floor = max(0.0, _to_float('MIN_PER_SYMBOL_USDT', 0.0))
            max_cap = max(0.0, _to_float('MAX_PER_SYMBOL_USDT', 0.0))

            if min_floor > 0 and allocated_amount < min_floor:
                allocated_amount = min_floor
            if max_cap > 0 and allocated_amount > max_cap:
                allocated_amount = max_cap

            logger.info(f"💵 资金分配: 模式=平均分配, 总余额={balance:.4f}U, 币种数={num_symbols}, 因子={factor:.2f}, 本币目标={allocated_amount:.4f}U")
            if allocated_amount <= 0:
                logger.warning(f"⚠️ {symbol}最终分配金额为0，跳过")
                return 0.0

            return allocated_amount

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
                if step and step > 0:
                    try:
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
                        contract_size = min_amount
                        if step and step > 0:
                            contract_size = math.ceil(contract_size / step) * step
                        contract_size = round(contract_size, amount_precision)
            except Exception:
                pass

            if contract_size <= 0:
                logger.warning(f"⚠️ {symbol}最终数量无效: {contract_size}")
                return False

            logger.info(f"📝 准备下单: {symbol} {side} 金额:{amount:.4f}U 价格:{current_price:.4f} 数量:{contract_size:.8f}")
            try:
                est_cost = contract_size * current_price
                logger.info(f"🧮 下单成本对齐: 分配金额={amount:.4f}U | 预计成本={est_cost:.4f}U | 数量={contract_size:.8f} | minSz={min_amount} | lotSz={lot_sz}")
            except Exception:
                pass

            pos_side = 'long' if side == 'buy' else 'short'
            order_id = None
            last_err = None

            import traceback

            native_only = False
            try:
                native_only = (os.environ.get('USE_OKX_NATIVE_ONLY', '').strip().lower() in ('1', 'true', 'yes'))
            except Exception:
                native_only = False

            if not native_only:
                try:
                    params = {'tdMode': 'cross', 'posSide': pos_side}
                    resp = self.exchange.create_order(symbol, 'market', side, contract_size, None, params)
                    if isinstance(resp, dict):
                        order_id = resp.get('id') or resp.get('orderId') or resp.get('ordId') or resp.get('clOrdId')
                    elif isinstance(resp, list) and resp and isinstance(resp[0], dict):
                        order_id = resp[0].get('id') or resp[0].get('orderId') or resp[0].get('ordId') or resp[0].get('clOrdId')
                    if order_id:
                        logger.info(f"✅ 成功创建{symbol} {side}订单，数量:{contract_size:.8f}，订单ID:{order_id}")
                    else:
                        logger.warning(f"⚠️ create_order 返回未包含订单ID，响应: {resp}")
                except Exception as e1:
                    last_err = e1
                    logger.error(f"❌ create_order 异常: {e1}")
                    logger.debug(traceback.format_exc())

            if not order_id and not native_only:
                try:
                    params = {'tdMode': 'cross', 'posSide': pos_side}
                    resp = self.exchange.create_market_order(symbol, side, contract_size, None, params)  # type: ignore[arg-type]
                    if isinstance(resp, dict):
                        order_id = resp.get('id') or resp.get('orderId') or resp.get('ordId') or resp.get('clOrdId')
                    elif isinstance(resp, list) and resp and isinstance(resp[0], dict):
                        order_id = resp[0].get('id') or resp[0].get('orderId') or resp[0].get('ordId') or resp[0].get('clOrdId')
                    if order_id:
                        logger.info(f"✅ 成功创建{symbol} {side}订单（market API），数量:{contract_size:.8f}，订单ID:{order_id}")
                    else:
                        logger.warning(f"⚠️ create_market_order 返回未包含订单ID，响应: {resp}")
                except Exception as e2:
                    last_err = e2
                    logger.error(f"❌ create_market_order 异常: {e2}")
                    logger.debug(traceback.format_exc())

            if not order_id:
                try:
                    inst_id = self.symbol_to_inst_id(symbol)
                    raw_params = {
                        'instId': inst_id,
                        'tdMode': 'cross',
                        'side': side,
                        'posSide': pos_side,
                        'ordType': 'market',
                        'sz': str(contract_size)
                    }
                    resp = self.exchange.privatePostTradeOrder(raw_params)
                    if isinstance(resp, dict):
                        data = resp.get('data') or []
                        if isinstance(data, list) and data:
                            order_id = data[0].get('ordId') or data[0].get('clOrdId') or data[0].get('id')
                        else:
                            order_id = resp.get('ordId') or resp.get('clOrdId') or resp.get('id')
                    if order_id:
                        logger.info(f"✅ 成功创建{symbol} {side}订单（OKX原生兜底），数量:{contract_size:.8f}，订单ID:{order_id}")
                    else:
                        logger.error(f"❌ OKX原生下单无订单ID，响应: {resp}")
                except Exception as e3:
                    last_err = e3
                    logger.error(f"❌ OKX原生下单异常: {e3}")
                    logger.debug(traceback.format_exc())

            if order_id:
                time.sleep(2)
                pos = self.get_position(symbol, force_refresh=True)
                try:
                    kl = self.get_klines(symbol, 50)
                    atr_p = int((os.environ.get('ATR_PERIOD') or '14').strip())
                    atr_val = self.calculate_atr(kl, atr_p) if kl else 0.0
                    if pos and pos.get('size', 0) > 0 and atr_val > 0:
                        self._set_initial_sl_tp(symbol, float(pos.get('entry_price', 0) or 0), atr_val, pos.get('side', 'long'))
                        st = self.sl_tp_state.get(symbol)
                        if st:
                            logger.info(f"🎯 初始化SL/TP {symbol}: SL={st['sl']:.6f}, TP={st['tp']:.6f} (N={self.get_symbol_cfg(symbol).get('n')}, M={self.get_symbol_cfg(symbol).get('m')}, ATR={atr_val:.6f})")
                            okx_ok = self.place_okx_tp_sl(symbol, float(pos.get('entry_price', 0) or 0), pos.get('side', 'long'), atr_val)
                            if okx_ok:
                                logger.info(f"📌 已在交易所侧挂TP/SL {symbol}")
                            else:
                                logger.warning(f"⚠️ 交易所侧TP/SL挂单失败 {symbol}")
                except Exception:
                    pass
                return True

            if last_err:
                logger.error(f"❌ 创建{symbol} {side}订单失败：{last_err}")
            return False

        except Exception as e:
            logger.error(f"❌ 创建{symbol} {side}订单异常: {e}")
            import traceback as _tb
            logger.debug(_tb.format_exc())
            return False
    
    def reduce_only_market(self, symbol: str, side: str, size: float, pos_side: str) -> bool:
        """以 reduceOnly 市价减仓，避免反向开仓"""
        try:
            if size <= 0:
                return True
            inst_id = self.symbol_to_inst_id(symbol)
            raw_params = {
                'instId': inst_id,
                'tdMode': 'cross',
                'side': side,             # 对应平仓方向：long->sell, short->buy
                'posSide': pos_side,      # 'long' 或 'short'
                'reduceOnly': True,
                'ordType': 'market',
                'sz': f"{size}"
            }
            resp = self.exchange.privatePostTradeOrder(raw_params)
            # 简单成功判断
            if isinstance(resp, dict):
                code = str(resp.get('code', ''))
                if code in ('0', '200'):
                    return True
                data = resp.get('data') or []
                if isinstance(data, list) and data:
                    return str(data[0].get('sCode', '')) == '0'
            return False
        except Exception as e:
            logger.warning(f"⚠️ reduceOnly 市价减仓异常 {symbol}: {e}")
            return False

    def close_position(self, symbol: str, open_reverse: bool = False) -> bool:
        """平仓"""
        try:
            if self.has_open_orders(symbol):
                logger.info(f"📄 平仓前先取消{symbol}的挂单")
                self.cancel_all_orders(symbol)
                time.sleep(1)
            
            position = self.get_position(symbol, force_refresh=True)
            
            if position['size'] == 0:
                logger.info(f"ℹ️ {symbol}无持仓，无需平仓")
                return True
            
            pnl = position.get('unrealized_pnl', 0)
            position_side = position.get('side', 'unknown')
            size = float(position.get('size', 0) or 0)
            side = 'sell' if position.get('side') == 'long' else 'buy'
            
            logger.info(f"📝 准备平仓: {symbol} {side} 数量:{size:.6f} 预计盈亏:{pnl:.2f}U")

            import traceback as _tb
            order_id = None
            last_err = None

            try:
                params = {'reduceOnly': True, 'posSide': position_side, 'tdMode': 'cross'}
                resp = self.exchange.create_order(symbol, 'market', side, size, None, params)
                if isinstance(resp, dict):
                    order_id = resp.get('id') or resp.get('orderId') or resp.get('ordId') or resp.get('clOrdId')
                elif isinstance(resp, list) and resp and isinstance(resp[0], dict):
                    order_id = resp[0].get('id') or resp[0].get('orderId') or resp[0].get('ordId') or resp[0].get('clOrdId')
            except Exception as e1:
                last_err = e1
                logger.error(f"❌ 平仓 create_order 异常: {e1}")
                logger.debug(_tb.format_exc())

            if not order_id:
                try:
                    params = {'reduceOnly': True, 'posSide': position_side, 'tdMode': 'cross'}
                    resp = self.exchange.create_market_order(symbol, side, size, None, params)  # type: ignore[arg-type]
                    if isinstance(resp, dict):
                        order_id = resp.get('id') or resp.get('orderId') or resp.get('ordId') or resp.get('clOrdId')
                    elif isinstance(resp, list) and resp and isinstance(resp[0], dict):
                        order_id = resp[0].get('id') or resp[0].get('orderId') or resp[0].get('ordId') or resp[0].get('clOrdId')
                except Exception as e2:
                    last_err = e2
                    logger.error(f"❌ 平仓 create_market_order 异常: {e2}")
                    logger.debug(_tb.format_exc())

            if not order_id:
                try:
                    inst_id = self.symbol_to_inst_id(symbol)
                    raw_params = {
                        'instId': inst_id,
                        'tdMode': 'cross',
                        'side': side,
                        'posSide': position_side,
                        'reduceOnly': True,
                        'ordType': 'market',
                        'sz': str(size)
                    }
                    resp = self.exchange.privatePostTradeOrder(raw_params)
                    if isinstance(resp, dict):
                        data = resp.get('data') or []
                        if isinstance(data, list) and data:
                            order_id = data[0].get('ordId') or data[0].get('clOrdId') or data[0].get('id')
                        else:
                            order_id = resp.get('ordId') or resp.get('clOrdId') or resp.get('id')
                except Exception as e3:
                    last_err = e3
                    logger.error(f"❌ 平仓 OKX 原生接口异常: {e3}")
                    logger.debug(_tb.format_exc())

            if order_id:
                logger.info(f"✅ 成功平仓{symbol}，方向: {side}，数量: {size:.6f}，盈亏: {pnl:.2f}U")
                self.stats.add_trade(symbol, position_side, pnl)
                time.sleep(2)
                self.get_position(symbol, force_refresh=True)
                self.last_position_state[symbol] = 'none'

                if open_reverse:
                    reverse_side = 'sell' if position_side == 'long' else 'buy'
                    amount = self.calculate_order_amount(symbol)
                    if amount > 0:
                        if self.create_order(symbol, reverse_side, amount):
                            logger.info(f"🔄 平仓后已反手开仓 {symbol} -> {reverse_side}")
                return True

            logger.error(f"❌ 平仓{symbol}失败")
            if last_err:
                logger.error(f"❌ 平仓最后错误：{last_err}")
            return False
                
        except Exception as e:
            logger.error(f"❌ 平仓{symbol}失败: {e}")
            return False
    
    # ATR/ADX/MACD计算方法（保持原有逻辑）
    def calculate_macd(self, prices: List[float]) -> Dict[str, Any]:
        """计算MACD指标"""
        close_array = np.array(prices)
        ema_fast = pd.Series(close_array).ewm(span=self.fast_period, adjust=False).mean().values
        ema_slow = pd.Series(close_array).ewm(span=self.slow_period, adjust=False).mean().values
        macd_line = ema_fast - ema_slow
        signal_line = pd.Series(macd_line).ewm(span=self.signal_period, adjust=False).mean().values
        histogram = macd_line - signal_line
        return {
            'macd': macd_line[-1],
            'signal': signal_line[-1],
            'histogram': histogram[-1],
            'macd_line': macd_line,
            'signal_line': signal_line
        }
    
    def calculate_macd_with_params(self, prices: List[float], f: int, s: int, si: int) -> Dict[str, Any]:
        """按指定参数计算MACD"""
        close_array = np.array(prices)
        ema_fast = pd.Series(close_array).ewm(span=f, adjust=False).mean().values
        ema_slow = pd.Series(close_array).ewm(span=s, adjust=False).mean().values
        macd_line = ema_fast - ema_slow
        signal_line = pd.Series(macd_line).ewm(span=si, adjust=False).mean().values
        histogram = macd_line - signal_line
        return {
            'macd': macd_line[-1],
            'signal': signal_line[-1],
            'histogram': histogram[-1],
            'macd_line': macd_line,
            'signal_line': signal_line
        }
    
    def get_symbol_cfg(self, symbol: str) -> Dict[str, float | str]:
        """返回币种配置，允许 per_symbol_params 覆盖 n/m（sl_n/tp_m）"""
        try:
            base = dict(self.symbol_cfg.get(symbol, {}))
        except Exception:
            base = {}
        if not base:
            base = {"period": 20, "n": 2.0, "m": 3.0, "trigger_pct": 0.010, "trail_pct": 0.006, "update_basis": "close"}
        try:
            p = getattr(self, 'per_symbol_params', {}).get(symbol, {})
            if isinstance(p, dict):
                if 'sl_n' in p:
                    try:
                        base['n'] = float(p['sl_n'])
                    except Exception:
                        pass
                if 'tp_m' in p:
                    try:
                        base['m'] = float(p['tp_m'])
                    except Exception:
                        pass
                if 'tp_pct' in p:
                    try:
                        base['tp_pct'] = float(p['tp_pct'])
                    except Exception:
                        pass
        except Exception:
            pass
        return base

    def _set_initial_sl_tp(self, symbol: str, entry_price: float, atr_val: float, side: str):
        """设置初始 SL/TP"""
        try:
            if atr_val <= 0 or entry_price <= 0 or side not in ('long', 'short'):
                return
            cfg = self.get_symbol_cfg(symbol)
            n = float(cfg['n']); m = float(cfg['m'])
            # 支持可选的固定止盈百分比（tp_pct），优先于 ATR×m
            tp_pct = None
            try:
                tp_pct = float(cfg.get('tp_pct')) if 'tp_pct' in cfg else None
            except Exception:
                tp_pct = None
            if side == 'long':
                sl = entry_price - n * atr_val
                tp = (entry_price * (1 + tp_pct)) if (tp_pct and tp_pct > 0) else (entry_price + m * atr_val)
                side_num = 1.0
                self.trailing_peak[symbol] = max(entry_price, self.trailing_peak.get(symbol, entry_price))
            else:
                sl = entry_price + n * atr_val
                tp = (entry_price * (1 - tp_pct)) if (tp_pct and tp_pct > 0) else (entry_price - m * atr_val)
                side_num = -1.0
                self.trailing_trough[symbol] = min(entry_price, self.trailing_trough.get(symbol, entry_price)) if symbol in self.trailing_trough else entry_price
            self.sl_tp_state[symbol] = {'sl': float(sl), 'tp': float(tp), 'side': side_num, 'entry': float(entry_price)}
        except Exception:
            pass

    def _update_trailing_stop(self, symbol: str, current_price: float, atr_val: float, side: str):
        """动态移动止损"""
        try:
            st = self.sl_tp_state.get(symbol)
            if not st or atr_val <= 0 or current_price <= 0 or side not in ('long', 'short'):
                return
            cfg = self.get_symbol_cfg(symbol)
            n = float(cfg['n']); trigger_pct = float(cfg['trigger_pct']); trail_pct = float(cfg['trail_pct'])
            entry = float(st.get('entry', 0) or 0)
            if entry <= 0:
                return

            basis_price = float(current_price)
            if side == 'long':
                peak = max(self.trailing_peak.get(symbol, entry), basis_price)
                self.trailing_peak[symbol] = peak
                # 当价格偏离入场 ≥1.2×ATR 时也启动锁盈
                activated = (basis_price >= entry * (1 + trigger_pct)) or ((basis_price - entry) >= (1.2 * atr_val))
                atr_sl = basis_price - n * atr_val
                percent_sl = peak * (1 - trail_pct) if activated else st['sl']
                new_sl = max(st['sl'], atr_sl, percent_sl)
                # 仅当新SL更有利且至少提升最小阈值(×ATR)才更新
                if new_sl > st['sl'] and (new_sl - st['sl']) >= (self.trail_sl_min_delta_atr * atr_val):
                    st['sl'] = float(new_sl)
            else:
                trough_prev = self.trailing_trough.get(symbol, entry)
                trough = min(trough_prev, basis_price) if trough_prev else basis_price
                self.trailing_trough[symbol] = trough
                # 当价格偏离入场 ≥1.2×ATR 时也启动锁盈（空头）
                activated = (basis_price <= entry * (1 - trigger_pct)) or ((entry - basis_price) >= (1.2 * atr_val))
                atr_sl = basis_price + n * atr_val
                percent_sl = trough * (1 + trail_pct) if activated else st['sl']
                new_sl = min(st['sl'], atr_sl, percent_sl)
                # 仅当新SL更有利且至少提升最小阈值(×ATR)才更新（空头）
                if new_sl < st['sl'] and (st['sl'] - new_sl) >= (self.trail_sl_min_delta_atr * atr_val):
                    st['sl'] = float(new_sl)
            # 三阶段追踪：锁本 -> 锁小利 -> 分批止盈/平仓
            try:
                entry = float(st.get('entry', 0) or 0)
                if entry > 0 and atr_val > 0:
                    profit = (basis_price - entry) if side == 'long' else (entry - basis_price)
                    atr_mult = profit / atr_val if atr_val > 0 else 0.0
                    # Stage1: 锁本
                    if atr_mult >= self.trail_stage_1:
                        if side == 'long':
                            st['sl'] = max(st['sl'], entry)
                        else:
                            st['sl'] = min(st['sl'], entry)
                    # Stage2: 锁小利（entry ± 0.8×ATR）
                    if atr_mult >= self.trail_stage_2:
                        if side == 'long':
                            st['sl'] = max(st['sl'], entry + self.trail_stage2_offset * atr_val)
                        else:
                            st['sl'] = min(st['sl'], entry - self.trail_stage2_offset * atr_val)
                    # Stage3: 分批止盈（默认减仓30%，仅执行一次）
                    if atr_mult >= self.trail_stage_3 and (not self.stage3_done.get(symbol, False)):
                        try:
                            pos = self.get_position(symbol, force_refresh=True)
                            sz = float(pos.get('size', 0) or 0)
                            if sz > 0 and 0 < self.partial_tp_ratio_stage3 < 1:
                                cut = max(0.0, min(sz, sz * self.partial_tp_ratio_stage3))
                                if cut > 0:
                                    reduce_side = 'sell' if side == 'long' else 'buy'
                                    pos_side = 'long' if side == 'long' else 'short'
                                    if self.reduce_only_market(symbol, reduce_side, cut, pos_side):
                                        logger.info(f"✅ Stage3分批止盈 {symbol}: 减仓 {cut:.6f} ({self.partial_tp_ratio_stage3:.2f})")
                                        self.stage3_done[symbol] = True
                        except Exception as _e:
                            logger.warning(f"⚠️ Stage3 分批止盈异常 {symbol}: {_e}")
            except Exception:
                pass
            self.sl_tp_state[symbol] = st
        except Exception:
            pass

    def _check_hard_stop(self, symbol: str, current_price: float, side: str) -> bool:
        """硬止损：当亏损超过阈值(按入场价百分比)立即市价平仓。返回是否已执行平仓。"""
        try:
            st = self.sl_tp_state.get(symbol)
            if not st:
                return False
            entry = float(st.get('entry', 0) or 0)
            if entry <= 0 or current_price <= 0:
                return False
            max_loss_pct = float(self.hard_sl_max_loss_pct or 0.0)
            if max_loss_pct <= 0:
                return False
            if side == 'long':
                loss_pct = max(0.0, (entry - current_price) / entry)
            else:
                loss_pct = max(0.0, (current_price - entry) / entry)
            if loss_pct >= max_loss_pct:
                logger.warning(f"🛑 硬止损触发 {symbol}: 亏损比例={loss_pct:.4%} ≥ 阈值={max_loss_pct:.2%}，立即平仓")
                self.close_position(symbol, open_reverse=False)
                return True
            return False
        except Exception as e:
            logger.warning(f"⚠️ 硬止损检查异常 {symbol}: {e}")
            return False

    def _maybe_partial_take_profit(self, symbol: str, current_price: float, atr_val: float, side: str):
        """分批止盈：基于 ATR 阶梯，达到阈值即按比例减仓"""
        try:
            tiers_str = os.environ.get('PARTIAL_TP_TIERS', '').strip()  # 例如: "1.5:0.3,3.0:0.3"
            if not tiers_str or atr_val <= 0:
                return
            st = self.sl_tp_state.get(symbol)
            pos = self.get_position(symbol, force_refresh=True)
            size = float(pos.get('size', 0) or 0)
            if size <= 0 or not st:
                return
            entry = float(st.get('entry', 0) or 0)
            if entry <= 0 or current_price <= 0:
                return
            # 计算浮盈(以 ATR 倍数)
            profit = (current_price - entry) if side == 'long' else (entry - current_price)
            atr_mult = profit / atr_val if atr_val > 0 else 0.0
            done = self.partial_tp_done.setdefault(symbol, set())
            for seg in tiers_str.split(','):
                seg = seg.strip()
                if not seg or ':' not in seg:
                    continue
                th_s, ratio_s = seg.split(':', 1)
                try:
                    th = float(th_s); ratio = float(ratio_s)
                except Exception:
                    continue
                key = f"{th:.3f}"
                if atr_mult >= th and key not in done and 0 < ratio < 1:
                    # 执行部分减仓
                    qty = max(0.0, min(size * ratio, size))
                    if qty <= 0:
                        continue
                    side_reduce = 'sell' if side == 'long' else 'buy'
                    if self.reduce_only_market(symbol, side_reduce, qty, side):
                        done.add(key)
                        logger.info(f"✅ 分批止盈 {symbol}: 触发 {th}×ATR，减仓比例 {ratio:.2f}，数量 {qty:.6f}")
                        # 更新剩余持仓尺寸
                        size -= qty
                        if size <= 0:
                            break
                    else:
                        logger.warning(f"⚠️ 分批止盈下单失败 {symbol}: 阶梯 {th}×ATR, 比例 {ratio:.2f}")
        except Exception as e:
            logger.warning(f"⚠️ 分批止盈异常 {symbol}: {e}")
    
    def place_okx_tp_sl(self, symbol: str, entry_price: float, side: str, atr_val: float) -> bool:
        """在OKX侧同时挂TP/SL条件单。优先 oco，失败(51000)回退 tp_sl；严格以 sCode 判定成功。"""
        try:
            if self.okx_tp_sl_placed.get(symbol):
                return True
            inst_id = self.symbol_to_inst_id(symbol)
            if not inst_id or entry_price <= 0 or atr_val <= 0 or side not in ('long', 'short'):
                return False
            pos = self.get_position(symbol, force_refresh=True)
            size = float(pos.get('size', 0) or 0)
            if size <= 0:
                logger.warning(f"⚠️ 无有效持仓数量，跳过挂TP/SL {symbol}")
                return False

            try:
                self.cancel_symbol_tp_sl(symbol)
                time.sleep(0.3)
            except Exception:
                pass

            cfg = self.get_symbol_cfg(symbol)
            n = float(cfg.get('n', self.atr_sl_n)); m = float(cfg.get('m', self.atr_tp_m))
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
            
            try:
                last_price = 0.0
                tkr = self.exchange.publicGetMarketTicker({'instId': inst_id})
                if isinstance(tkr, dict):
                    d = tkr.get('data') or []
                    if isinstance(d, list) and d:
                        last_price = float(d[0].get('last') or d[0].get('lastPx') or 0.0)
                price_prec = int(self.markets_info.get(symbol, {}).get('price_precision', 4))
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
            except Exception:
                pass

            def _post_algo(ord_type: str):
                payload = {
                    'instId': inst_id,
                    'tdMode': 'cross',
                    'posSide': pos_side,
                    'side': ord_side,
                    'ordType': ord_type,
                    'reduceOnly': True,
                    'sz': f"{size}",
                    'tpTriggerPx': f"{tp_trigger}",
                    'tpOrdPx': '-1',
                    'slTriggerPx': f"{sl_trigger}",
                    'slOrdPx': '-1',

                }
                return self.exchange.privatePostTradeOrderAlgo(payload)

            def _is_success(resp_obj: Any) -> bool:
                try:
                    if not isinstance(resp_obj, dict):
                        return False
                    if str(resp_obj.get('code', '')) not in ('0', '200', '0.0'):
                        return False
                    data = resp_obj.get('data') or []
                    if isinstance(data, list) and data:
                        # 任一条 sCode == '0' 视为成功
                        return any(str(x.get('sCode', '')) == '0' for x in data if isinstance(x, dict))
                    # 没有 data 时，也不视为成功，避免误判
                    return False
                except Exception:
                    return False

            # 先试 oco
            resp = _post_algo('oco')
            if _is_success(resp):
                logger.info(f"📌 交易所侧TP/SL已挂 {symbol}: size={size:.6f} TP@{tp_trigger:.6f} SL@{sl_trigger:.6f} (ordType=oco)")
                self.okx_tp_sl_placed[symbol] = True
                self.tp_sl_last_placed[symbol] = time.time()
                return True

            # 如出现 51000 等参数问题，回退尝试 tp_sl
            try:
                msg = str(resp)
            except Exception:
                msg = ''
            if '51000' in msg or 'ordType' in msg.lower():
                try:
                    resp2 = _post_algo('tp_sl')
                    if _is_success(resp2):
                        logger.info(f"📌 交易所侧TP/SL已挂 {symbol}: size={size:.6f} TP@{tp_trigger:.6f} SL@{sl_trigger:.6f} (ordType=tp_sl)")
                        self.okx_tp_sl_placed[symbol] = True
                        self.tp_sl_last_placed[symbol] = time.time()
                        return True
                    logger.warning(f"⚠️ 交易所侧TP/SL挂单失败 {symbol} 回退tp_sl: {resp2}")
                    return False
                except Exception as _e2:
                    logger.warning(f"⚠️ 交易所侧TP/SL挂单异常(回退tp_sl) {symbol}: {_e2}")
                    return False

            logger.warning(f"⚠️ 交易所侧TP/SL挂单失败 {symbol}: {resp}")
            return False
        except Exception as e:
            logger.warning(f"⚠️ 交易所侧TP/SL挂单异常 {symbol}: {e}")
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
        except Exception:
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
        except Exception:
            return 0.0

    def analyze_symbol(self, symbol: str) -> Dict[str, str]:
        """分析单个交易对"""
        try:
            klines = self.get_klines(symbol, 100)
            if not klines:
                return {'signal': 'hold', 'reason': '数据获取失败'}
            
            closes = [kline['close'] for kline in klines]

            if len(closes) < 2:
                return {'signal': 'hold', 'reason': '数据不足'}

            try:
                ps = getattr(self, 'per_symbol_params', {}).get(symbol, {})
            except Exception:
                ps = {}
            try:
                if isinstance(ps, dict) and ps.get('atr_period'):
                    atr_period = int(ps.get('atr_period'))
                else:
                    atr_period = int((os.environ.get('ATR_PERIOD') or '14').strip())
            except Exception:
                atr_period = 14
            try:
                atr_ratio_thresh = float((os.environ.get('ATR_RATIO_THRESH') or '0.004').strip())
            except Exception:
                atr_ratio_thresh = 0.004
            try:
                if isinstance(ps, dict) and ps.get('adx_period'):
                    adx_period = int(ps.get('adx_period'))
                else:
                    adx_period = int((os.environ.get('ADX_PERIOD') or '14').strip())
            except Exception:
                adx_period = 14
            try:
                adx_min_trend = float((os.environ.get('ADX_MIN_TREND') or '25').strip())
            except Exception:
                adx_min_trend = 18.0

            close_price = float(closes[-1])
            atr_val = self.calculate_atr(klines, atr_period)
            adx_val = self.calculate_adx(klines, adx_period)

            if atr_val > 0 and close_price > 0:
                atr_ratio = atr_val / close_price
                if atr_ratio < atr_ratio_thresh:
                    logger.debug(f"ATR滤波提示：波动率低（ATR/收盘={atr_ratio:.4f} < {atr_ratio_thresh}），不拦截信号")

            if adx_val > 0 and adx_val < adx_min_trend:
                logger.debug(f"ADX滤波提示：趋势不足（ADX={adx_val:.1f} < {adx_min_trend}），不拦截信号")

            _p = getattr(self, 'per_symbol_params', {}).get(symbol, {})
            _macd_params = _p.get('macd') if isinstance(_p, dict) else None
            if isinstance(_macd_params, (list, tuple)) and len(_macd_params) == 3:
                macd_current = self.calculate_macd_with_params(closes, int(_macd_params[0]), int(_macd_params[1]), int(_macd_params[2]))
            else:
                macd_current = self.calculate_macd(closes)
            _p = getattr(self, 'per_symbol_params', {}).get(symbol, {})
            _macd_params = _p.get('macd') if isinstance(_p, dict) else None
            if isinstance(_macd_params, (list, tuple)) and len(_macd_params) == 3:
                macd_prev = self.calculate_macd_with_params(closes[:-1], int(_macd_params[0]), int(_macd_params[1]), int(_macd_params[2]))
            else:
                macd_prev = self.calculate_macd(closes[:-1])
            
            position = self.get_position(symbol, force_refresh=True)
            try:
                logger.debug(f"🔍 {symbol} ATR({atr_period})={atr_val:.6f}, ATR/Close={atr_val/close_price:.6f} | ADX({adx_period})={adx_val:.2f}")
            except Exception:
                pass
            
            prev_macd = macd_prev['macd']
            prev_signal = macd_prev['signal']
            prev_hist = macd_prev['histogram']
            current_macd = macd_current['macd']
            current_signal = macd_current['signal']
            current_hist = macd_current['histogram']
            
            logger.debug(f"📊 {symbol} MACD(实时) - 当前: MACD={current_macd:.6f}, Signal={current_signal:.6f}, Hist={current_hist:.6f}")
            
            try:
                _p2 = getattr(self, 'per_symbol_params', {}).get(symbol, {})
                _th = float(_p2.get('adx_min_trend', 0) or 0)
                if _th > 0 and adx_val > 0 and adx_val < _th:
                    return {'signal': 'hold', 'reason': f'ADX不足 {adx_val:.1f} < {_th:.1f}'}
            except Exception:
                pass
            
            if position['size'] == 0:
                buy_cross = (prev_macd <= prev_signal and current_macd > current_signal)
                buy_color = (prev_hist <= 0 and current_hist > 0)
                sell_cross = (prev_macd >= prev_signal and current_macd < current_signal)
                sell_color = (prev_hist >= 0 and current_hist < 0)

                if buy_cross and buy_color:
                    return {'signal': 'buy', 'reason': '双确认：金叉 + 柱状图由负转正'}
                elif sell_cross and sell_color:
                    return {'signal': 'sell', 'reason': '双确认：死叉 + 柱状图由正转负'}
                else:
                    return {'signal': 'hold', 'reason': '等待双确认信号'}
            
            else:
                current_position_side = position['side']
                
                if current_position_side == 'long':
                    if (prev_macd >= prev_signal and current_macd < current_signal) and (current_hist < 0):
                        return {'signal': 'close', 'reason': '多头双确认平仓：死叉且柱状图为负'}
                    else:
                        return {'signal': 'hold', 'reason': '持有多头'}
                
                else:
                    if (prev_macd <= prev_signal and current_macd > current_signal) and (current_hist > 0):
                        return {'signal': 'close', 'reason': '空头双确认平仓：金叉且柱状图为正'}
                    else:
                        return {'signal': 'hold', 'reason': '持有空头'}
                        
        except Exception as e:
            logger.error(f"❌ 分析{symbol}失败: {e}")
            return {'signal': 'hold', 'reason': f'分析异常: {e}'}
    
    def execute_strategy(self):
        """执行策略"""
        logger.info("=" * 70)
        logger.info(f"🚀 开始执行MACD策略 (11个币种，{self.timeframe} 周期)")
        logger.info("=" * 70)
        
        try:
            self.check_sync_needed()
            
            balance = self.get_account_balance()
            logger.info(f"💰 当前账户余额: {balance:.2f} USDT")
            # 熔断机制已移除
            
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
                    if kl:
                        close_price = float(kl[-1]['close'])
                        ps = getattr(self, 'per_symbol_params', {}).get(symbol, {})
                        if isinstance(ps, dict) and ps.get('atr_period'):
                            atr_p = int(ps.get('atr_period'))
                        else:
                            atr_p = int((os.environ.get('ATR_PERIOD') or '14').strip())
                        atr_val = self.calculate_atr(kl, atr_p)
                        if current_position and current_position.get('size', 0) > 0 and atr_val > 0:
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
                                        profit = (close_price - entry_px) if current_position.get('side') == 'long' else (entry_px - close_price)
                                        if profit >= 2.5 * atr_val:
                                            st['sl'] = max(st['sl'], close_price - 1.0 * atr_val) if current_position.get('side') == 'long' else min(st['sl'], close_price + 1.0 * atr_val)
                                        elif profit >= 1.5 * atr_val:
                                            st['sl'] = max(st['sl'], close_price - 1.2 * atr_val) if current_position.get('side') == 'long' else min(st['sl'], close_price + 1.2 * atr_val)
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
                                        okx_ok = self.place_okx_tp_sl(symbol, entry_px, current_position.get('side', 'long'), atr_val)
                                        if okx_ok:
                                            logger.info(f"🔄 更新追踪止盈：冷却达到，已重挂 {symbol}")
                                        else:
                                            logger.warning(f"⚠️ 更新追踪止盈重挂失败 {symbol}")
                                    else:
                                        logger.debug(f"⏳ 距上次挂单未达冷却({self.tp_sl_refresh_interval}s)，跳过重挂 {symbol}")
                                except Exception as _e:
                                    logger.warning(f"⚠️ 更新追踪止盈重挂失败 {symbol}: {_e}")
                                if current_position.get('side') == 'long':
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
                    _pp = getattr(self, 'per_symbol_params', {}).get(symbol, {})
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
    logger.info("🎯 MACD策略程序启动中... (11个币种版本)")
    logger.info("=" * 70)
    
def main():
    """主函数"""
    logger.info("=" * 70)
    logger.info("🎯 MACD策略程序启动中... (11个币种版本)")
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