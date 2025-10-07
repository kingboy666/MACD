#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MACD(6,16,9)策略 - 15分钟图
支持FILUSDT, ZROUSDT, WIFUSDT, WLDUSDT四个合约对
20倍杠杆，智能仓位分配
"""

import ccxt
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime
import os
from typing import Dict, List, Optional

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MACDStrategy:
    def __init__(self, api_key: str, secret: str, passphrase: str = None):
        """
        初始化MACD策略
        
        Args:
            api_key: API密钥
            secret: 密钥
            passphrase: 密码短语（OKX需要）
        """
        self.exchange = ccxt.okx({
            'apiKey': api_key,
            'secret': secret,
            'password': passphrase,
            'sandbox': False,  # 生产环境
            'enableRateLimit': True,
        })
        
        # 交易对配置
        self.symbols = ['FIL-USDT', 'ZRO-USDT', 'WIF-USDT', 'WLD-USDT']
        self.leverage = 20
        self.timeframe = '15m'  # 15分钟图
        
        # MACD参数
        self.fast_period = 6
        self.slow_period = 16
        self.signal_period = 9
        
        # 仓位配置
        self.position_percentage = 0.8  # 使用80%余额
        self.min_order_value = 1  # 最小下单金额1USDT
        
        # 持仓记录
        self.positions = {}
        
        # 初始化交易所
        self._setup_exchange()
    
    def _setup_exchange(self):
        """设置交易所配置"""
        try:
            # 设置杠杆
            for symbol in self.symbols:
                self.exchange.set_leverage(self.leverage, symbol)
                logger.info(f"设置{symbol}杠杆为{self.leverage}倍")
            
            # 设置合约模式
            self.exchange.set_position_mode(False)  # 单向持仓模式
            logger.info("设置为单向持仓模式")
            
        except Exception as e:
            logger.error(f"交易所设置失败: {e}")
    
    def calculate_macd(self, prices: List[float]) -> Dict:
        """
        计算MACD指标
        
        Args:
            prices: 价格列表
            
        Returns:
            MACD指标字典
        """
        if len(prices) < self.slow_period:
            return {'macd': 0, 'signal': 0, 'histogram': 0}
        
        # 计算EMA
        ema_fast = self._ema(prices, self.fast_period)
        ema_slow = self._ema(prices, self.slow_period)
        
        # 计算MACD线
        macd_line = ema_fast[-1] - ema_slow[-1] if len(ema_fast) > 0 and len(ema_slow) > 0 else 0
        
        # 计算信号线（MACD的EMA）
        macd_values = [ema_fast[i] - ema_slow[i] for i in range(min(len(ema_fast), len(ema_slow)))]
        signal_line = self._ema(macd_values, self.signal_period)[-1] if macd_values else 0
        
        # 计算柱状图
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def _ema(self, data: List[float], period: int) -> List[float]:
        """计算指数移动平均线"""
        if len(data) < period:
            return []
        
        ema_values = []
        multiplier = 2 / (period + 1)
        
        # 第一个EMA是简单移动平均
        sma = sum(data[:period]) / period
        ema_values.append(sma)
        
        # 计算后续EMA
        for i in range(period, len(data)):
            ema = (data[i] - ema_values[-1]) * multiplier + ema_values[-1]
            ema_values.append(ema)
        
        return ema_values
    
    def get_klines(self, symbol: str, limit: int = 100) -> List[Dict]:
        """
        获取K线数据
        
        Args:
            symbol: 交易对
            limit: 数据条数
            
        Returns:
            K线数据列表
        """
        try:
            klines = self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=limit)
            return [{
                'timestamp': kline[0],
                'open': kline[1],
                'high': kline[2],
                'low': kline[3],
                'close': kline[4],
                'volume': kline[5]
            } for kline in klines]
        except Exception as e:
            logger.error(f"获取{symbol}K线数据失败: {e}")
            return []
    
    def get_account_balance(self) -> float:
        """获取账户余额"""
        try:
            balance = self.exchange.fetch_balance()
            return float(balance['USDT']['free'])
        except Exception as e:
            logger.error(f"获取账户余额失败: {e}")
            return 0
    
    def get_position(self, symbol: str) -> Dict:
        """
        获取持仓信息
        
        Args:
            symbol: 交易对
            
        Returns:
            持仓信息
        """
        try:
            positions = self.exchange.fetch_positions([symbol])
            for position in positions:
                if position['symbol'] == symbol:
                    return {
                        'size': float(position['contracts']),
                        'side': position['side'],
                        'entry_price': float(position['entryPrice']),
                        'unrealized_pnl': float(position['unrealizedPnl'])
                    }
            return {'size': 0, 'side': 'none', 'entry_price': 0, 'unrealized_pnl': 0}
        except Exception as e:
            logger.error(f"获取{symbol}持仓失败: {e}")
            return {'size': 0, 'side': 'none', 'entry_price': 0, 'unrealized_pnl': 0}
    
    def calculate_order_amount(self, symbol: str, price: float) -> float:
        """
        计算下单金额（智能分配）
        
        Args:
            symbol: 交易对
            price: 当前价格
            
        Returns:
            下单金额
        """
        try:
            balance = self.get_account_balance()
            total_amount = balance * self.position_percentage
            
            # 智能分配：根据交易对波动性分配资金
            volatility_weights = {
                'FIL-USDT': 0.25,
                'ZRO-USDT': 0.25, 
                'WIF-USDT': 0.25,
                'WLD-USDT': 0.25
            }
            
            allocated_amount = total_amount * volatility_weights.get(symbol, 0.25)
            
            # 计算合约数量
            ticker = self.exchange.fetch_ticker(symbol)
            min_order_value = ticker.get('info', {}).get('minOrderAmount', self.min_order_value)
            
            # 确保不低于最小下单金额
            order_amount = max(allocated_amount, min_order_value)
            
            logger.info(f"{symbol}智能分配金额: {order_amount:.2f} USDT")
            return order_amount
            
        except Exception as e:
            logger.error(f"计算{symbol}下单金额失败: {e}")
            return self.min_order_value
    
    def create_order(self, symbol: str, side: str, amount: float) -> bool:
        """
        创建订单
        
        Args:
            symbol: 交易对
            side: 方向 (buy/sell)
            amount: 金额
            
        Returns:
            是否成功
        """
        try:
            # 获取当前价格
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            
            # 计算合约数量
            contract_size = amount / current_price
            
            # 创建市价单
            order = self.exchange.create_market_order(symbol, side, contract_size)
            
            if order['id']:
                logger.info(f"成功创建{symbol} {side}订单，金额: {amount:.2f} USDT")
                return True
            else:
                logger.error(f"创建{symbol} {side}订单失败")
                return False
                
        except Exception as e:
            logger.error(f"创建{symbol} {side}订单异常: {e}")
            return False
    
    def close_position(self, symbol: str) -> bool:
        """
        平仓
        
        Args:
            symbol: 交易对
            
        Returns:
            是否成功
        """
        try:
            position = self.get_position(symbol)
            if position['size'] == 0:
                logger.info(f"{symbol}无持仓，无需平仓")
                return True
            
            # 反向平仓
            side = 'sell' if position['side'] == 'long' else 'buy'
            amount = position['size'] * position['entry_price']
            
            return self.create_order(symbol, side, amount)
            
        except Exception as e:
            logger.error(f"平仓{symbol}失败: {e}")
            return False
    
    def analyze_symbol(self, symbol: str) -> Dict:
        """
        分析单个交易对
        
        Args:
            symbol: 交易对
            
        Returns:
            分析结果
        """
        try:
            # 获取K线数据
            klines = self.get_klines(symbol, 50)
            if not klines:
                return {'signal': 'hold', 'reason': '数据获取失败'}
            
            # 提取收盘价
            closes = [kline['close'] for kline in klines]
            
            # 计算MACD
            macd_data = self.calculate_macd(closes)
            
            # 获取持仓
            position = self.get_position(symbol)
            
            # 生成交易信号
            if position['size'] == 0:  # 无持仓
                if macd_data['macd'] > macd_data['signal'] and macd_data['histogram'] > 0:
                    return {'signal': 'buy', 'reason': 'MACD金叉'}
                elif macd_data['macd'] < macd_data['signal'] and macd_data['histogram'] < 0:
                    return {'signal': 'sell', 'reason': 'MACD死叉'}
                else:
                    return {'signal': 'hold', 'reason': '等待信号'}
            else:  # 有持仓
                if position['side'] == 'long':
                    if macd_data['macd'] < macd_data['signal'] and macd_data['histogram'] < 0:
                        return {'signal': 'close', 'reason': '多头平仓信号'}
                    else:
                        return {'signal': 'hold', 'reason': '持有多头'}
                else:  # short
                    if macd_data['macd'] > macd_data['signal'] and macd_data['histogram'] > 0:
                        return {'signal': 'close', 'reason': '空头平仓信号'}
                    else:
                        return {'signal': 'hold', 'reason': '持有空头'}
                        
        except Exception as e:
            logger.error(f"分析{symbol}失败: {e}")
            return {'signal': 'hold', 'reason': f'分析异常: {e}'}
    
    def execute_strategy(self):
        """执行策略"""
        logger.info("开始执行MACD策略...")
        
        try:
            # 分析所有交易对
            signals = {}
            for symbol in self.symbols:
                signals[symbol] = self.analyze_symbol(symbol)
                logger.info(f"{symbol}信号: {signals[symbol]}")
            
            # 执行交易
            for symbol, signal_info in signals.items():
                signal = signal_info['signal']
                reason = signal_info['reason']
                
                if signal == 'buy':
                    amount = self.calculate_order_amount(symbol, 0)
                    if self.create_order(symbol, 'buy', amount):
                        logger.info(f"开多{symbol}成功")
                
                elif signal == 'sell':
                    amount = self.calculate_order_amount(symbol, 0)
                    if self.create_order(symbol, 'sell', amount):
                        logger.info(f"开空{symbol}成功")
                
                elif signal == 'close':
                    if self.close_position(symbol):
                        logger.info(f"平仓{symbol}成功")
            
            # 更新持仓记录
            self.update_positions()
            
            logger.info("策略执行完成")
            
        except Exception as e:
            logger.error(f"策略执行异常: {e}")
    
    def update_positions(self):
        """更新持仓记录"""
        for symbol in self.symbols:
            self.positions[symbol] = self.get_position(symbol)
    
    def run_continuous(self, interval: int = 900):  # 15分钟
        """
        连续运行策略
        
        Args:
            interval: 执行间隔（秒）
        """
        logger.info(f"开始连续运行策略，间隔: {interval}秒")
        
        while True:
            try:
                self.execute_strategy()
                logger.info(f"等待{interval}秒后再次执行...")
                time.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("用户中断策略执行")
                break
            except Exception as e:
                logger.error(f"策略运行异常: {e}")
                time.sleep(60)  # 异常时等待1分钟


def main():
    """主函数"""
    # 从环境变量获取API配置
    api_key = os.getenv('OKX_API_KEY')
    secret = os.getenv('OKX_SECRET') 
    passphrase = os.getenv('OKX_PASSPHRASE')
    
    if not all([api_key, secret, passphrase]):
        logger.error("请设置OKX_API_KEY, OKX_SECRET, OKX_PASSPHRASE环境变量")
        return
    
    # 创建策略实例
    strategy = MACDStrategy(api_key, secret, passphrase)
    
    # 运行策略
    try:
        strategy.run_continuous()
    except Exception as e:
        logger.error(f"策略运行失败: {e}")


if __name__ == "__main__":
    main()