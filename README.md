# 11交易对 - Railway 实盘部署说明

## 启动方式
- 平台自动执行根目录 Procfile：
  - `start: python -u "11交易对/main.py"`
- main.py 会以 `__main__` 环境执行根目录的 `1.txt`，无需修改原策略代码。

## 依赖安装
- Railway 的 Nixpacks 会根据根目录 `requirements.txt` 自动安装：
  - ccxt、pandas、numpy、pytz

## 必需环境变量（已在平台配置）
- `OKX_API_KEY`
- `OKX_SECRET_KEY`
- `OKX_PASSPHRASE`

## 可选环境变量（策略参数）
- `SCAN_INTERVAL` 扫描间隔（秒），默认 2
- `ATR_PERIOD` 默认 14
- `ATR_SL_N` 默认 2.0
- `ATR_TP_M` 默认 3.0
- `ORDER_NOTIONAL_FACTOR` 下单名义金额放大因子，默认 50
- `TARGET_NOTIONAL_USDT` 固定每次下单名义金额（设置后优先生效）
- `MIN_PER_SYMBOL_USDT` / `MAX_PER_SYMBOL_USDT` 每币种金额上下限
- 其余每币种参数在代码内 symbol_cfg / per_symbol_params 已有默认

## 运行与日志
- 策略为长循环，实时巡检。日志输出到标准输出，可在 Railway 控制台查看。
- 交易统计文件 `trading_stats.json` 会在运行工作目录生成（持久化需使用平台存储方案）。

## 注意事项
- 账户需开通 OKX USDT 计价永续合约权限（SWAP），并确保余额充足。
- 若启动前已有持仓，程序会为持仓补挂 OCO 止盈止损，并继续根据信号管理。

---

## 部署步骤详解

1) 代码准备
- 根目录包含：`Procfile`、`requirements.txt`、`1.txt`，以及目录 `11交易对/`（含 `main.py`）。
- 不在代码中写入任何密钥，全部通过 Railway 环境变量提供。

2) 创建 Railway 项目
- 在 Railway 控制台 New Project，导入该代码仓库或上传。
- 平台会自动依据 `requirements.txt` 安装依赖。

3) 配置环境变量
- 在 Variables 中设置：`OKX_API_KEY`、`OKX_SECRET_KEY`、`OKX_PASSPHRASE`。
- 可选：`SCAN_INTERVAL`、`ATR_PERIOD`、`ATR_SL_N`、`ATR_TP_M`、`ORDER_NOTIONAL_FACTOR`、`TARGET_NOTIONAL_USDT`、`MIN_PER_SYMBOL_USDT`、`MAX_PER_SYMBOL_USDT`。

4) 启动命令
- Railway 根据 `Procfile` 自动使用：`python -u "11交易对/main.py"`。
- 若需手动，进入服务 Start Command 填写上述命令保存。

5) 部署与运行
- 点击 Deploy 或等待自动部署完成。
- 在 Logs 中查看实时输出与策略运行状态。

6) 验证与排障
- 启动日志包含：环境变量检查、市场与杠杆设置、余额/持仓/信号分析。
- 若提示缺少环境变量，补齐后重新部署。
- 若找不到 `1.txt`，确认其位于项目根目录。
- 若 OCO 挂单失败，检查价格精度与最小间距（系统会自动对齐 tick 并重挂）。

7) 建议
- 先小额试跑，逐步提高 `TARGET_NOTIONAL_USDT` 或 `ORDER_NOTIONAL_FACTOR`。
- 如需固定 Python 版本，可在 Railway 设置中指定 Python 3.11（或添加平台支持的运行时配置）。