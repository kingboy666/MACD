# MACD交易策略 - 本地部署说明

## 环境要求

- Python 3.7+
- 必要的Python包

## 安装依赖

```bash
pip install pandas numpy ccxt pandas_ta tenacity python-dotenv
```

## API配置

### 方式1: OKX.env文件 (推荐，最安全)

在代码同目录下创建 `OKX.env` 文件：
```env
OKX_API_KEY=你的API_KEY
OKX_API_SECRET=你的SECRET_KEY
OKX_API_PASSPHRASE=你的PASSPHRASE
```

**注意**: 程序会优先查找 `OKX.env` 文件，如果不存在则查找 `.env` 文件。

### 方式2: 环境变量

**Windows:**
```cmd
set OKX_API_KEY=your_api_key
set OKX_API_SECRET=your_secret_key
set OKX_API_PASSPHRASE=your_passphrase
```

**Linux/Mac:**
```bash
export OKX_API_KEY=your_api_key
export OKX_API_SECRET=your_secret_key
export OKX_API_PASSPHRASE=your_passphrase
```

### 方式3: 代码中直接设置

在代码中调用：
```python
set_okx_api('your_api_key', 'your_secret_key', 'your_passphrase')
```

### 方式4: 修改配置字典

直接在代码中修改 `OKX_CONFIG` 字典：
```python
OKX_CONFIG = {
    'api_key': 'your_api_key',
    'secret_key': 'your_secret_key',  
    'passphrase': 'your_passphrase'
}
```

## 运行程序

```bash
python "MACD(6,32,9).txt"
```

## 策略说明

- **交易所**: OKX永续合约
- **指标**: MACD(6,32,9) + ADX过滤
- **时间框架**: 30分钟
- **交易对**: 热度前10合约 + FILUSDT, ZROUSDT, WIFUSDT, WLDUSDT
- **风险管理**: 2%止损, 4%止盈, 移动止盈
- **资金管理**: 80%资金智能分配到多个仓位
- **杠杆**: 自动调整(最高100倍)

## 注意事项

1. 确保API权限包含合约交易
2. 建议先在模拟环境测试
3. 监控程序运行状态
4. 定期检查持仓和风险