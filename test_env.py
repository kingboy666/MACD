#!/usr/bin/env python3
import os
from dotenv import load_dotenv

print("=== 环境变量诊断工具 ===")

# 尝试加载 .env 文件
load_dotenv()

print(f"当前工作目录: {os.getcwd()}")
print(f".env 文件是否存在: {os.path.exists('.env')}")

print("\n=== 检查 OKX 相关环境变量 ===")
okx_vars = ['OKX_API_KEY', 'OKX_SECRET_KEY', 'OKX_PASSPHRASE']

for var in okx_vars:
    value = os.getenv(var)
    if value:
        print(f"✅ {var}: 已设置 (长度: {len(value)})")
    else:
        print(f"❌ {var}: 未设置或为空")

print("\n=== 所有环境变量 ===")
all_vars = list(os.environ.keys())
okx_related = [var for var in all_vars if 'OKX' in var.upper()]

if okx_related:
    print("找到 OKX 相关变量:")
    for var in okx_related:
        print(f"  - {var}")
else:
    print("未找到任何 OKX 相关环境变量")

print(f"\n总共有 {len(all_vars)} 个环境变量")

# 显示所有环境变量（前50个字符）
print("\n=== 所有环境变量列表 ===")
for i, var in enumerate(sorted(all_vars)):
    value = os.environ.get(var, '')
    if len(value) > 50:
        display_value = value[:47] + "..."
    else:
        display_value = value
    print(f"{i+1:2d}. {var}: {display_value}")
    
    # 如果变量太多，只显示前30个
    if i >= 29:
        print(f"... 还有 {len(all_vars) - 30} 个变量")
        break

# 显示一些常见的系统环境变量
common_vars = ['PATH', 'HOME', 'USER', 'PWD', 'RAILWAY_ENVIRONMENT', 'RAILWAY_PROJECT_ID']
print("\n=== 重要系统变量 ===")
for var in common_vars:
    value = os.getenv(var)
    if value:
        print(f"{var}: {value[:50]}...")
    else:
        print(f"{var}: 未设置")

print("\n=== Railway 特定检查 ===")
railway_vars = [var for var in all_vars if 'RAILWAY' in var.upper()]
if railway_vars:
    print("找到 Railway 相关变量:")
    for var in railway_vars:
        value = os.environ.get(var, '')
        print(f"  - {var}: {value[:30]}...")
else:
    print("未找到 Railway 相关环境变量")

print("\n=== 诊断完成 ===")
print("如果 OKX 变量显示未设置，请检查 Railway Variables 设置")