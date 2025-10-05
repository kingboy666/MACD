import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 输出到文件，避免PowerShell输出问题
with open('test_result.txt', 'w', encoding='utf-8') as f:
    f.write("=== 环境变量测试结果 ===\n")
    
    # 检查 OKX API 配置
    api_key = os.getenv('OKX_API_KEY')
    secret_key = os.getenv('OKX_SECRET_KEY')  
    passphrase = os.getenv('OKX_PASSPHRASE')
    
    if api_key:
        f.write(f"✅ OKX_API_KEY: 已设置 (长度: {len(api_key)})\n")
    else:
        f.write("❌ OKX_API_KEY: 未设置\n")
        
    if secret_key:
        f.write(f"✅ OKX_SECRET_KEY: 已设置 (长度: {len(secret_key)})\n")
    else:
        f.write("❌ OKX_SECRET_KEY: 未设置\n")
        
    if passphrase:
        f.write(f"✅ OKX_PASSPHRASE: 已设置 (长度: {len(passphrase)})\n")
    else:
        f.write("❌ OKX_PASSPHRASE: 未设置\n")
    
    # 检查是否所有配置都正确
    if all([api_key, secret_key, passphrase]):
        f.write("\n🎉 所有 API 配置都已正确设置！\n")
        f.write("现在可以尝试运行交易程序了。\n")
    else:
        f.write("\n⚠️  请检查 .env 文件中的 API 配置\n")
    
    f.write("\n=== 测试完成 ===\n")

print("测试完成，结果已保存到 test_result.txt 文件")