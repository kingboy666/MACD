#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
入口程序：在 Railway 上运行 11 个交易对 MACD 策略
此入口会加载并以 __main__ 方式执行项目根目录的 1.txt（完整策略代码）。
"""
import os
import sys

def main():
    # 项目结构：当前文件在 11交易对/ 目录，下级运行根目录的 1.txt
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(base_dir, ".."))
    script_path = os.path.join(repo_root, "1.txt")

    if not os.path.exists(script_path):
        sys.stderr.write(f"❌ 未找到策略文件: {script_path}\n")
        sys.exit(1)

    # 以 __main__ 环境执行原策略文件，使其 main() 正常触发
    # 注意：1.txt 内部有重复的 main() 定义，后者覆盖前者，不影响运行
    g = {"__name__": "__main__"}
    with open(script_path, "r", encoding="utf-8") as f:
        code = f.read()
    exec(compile(code, script_path, "exec"), g)

if __name__ == "__main__":
    main()