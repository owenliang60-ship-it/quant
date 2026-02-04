#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一每日扫描 - 串行执行所有扫描器

直接复用现有的扫描器代码，串行执行避免API冲突。

Author: Claude Code
Date: 2026-01-18
"""

import sys
import time
from datetime import datetime
from pathlib import Path

# 切换到scanners目录
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def main():
    log("=" * 60)
    log("每日统一扫描开始")
    log("=" * 60)

    start_time = time.time()

    # 1. PMARP扫描
    log("\n[1/4] 运行 PMARP 扫描器...")
    try:
        import binance_pmarp_scanner
        binance_pmarp_scanner.main()
    except Exception as e:
        log(f"PMARP扫描器错误: {e}")

    time.sleep(2)  # 间隔2秒

    # 2. RVOL扫描
    log("\n[2/4] 运行 RVOL 扫描器...")
    try:
        import binance_rvol_scanner
        binance_rvol_scanner.main()
    except Exception as e:
        log(f"RVOL扫描器错误: {e}")

    time.sleep(2)

    # 3. ADL扫描
    log("\n[3/4] 运行 ADL Ratio 扫描器...")
    try:
        import binance_adl_ratio_scanner
        binance_adl_ratio_scanner.main()
    except Exception as e:
        log(f"ADL扫描器错误: {e}")

    time.sleep(2)

    # 4. EMA120扫描
    log("\n[4/4] 运行 EMA120 扫描器...")
    try:
        import binance_ema120_scanner
        binance_ema120_scanner.main()
    except Exception as e:
        log(f"EMA120扫描器错误: {e}")

    elapsed = time.time() - start_time
    log("\n" + "=" * 60)
    log(f"全部扫描完成，总耗时 {elapsed/60:.1f} 分钟")
    log("=" * 60)

if __name__ == "__main__":
    main()
