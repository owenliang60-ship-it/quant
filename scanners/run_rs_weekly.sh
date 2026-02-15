#!/bin/bash
# RS 长周期动量扫描 (30d/14d/7d) — 每周一次
export TELEGRAM_BOT_TOKEN="8597632779:AAEo_pSVq99hQ62TE6GQbfR5VlBK1U34j-Q"
export TELEGRAM_CHAT_ID="416677679"

cd /root/workspace/Quant/scanners
python3 binance_rs_scanner.py >> /root/workspace/Quant/logs/scan_rs_weekly_$(date +%Y%m%d).log 2>&1
