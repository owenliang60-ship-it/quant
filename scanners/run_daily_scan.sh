#!/bin/bash
cd /root/workspace/Quant/scanners
python3 daily_scan_all.py >> /root/workspace/Quant/logs/scan_$(date +%Y%m%d).log 2>&1
