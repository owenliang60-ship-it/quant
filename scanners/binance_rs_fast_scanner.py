#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
币安合约 RS 快速动量扫描器 (7d/3d/1d 短周期版)

与 binance_rs_scanner.py (30d/14d/7d) 互补，捕捉更短期的动量信号。
复用 RS scanner 的 API/缓存/Telegram 模块。

窗口: 7d / 3d / 1d
跳过: 1 天
年化: √365
最小数据: 15 天

Author: Claude Code
Date: 2026-02-15
"""

import time
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import rankdata, zscore as scipy_zscore, linregress

from binance_rs_scanner import (
    get_all_usdt_futures,
    update_daily_cache,
    send_telegram_alert,
)

# ============================================================
# 配置 (短周期版)
# ============================================================
FAST_CONFIG = {
    "window_long": 7,     # 1周
    "window_mid": 3,      # 3天
    "window_short": 1,    # 1天
    "skip_days": 1,
    "annual_factor": 365,
    "min_data_days": 15,

    # 成交量筛选
    "volume_filter_days": 7,  # 用7天总成交量筛选
    "volume_top_n": 100,      # 只取 Top 100

    "top_n": 20,
}


# ============================================================
# RS Rating 计算核心 (短周期版)
# ============================================================

def compute_fast_rs_b(price_dict: dict) -> pd.DataFrame:
    """
    Method B — 风险调整 Z-Score (7d/3d/1d)

    7d 经年化波动率调整, 3d 和 1d 不调整 (数据点太少)
    """
    cfg = FAST_CONFIG
    skip = cfg["skip_days"]
    w_long = cfg["window_long"]    # 7
    w_mid = cfg["window_mid"]      # 3
    w_short = cfg["window_short"]  # 1
    min_days = cfg["min_data_days"]
    ann = np.sqrt(cfg["annual_factor"])

    records = []

    for symbol, close in price_dict.items():
        n = len(close)
        if n < min_days:
            continue

        end_idx = n - 1 - skip
        if end_idx < 0 or (end_idx - w_long) < 0:
            continue

        # --- 收益率 ---
        ret_7d = close[end_idx] / close[end_idx - w_long] - 1
        ret_3d = close[end_idx] / close[end_idx - w_mid] - 1
        ret_1d = close[end_idx] / close[end_idx - w_short] - 1

        # --- 风险调整 (仅 7d) ---
        daily_returns = np.diff(close) / close[:-1]
        vol_start = max(0, end_idx - w_long)
        vol_7d = np.std(daily_returns[vol_start:end_idx], ddof=1) * ann

        ra_7d = ret_7d / vol_7d if vol_7d > 1e-10 else 0.0
        ra_3d = ret_3d  # 3d 太短不做风险调整
        ra_1d = ret_1d  # 1d 不做风险调整

        records.append({
            "symbol": symbol,
            "ret_7d": ret_7d,
            "ret_3d": ret_3d,
            "ret_1d": ret_1d,
            "_ra_7d": ra_7d,
            "_ra_3d": ra_3d,
            "_ra_1d": ra_1d,
        })

    if not records:
        return pd.DataFrame(columns=[
            "symbol", "ret_7d", "ret_3d", "ret_1d",
            "z_7d", "z_3d", "z_1d", "composite", "rs_rank",
        ])

    df = pd.DataFrame(records)

    # --- 横截面 Z-Score ---
    if len(df) <= 1:
        df["z_7d"] = 0.0
        df["z_3d"] = 0.0
        df["z_1d"] = 0.0
    else:
        df["z_7d"] = np.clip(scipy_zscore(df["_ra_7d"], ddof=1), -3, 3)
        df["z_3d"] = np.clip(scipy_zscore(df["_ra_3d"], ddof=1), -3, 3)
        df["z_1d"] = np.clip(scipy_zscore(df["_ra_1d"], ddof=1), -3, 3)

    # --- 加权合成 ---
    df["composite"] = (
        0.40 * df["z_7d"]
        + 0.35 * df["z_3d"]
        + 0.25 * df["z_1d"]
    )

    # --- 百分位排名 0-99 ---
    if len(df) <= 1:
        df["rs_rank"] = 50
    else:
        pct = rankdata(df["composite"], method="average") / len(df)
        df["rs_rank"] = np.clip(np.floor(pct * 100).astype(int), 0, 99)

    df = df[[
        "symbol", "ret_7d", "ret_3d", "ret_1d",
        "z_7d", "z_3d", "z_1d", "composite", "rs_rank",
    ]].reset_index(drop=True)

    return df


def _clenow_momentum_fast(prices: np.ndarray, window: int) -> float:
    """Clenow 动量 (币圈版), window < 2 返回 0.0"""
    if window < 2 or len(prices) < window:
        return 0.0

    tail = prices[-window:]
    if np.any(tail <= 0):
        return 0.0

    log_prices = np.log(tail)
    x = np.arange(window)

    try:
        slope, _, r_value, _, _ = linregress(x, log_prices)
    except Exception:
        return 0.0

    r_squared = r_value ** 2
    annualized = (np.exp(slope) ** 365) - 1
    annualized = np.clip(annualized, -10, 100)
    return annualized * r_squared


def compute_fast_rs_c(price_dict: dict) -> pd.DataFrame:
    """
    Method C — Clenow 回归动量 (7d/3d/1d)

    注: 1d 窗口不足以做回归 (需 ≥2 点), 该分量为 0, 不影响排名
    """
    cfg = FAST_CONFIG
    w_long = cfg["window_long"]
    w_mid = cfg["window_mid"]
    w_short = cfg["window_short"]
    min_days = cfg["min_data_days"]

    records = []

    for symbol, close in price_dict.items():
        if len(close) < min_days:
            continue

        c7 = _clenow_momentum_fast(close, w_long)
        c3 = _clenow_momentum_fast(close, w_mid)
        c1 = _clenow_momentum_fast(close, w_short)  # window=1 → 0.0

        records.append({
            "symbol": symbol,
            "clenow_7d": c7,
            "clenow_3d": c3,
            "clenow_1d": c1,
        })

    if not records:
        return pd.DataFrame(columns=[
            "symbol", "clenow_7d", "clenow_3d", "clenow_1d",
            "composite", "rs_rank",
        ])

    df = pd.DataFrame(records)

    # --- 加权合成 (c1d 为 0 不影响排名) ---
    df["composite"] = (
        0.50 * df["clenow_7d"]
        + 0.30 * df["clenow_3d"]
        + 0.20 * df["clenow_1d"]
    )

    # --- 百分位排名 0-99 ---
    if len(df) <= 1:
        df["rs_rank"] = 50
    else:
        pct = rankdata(df["composite"], method="average") / len(df)
        df["rs_rank"] = np.clip(np.floor(pct * 100).astype(int), 0, 99)

    df = df[[
        "symbol", "clenow_7d", "clenow_3d", "clenow_1d",
        "composite", "rs_rank",
    ]].reset_index(drop=True)

    return df


# ============================================================
# Telegram 推送
# ============================================================

def format_rs_fast_message(df_b: pd.DataFrame, df_c: pd.DataFrame,
                           total_symbols: int) -> list:
    """格式化短周期 RS 排名推送消息"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    top_n = FAST_CONFIG["top_n"]

    messages = []

    # --- Method B ---
    top_b = df_b.nlargest(top_n, "rs_rank")

    msg_b = f"*RS 短期动量排名 (7D/3D/1D)*\n"
    msg_b += f"时间: {now}\n\n"
    msg_b += f"📊 *Method B — 风险调整动量 Top {top_n}*\n"

    for i, row in enumerate(top_b.itertuples(), 1):
        sym = row.symbol.replace("USDT", "")
        rs = int(row.rs_rank)
        r7d = row.ret_7d * 100
        r3d = row.ret_3d * 100
        r1d = row.ret_1d * 100
        msg_b += (
            f"{i:>2}. {sym:<10} RS:{rs:>2}  "
            f"7D:{r7d:>+6.1f}%  3D:{r3d:>+5.1f}%  1D:{r1d:>+5.1f}%\n"
        )

    messages.append(msg_b)

    # --- Method C ---
    top_c = df_c.nlargest(top_n, "rs_rank")

    msg_c = f"📈 *Method C — Clenow 趋势动量 Top {top_n}*\n"

    for i, row in enumerate(top_c.itertuples(), 1):
        sym = row.symbol.replace("USDT", "")
        rs = int(row.rs_rank)
        c7 = row.clenow_7d
        c3 = row.clenow_3d
        msg_c += (
            f"{i:>2}. {sym:<10} RS:{rs:>2}  "
            f"C7:{c7:>6.2f}  C3:{c3:>5.2f}\n"
        )

    msg_c += (
        f"\n筛选: 7天成交量 Top {total_symbols} | "
        f"全部USDT合约"
    )

    messages.append(msg_c)

    return messages


# ============================================================
# 主流程
# ============================================================

def scan_rs_fast() -> tuple:
    """扫描所有币种的短周期 RS 动量排名"""
    print("=" * 50)
    print(f"[开始] RS 短期动量扫描 (7D/3D/1D) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    symbols = get_all_usdt_futures()
    if symbols is None or len(symbols) == 0:
        print("[错误] 无法获取交易对列表")
        return None, None, 0

    all_data = {}
    skipped = 0
    total = len(symbols)

    for idx, symbol in enumerate(symbols, 1):
        if idx % 50 == 0 or idx == 1:
            print(f"[{idx}/{total}] 加载数据中...")

        df = update_daily_cache(symbol)

        if df.empty or len(df) < FAST_CONFIG["min_data_days"]:
            skipped += 1
            continue

        all_data[symbol] = df

        if idx % 20 == 0:
            time.sleep(1)

    print(f"\n[数据] 有效标的: {len(all_data)}, 跳过: {skipped}")

    if len(all_data) < 10:
        print("[错误] 有效标的太少")
        return None, None, 0

    # 按7天总成交量筛选 Top 100
    vol_days = FAST_CONFIG["volume_filter_days"]
    vol_top_n = FAST_CONFIG["volume_top_n"]
    volume_ranks = {}
    for symbol, df in all_data.items():
        tail = df.tail(vol_days)
        volume_ranks[symbol] = tail["quote_volume"].sum() if "quote_volume" in tail.columns else 0

    sorted_by_vol = sorted(volume_ranks.items(), key=lambda x: x[1], reverse=True)
    top_symbols = {s for s, _ in sorted_by_vol[:vol_top_n]}
    print(f"[筛选] {vol_days}天成交量 Top {vol_top_n} (从 {len(all_data)} 只中筛选)")

    price_dict = {s: df["close"].values for s, df in all_data.items() if s in top_symbols}

    print("[计算] Method B — 风险调整 Z-Score (7D/3D/1D)...")
    df_b = compute_fast_rs_b(price_dict)
    print(f"  计算完成: {len(df_b)} 只标的")

    print("[计算] Method C — Clenow 回归动量 (7D/3D)...")
    df_c = compute_fast_rs_c(price_dict)
    print(f"  计算完成: {len(df_c)} 只标的")

    return df_b, df_c, len(price_dict)


def main():
    """主函数"""
    try:
        df_b, df_c, total = scan_rs_fast()

        print("\n" + "=" * 50)
        print("[结果]")
        print("=" * 50)

        if df_b is None or df_c is None:
            print("[错误] 扫描失败")
            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            msg = (
                f"*RS 短期动量扫描失败*\n"
                f"时间: {now}\n\n"
                f"⚠️ 数据获取失败，请检查网络连接"
            )
            send_telegram_alert(msg)
            return

        # 打印 Top 10
        print("\n=== Method B: 风险调整 Z-Score (7D/3D/1D) Top 10 ===")
        top_b = df_b.nlargest(10, "rs_rank")
        for _, row in top_b.iterrows():
            print(
                f"  {row['symbol']:<14} RS:{int(row['rs_rank']):>2}  "
                f"7D:{row['ret_7d']*100:>+6.1f}%  "
                f"3D:{row['ret_3d']*100:>+5.1f}%  "
                f"1D:{row['ret_1d']*100:>+5.1f}%"
            )

        print("\n=== Method C: Clenow 趋势动量 (7D/3D) Top 10 ===")
        top_c = df_c.nlargest(10, "rs_rank")
        for _, row in top_c.iterrows():
            print(
                f"  {row['symbol']:<14} RS:{int(row['rs_rank']):>2}  "
                f"C7:{row['clenow_7d']:>6.2f}  "
                f"C3:{row['clenow_3d']:>5.2f}"
            )

        # Telegram 推送
        messages = format_rs_fast_message(df_b, df_c, total)
        for i, msg in enumerate(messages):
            send_telegram_alert(msg)
            if i < len(messages) - 1:
                time.sleep(1)

        print(f"\n[完成] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        print(f"[致命错误] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
