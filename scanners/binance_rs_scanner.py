#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
币安合约 RS 动量扫描器 (Relative Strength Rating)

两种计算方法，移植自 Finance/src/indicators/rs_rating.py，参数适配币圈日线:

Method B — Risk-Adjusted Z-Score
    多周期收益率 (30d/14d/7d)，经波动率调整后做横截面 Z-Score，
    加权合成后转换为 0-99 百分位排名。
    跳过最近 1 天以规避短期反转效应。

Method C — Clenow Exponential Regression
    对数价格线性回归，斜率年化后乘以 R²，
    三窗口 (30d/14d/7d) 加权合成后转换为 0-99 百分位排名。

参数对比 (股票 → 币圈):
    长窗口:  63 交易日 → 30 天
    中窗口:  21 交易日 → 14 天
    短窗口:  10 交易日 → 7 天
    跳过:    5 交易日  → 1 天
    年化:    √252      → √365
    最小数据: 70 天     → 35 天

Author: Claude Code
Date: 2026-02-15
"""

import os
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from scipy.stats import rankdata, zscore as scipy_zscore, linregress

# ============================================================
# 路径配置
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# ============================================================
# 配置
# ============================================================
CONFIG = {
    # 币安合约API
    "base_url": "https://fapi.binance.com",

    # RS 参数 (币圈适配)
    "window_long": 30,    # 1个月
    "window_mid": 14,     # 2周
    "window_short": 7,    # 1周
    "skip_days": 1,       # 跳过最近1天
    "annual_factor": 365,  # 年化因子 (加密市场全年交易)
    "min_data_days": 35,  # 最小数据要求

    # 排名展示
    "top_n": 20,

    # 数据缓存 (复用 EMA120 缓存目录)
    "cache_dir": str(PROJECT_ROOT / "cache" / "binance_daily_cache_full"),
    "history_days": 365,

    # Telegram配置
    "telegram_bot_token": os.environ.get("TELEGRAM_BOT_TOKEN", ""),
    "telegram_chat_id": os.environ.get("TELEGRAM_CHAT_ID", ""),
    "proxy": "",
}


# ============================================================
# 币安API封装（复用自 EMA120 scanner）
# ============================================================
def api_request_with_retry(url: str, params: dict = None,
                           max_retries: int = 3, timeout: int = 30) -> dict:
    """带重试机制的API请求"""
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            print(f"[API] 第{attempt}次请求超时...")
        except requests.exceptions.RequestException as e:
            print(f"[API] 第{attempt}次请求失败: {e}")

        if attempt < max_retries:
            wait_time = attempt * 2
            print(f"[API] {wait_time}秒后重试...")
            time.sleep(wait_time)

    return None


def get_all_usdt_futures() -> list:
    """获取所有USDT永续合约交易对"""
    url = f"{CONFIG['base_url']}/fapi/v1/exchangeInfo"
    data = api_request_with_retry(url)

    if data is None:
        print("[错误] 获取交易对列表失败（已重试3次）")
        return None

    symbols = []
    for s in data.get("symbols", []):
        if (s.get("quoteAsset") == "USDT" and
            s.get("contractType") == "PERPETUAL" and
            s.get("status") == "TRADING"):
            symbols.append(s["symbol"])

    print(f"[API] 获取到 {len(symbols)} 个USDT永续合约")
    return symbols


def fetch_klines_by_time(symbol: str, start_time: int, end_time: int = None,
                         interval: str = "1d") -> list:
    """按时间范围获取K线数据"""
    url = f"{CONFIG['base_url']}/fapi/v1/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "limit": 1500
    }
    if end_time:
        params["endTime"] = end_time

    data = api_request_with_retry(url, params=params, max_retries=2, timeout=30)
    if data is None:
        return []
    return data


# ============================================================
# 数据缓存模块（复用自 EMA120 scanner）
# ============================================================
def ensure_cache_dir():
    """确保缓存目录存在"""
    cache_dir = Path(CONFIG["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cache_path(symbol: str) -> Path:
    """获取某个币种的缓存文件路径"""
    cache_dir = ensure_cache_dir()
    return cache_dir / f"{symbol}.csv"


def load_cached_data(symbol: str) -> pd.DataFrame:
    """加载缓存的历史数据"""
    cache_path = get_cache_path(symbol)
    if not cache_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(cache_path, parse_dates=["timestamp"])
        return df
    except Exception as e:
        print(f"[警告] 读取{symbol}缓存失败: {e}")
        return pd.DataFrame()


def save_cached_data(symbol: str, df: pd.DataFrame):
    """保存数据到缓存"""
    cache_path = get_cache_path(symbol)
    df.to_csv(cache_path, index=False)


def klines_to_dataframe(klines: list) -> pd.DataFrame:
    """将K线数据转换为DataFrame"""
    if not klines:
        return pd.DataFrame()

    records = []
    for k in klines:
        records.append({
            "timestamp": pd.to_datetime(k[0], unit="ms"),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
            "quote_volume": float(k[7]),
        })
    return pd.DataFrame(records)


def filter_closed_klines(df: pd.DataFrame) -> pd.DataFrame:
    """过滤掉正在进行中的日线，只保留已收盘的"""
    if df.empty:
        return df
    today_utc = pd.Timestamp(
        datetime.now(timezone.utc).replace(tzinfo=None)
    ).normalize()
    return df[df["timestamp"] < today_utc]


def update_daily_cache(symbol: str) -> pd.DataFrame:
    """更新某个币种的日线数据缓存"""
    cache_df = load_cached_data(symbol)

    if cache_df.empty:
        start_time = int(
            (datetime.now(timezone.utc) - timedelta(days=CONFIG['history_days']))
            .timestamp() * 1000
        )
        klines = fetch_klines_by_time(symbol, start_time)
        if not klines:
            return pd.DataFrame()

        df = klines_to_dataframe(klines)
        df = filter_closed_klines(df)
        save_cached_data(symbol, df)
        return df
    else:
        today = pd.Timestamp(
            datetime.now(timezone.utc).replace(tzinfo=None)
        ).normalize()
        refresh_start = today - timedelta(days=3)
        cache_df = cache_df[cache_df["timestamp"] < refresh_start]

        start_time = int(refresh_start.timestamp() * 1000)
        klines = fetch_klines_by_time(symbol, start_time)

        if klines:
            new_df = klines_to_dataframe(klines)
            new_df = filter_closed_klines(new_df)

            df = pd.concat([cache_df, new_df], ignore_index=True)
            df = df.drop_duplicates(subset=["timestamp"], keep="last")
            df = df.sort_values("timestamp").reset_index(drop=True)

            cutoff = pd.Timestamp(
                datetime.now(timezone.utc).replace(tzinfo=None)
            ) - timedelta(days=CONFIG['history_days'])
            df = df[df["timestamp"] >= cutoff]

            save_cached_data(symbol, df)
            return df

        return cache_df


# ============================================================
# RS Rating 计算核心
# ============================================================

def compute_crypto_rs_b(price_dict: dict) -> pd.DataFrame:
    """
    Method B — 风险调整 Z-Score 横截面动量排名 (币圈版)

    三窗口: 30d / 14d / 7d, 跳过最近 1 天
    30d 和 14d 经年化波动率调整, 7d 不调整
    横截面 Z-Score (winsorize ±3), 加权合成, 百分位排名 0-99

    Args:
        price_dict: {symbol: close_array} — close 已按时间正序

    Returns:
        DataFrame [symbol, ret_1m, ret_2w, ret_1w, z_1m, z_2w, z_1w,
                   composite, rs_rank]
    """
    min_days = CONFIG["min_data_days"]
    skip = CONFIG["skip_days"]       # 1
    w_long = CONFIG["window_long"]   # 30
    w_mid = CONFIG["window_mid"]     # 14
    w_short = CONFIG["window_short"] # 7
    ann = np.sqrt(CONFIG["annual_factor"])  # √365

    records = []

    for symbol, close in price_dict.items():
        n = len(close)
        if n < min_days:
            continue

        # 需要: n - 1 - skip 做终点, n - 1 - skip - w_long 做起点
        end_idx = n - 1 - skip  # 跳过最近 skip 天
        if end_idx < 0 or (end_idx - w_long) < 0:
            continue

        # --- 收益率 ---
        ret_1m = close[end_idx] / close[end_idx - w_long] - 1
        ret_2w = close[end_idx] / close[end_idx - w_mid] - 1
        ret_1w = close[end_idx] / close[end_idx - w_short] - 1

        # --- 风险调整 (年化波动率) ---
        daily_returns = np.diff(close) / close[:-1]

        # 30d 波动率
        vol_start = max(0, end_idx - w_long)
        vol_1m = np.std(daily_returns[vol_start:end_idx], ddof=1) * ann
        # 14d 波动率
        vol_start_2w = max(0, end_idx - w_mid)
        vol_2w = np.std(daily_returns[vol_start_2w:end_idx], ddof=1) * ann

        ra_1m = ret_1m / vol_1m if vol_1m > 1e-10 else 0.0
        ra_2w = ret_2w / vol_2w if vol_2w > 1e-10 else 0.0
        ra_1w = ret_1w  # 7d 太短不做风险调整

        records.append({
            "symbol": symbol,
            "ret_1m": ret_1m,
            "ret_2w": ret_2w,
            "ret_1w": ret_1w,
            "_ra_1m": ra_1m,
            "_ra_2w": ra_2w,
            "_ra_1w": ra_1w,
        })

    if not records:
        return pd.DataFrame(columns=[
            "symbol", "ret_1m", "ret_2w", "ret_1w",
            "z_1m", "z_2w", "z_1w", "composite", "rs_rank",
        ])

    df = pd.DataFrame(records)

    # --- 横截面 Z-Score ---
    if len(df) <= 1:
        df["z_1m"] = 0.0
        df["z_2w"] = 0.0
        df["z_1w"] = 0.0
    else:
        df["z_1m"] = np.clip(scipy_zscore(df["_ra_1m"], ddof=1), -3, 3)
        df["z_2w"] = np.clip(scipy_zscore(df["_ra_2w"], ddof=1), -3, 3)
        df["z_1w"] = np.clip(scipy_zscore(df["_ra_1w"], ddof=1), -3, 3)

    # --- 加权合成 ---
    df["composite"] = (
        0.40 * df["z_1m"]
        + 0.35 * df["z_2w"]
        + 0.25 * df["z_1w"]
    )

    # --- 百分位排名 0-99 ---
    if len(df) <= 1:
        df["rs_rank"] = 50
    else:
        pct = rankdata(df["composite"], method="average") / len(df)
        df["rs_rank"] = np.clip(np.floor(pct * 100).astype(int), 0, 99)

    df = df[[
        "symbol", "ret_1m", "ret_2w", "ret_1w",
        "z_1m", "z_2w", "z_1w", "composite", "rs_rank",
    ]].reset_index(drop=True)

    return df


def _clenow_momentum_crypto(prices: np.ndarray, window: int) -> float:
    """
    计算单个窗口的 Clenow 动量分数 (币圈版)

    对最近 window 根日线的对数价格做线性回归，
    斜率年化 (exp(slope)^365 - 1) 后乘以 R²。

    Args:
        prices: 收盘价数组 (至少 window 个点)
        window: 回看窗口

    Returns:
        Clenow score = annualized_return * R²
    """
    if len(prices) < window:
        return 0.0

    tail = prices[-window:]
    if np.any(tail <= 0):
        return 0.0

    log_prices = np.log(tail)
    x = np.arange(window)

    try:
        slope, _intercept, r_value, _p, _se = linregress(x, log_prices)
    except Exception:
        return 0.0

    r_squared = r_value ** 2
    annualized = (np.exp(slope) ** 365) - 1
    # 币圈极端短期涨幅会导致年化溢出 (7天涨200% → 年化10^24)
    # clip 到 [-10, 100] 即 -1000% ~ +10000% 年化，保留排序信息
    annualized = np.clip(annualized, -10, 100)
    return annualized * r_squared


def compute_crypto_rs_c(price_dict: dict) -> pd.DataFrame:
    """
    Method C — Clenow 回归动量排名 (币圈版)

    三窗口: 30d / 14d / 7d
    加权: 0.50 / 0.30 / 0.20
    百分位排名 0-99

    Args:
        price_dict: {symbol: close_array}

    Returns:
        DataFrame [symbol, clenow_30d, clenow_14d, clenow_7d,
                   composite, rs_rank]
    """
    min_days = CONFIG["min_data_days"]
    w_long = CONFIG["window_long"]
    w_mid = CONFIG["window_mid"]
    w_short = CONFIG["window_short"]

    records = []

    for symbol, close in price_dict.items():
        if len(close) < min_days:
            continue

        c30 = _clenow_momentum_crypto(close, w_long)
        c14 = _clenow_momentum_crypto(close, w_mid)
        c7 = _clenow_momentum_crypto(close, w_short)

        records.append({
            "symbol": symbol,
            "clenow_30d": c30,
            "clenow_14d": c14,
            "clenow_7d": c7,
        })

    if not records:
        return pd.DataFrame(columns=[
            "symbol", "clenow_30d", "clenow_14d", "clenow_7d",
            "composite", "rs_rank",
        ])

    df = pd.DataFrame(records)

    # --- 加权合成 ---
    df["composite"] = (
        0.50 * df["clenow_30d"]
        + 0.30 * df["clenow_14d"]
        + 0.20 * df["clenow_7d"]
    )

    # --- 百分位排名 0-99 ---
    if len(df) <= 1:
        df["rs_rank"] = 50
    else:
        pct = rankdata(df["composite"], method="average") / len(df)
        df["rs_rank"] = np.clip(np.floor(pct * 100).astype(int), 0, 99)

    df = df[[
        "symbol", "clenow_30d", "clenow_14d", "clenow_7d",
        "composite", "rs_rank",
    ]].reset_index(drop=True)

    return df


# ============================================================
# Telegram 推送
# ============================================================
def get_proxies():
    """获取代理配置"""
    proxy = CONFIG.get("proxy", "")
    if proxy:
        return {"http": proxy, "https": proxy}
    return None


def send_telegram_alert(message: str, max_retries: int = 3) -> bool:
    """发送Telegram消息"""
    token = CONFIG["telegram_bot_token"]
    chat_id = CONFIG["telegram_chat_id"]

    if not token or not chat_id:
        print("[警告] Telegram未配置，跳过发送")
        print("[消息内容]\n" + message)
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    proxies = get_proxies()

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                url, json=payload, timeout=15, proxies=proxies
            )
            response.raise_for_status()
            print("[Telegram] 消息已发送")
            return True
        except Exception as e:
            print(f"[Telegram] 第{attempt}次发送失败: {e}")
            if attempt < max_retries:
                wait_time = attempt * 2
                print(f"[Telegram] {wait_time}秒后重试...")
                time.sleep(wait_time)

    print(f"[Telegram] 发送失败，已重试{max_retries}次")
    return False


def format_rs_message(df_b: pd.DataFrame, df_c: pd.DataFrame,
                      total_symbols: int) -> list:
    """
    格式化 RS 排名推送消息

    Returns:
        消息列表 (可能需要分条发送)
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    top_n = CONFIG["top_n"]

    messages = []

    # --- Method B ---
    top_b = df_b.nlargest(top_n, "rs_rank")

    msg_b = f"*RS 动量排名 (日线)*\n"
    msg_b += f"时间: {now}\n\n"
    msg_b += f"📊 *Method B — 风险调整动量 Top {top_n}*\n"

    for i, row in enumerate(top_b.itertuples(), 1):
        sym = row.symbol.replace("USDT", "")
        rs = int(row.rs_rank)
        r1m = row.ret_1m * 100
        r2w = row.ret_2w * 100
        # 对齐格式
        msg_b += (
            f"{i:>2}. {sym:<10} RS:{rs:>2}  "
            f"1M:{r1m:>+6.1f}%  2W:{r2w:>+5.1f}%\n"
        )

    messages.append(msg_b)

    # --- Method C ---
    top_c = df_c.nlargest(top_n, "rs_rank")

    msg_c = f"📈 *Method C — Clenow 趋势动量 Top {top_n}*\n"

    for i, row in enumerate(top_c.itertuples(), 1):
        sym = row.symbol.replace("USDT", "")
        rs = int(row.rs_rank)
        c30 = row.clenow_30d
        c14 = row.clenow_14d
        msg_c += (
            f"{i:>2}. {sym:<10} RS:{rs:>2}  "
            f"C30:{c30:>5.2f}  C14:{c14:>5.2f}\n"
        )

    msg_c += f"\n扫描范围: 全部USDT合约 ({total_symbols}个)"

    messages.append(msg_c)

    return messages


# ============================================================
# 主流程
# ============================================================
def scan_rs_rating() -> tuple:
    """
    扫描所有币种的 RS 动量排名

    Returns:
        (df_b, df_c, total_symbols) 或 (None, None, 0) 失败时
    """
    print("=" * 50)
    print(f"[开始] RS 动量扫描 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    # 1. 获取所有USDT永续合约
    symbols = get_all_usdt_futures()
    if symbols is None or len(symbols) == 0:
        print("[错误] 无法获取交易对列表")
        return None, None, 0

    # 2. 加载/更新所有价格数据
    price_dict = {}  # {symbol: np.ndarray of close prices}
    skipped = 0
    total = len(symbols)

    for idx, symbol in enumerate(symbols, 1):
        if idx % 50 == 0 or idx == 1:
            print(f"[{idx}/{total}] 加载数据中...")

        df = update_daily_cache(symbol)

        if df.empty or len(df) < CONFIG["min_data_days"]:
            skipped += 1
            continue

        price_dict[symbol] = df["close"].values

        # 限流：每20个请求休息1秒
        if idx % 20 == 0:
            time.sleep(1)

    print(f"\n[数据] 有效标的: {len(price_dict)}, 跳过: {skipped}")

    if len(price_dict) < 10:
        print("[错误] 有效标的太少，无法计算有意义的排名")
        return None, None, 0

    # 3. 计算 RS Rating
    print("[计算] Method B — 风险调整 Z-Score...")
    df_b = compute_crypto_rs_b(price_dict)
    print(f"  计算完成: {len(df_b)} 只标的")

    print("[计算] Method C — Clenow 回归动量...")
    df_c = compute_crypto_rs_c(price_dict)
    print(f"  计算完成: {len(df_c)} 只标的")

    return df_b, df_c, len(price_dict)


def main():
    """主函数"""
    try:
        df_b, df_c, total = scan_rs_rating()

        print("\n" + "=" * 50)
        print("[结果]")
        print("=" * 50)

        if df_b is None or df_c is None:
            print("[错误] 扫描失败")
            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            msg = (
                f"*RS 动量扫描失败*\n"
                f"时间: {now}\n\n"
                f"⚠️ 数据获取失败，请检查网络连接"
            )
            send_telegram_alert(msg)
            return

        # 打印 Top 10
        print("\n=== Method B: 风险调整 Z-Score Top 10 ===")
        top_b = df_b.nlargest(10, "rs_rank")
        for _, row in top_b.iterrows():
            print(
                f"  {row['symbol']:<14} RS:{int(row['rs_rank']):>2}  "
                f"1M:{row['ret_1m']*100:>+6.1f}%  "
                f"2W:{row['ret_2w']*100:>+5.1f}%  "
                f"1W:{row['ret_1w']*100:>+5.1f}%"
            )

        print("\n=== Method C: Clenow 趋势动量 Top 10 ===")
        top_c = df_c.nlargest(10, "rs_rank")
        for _, row in top_c.iterrows():
            print(
                f"  {row['symbol']:<14} RS:{int(row['rs_rank']):>2}  "
                f"C30:{row['clenow_30d']:>6.2f}  "
                f"C14:{row['clenow_14d']:>5.2f}  "
                f"C7:{row['clenow_7d']:>5.2f}"
            )

        # Telegram 推送
        messages = format_rs_message(df_b, df_c, total)
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
