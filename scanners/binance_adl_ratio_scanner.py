#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
币安合约 涨跌比率指标 (ADL Ratio) 扫描器
每天早上8点扫描全部USDT合约，计算涨跌比率及其分位数

涨跌比率 = 前一日涨幅>5%的交易对数 / 前一日跌幅>5%的交易对数
分位数 = 该比率在过去120天内的百分位排名

Author: Claude Code
Date: 2026-01-04
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ============================================================
# 路径配置 (相对路径，支持文件夹移动)
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent  # scanners/
PROJECT_ROOT = SCRIPT_DIR.parent              # 项目根目录

# ============================================================
# 配置
# ============================================================
CONFIG = {
    # 币安合约API（无需代理）
    "base_url": "https://fapi.binance.com",

    # 涨跌比率参数
    "rise_threshold": 5.0,    # 涨幅阈值 (%)
    "fall_threshold": -5.0,   # 跌幅阈值 (%)
    "lookback_days": 120,     # 分位数计算回看天数

    # 数据缓存 (相对路径)
    "cache_dir": str(PROJECT_ROOT / "cache" / "binance_daily_cache"),
    "ratio_history_file": str(PROJECT_ROOT / "results" / "adl_ratio_history.csv"),
    "history_days": 365,

    # Telegram配置 (从环境变量读取)
    "telegram_bot_token": os.environ.get("TELEGRAM_BOT_TOKEN", ""),
    "telegram_chat_id": os.environ.get("TELEGRAM_CHAT_ID", ""),
    "proxy": "",  # 仅Telegram需要
}

# ============================================================
# 币安API封装（带重试机制）
# ============================================================
def api_request_with_retry(url: str, params: dict = None, max_retries: int = 3, timeout: int = 30) -> dict:
    """
    带重试机制的API请求

    Args:
        url: 请求URL
        params: 请求参数
        max_retries: 最大重试次数
        timeout: 超时时间（秒）

    Returns:
        响应数据，失败返回None
    """
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
        print(f"[错误] 获取交易对列表失败（已重试3次）")
        return None  # 返回None表示API失败

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
    """按时间范围获取K线数据（带重试）"""
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
        print(f"[错误] 获取{symbol}K线失败（已重试）")
        return []

    return data


# ============================================================
# 数据缓存模块
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
            "quote_volume": float(k[7]),  # USDT成交量
        })

    df = pd.DataFrame(records)
    return df


def update_daily_cache(symbol: str) -> pd.DataFrame:
    """更新某个币种的日线数据缓存"""
    cache_df = load_cached_data(symbol)
    today_utc = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None)).normalize()

    if cache_df.empty:
        # 无缓存，下载完整历史
        start_time = int((datetime.now(timezone.utc) - timedelta(days=CONFIG['history_days'])).timestamp() * 1000)
        klines = fetch_klines_by_time(symbol, start_time)

        if not klines:
            return pd.DataFrame()

        df = klines_to_dataframe(klines)
        # 排除当天未收盘的K线
        df = df[df["timestamp"] < today_utc]
        save_cached_data(symbol, df)
        return df

    else:
        # 有缓存，需要更新最近几天的数据以确保收盘价准确
        # 删除最近3天的数据，重新获取（防止之前缓存的是未收盘价格）
        cutoff_recent = today_utc - timedelta(days=3)
        cache_df = cache_df[cache_df["timestamp"] < cutoff_recent]

        # 从3天前开始重新获取
        start_time = int(cutoff_recent.timestamp() * 1000)
        klines = fetch_klines_by_time(symbol, start_time)

        if klines:
            new_df = klines_to_dataframe(klines)
            # 排除当天未收盘的K线
            new_df = new_df[new_df["timestamp"] < today_utc]

            if not cache_df.empty:
                df = pd.concat([cache_df, new_df], ignore_index=True)
            else:
                df = new_df

            df = df.drop_duplicates(subset=["timestamp"], keep="last")
            df = df.sort_values("timestamp").reset_index(drop=True)

            # 只保留最近history_days天的数据
            cutoff_old = today_utc - timedelta(days=CONFIG['history_days'])
            df = df[df["timestamp"] >= cutoff_old]

            save_cached_data(symbol, df)
            return df

        return cache_df


# ============================================================
# 涨跌比率历史数据管理
# ============================================================
def load_ratio_history() -> pd.DataFrame:
    """加载涨跌比率历史数据"""
    history_file = Path(CONFIG["ratio_history_file"])

    if not history_file.exists():
        return pd.DataFrame(columns=["date", "rise_count", "fall_count", "ratio"])

    try:
        df = pd.read_csv(history_file, parse_dates=["date"])
        return df
    except Exception as e:
        print(f"[警告] 读取涨跌比率历史失败: {e}")
        return pd.DataFrame(columns=["date", "rise_count", "fall_count", "ratio"])


def save_ratio_history(df: pd.DataFrame):
    """保存涨跌比率历史数据"""
    history_file = Path(CONFIG["ratio_history_file"])
    history_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(history_file, index=False)


def add_ratio_record(date: datetime, rise_count: int, fall_count: int, ratio: float):
    """添加新的涨跌比率记录"""
    history_df = load_ratio_history()

    # 转换日期格式
    date_normalized = pd.Timestamp(date).normalize()

    # 检查是否已存在该日期的记录
    if not history_df.empty:
        history_df["date"] = pd.to_datetime(history_df["date"])
        existing = history_df[history_df["date"] == date_normalized]
        if not existing.empty:
            # 更新已有记录
            history_df.loc[history_df["date"] == date_normalized, ["rise_count", "fall_count", "ratio"]] = [rise_count, fall_count, ratio]
        else:
            # 添加新记录
            new_row = pd.DataFrame([{
                "date": date_normalized,
                "rise_count": rise_count,
                "fall_count": fall_count,
                "ratio": ratio
            }])
            history_df = pd.concat([history_df, new_row], ignore_index=True)
    else:
        history_df = pd.DataFrame([{
            "date": date_normalized,
            "rise_count": rise_count,
            "fall_count": fall_count,
            "ratio": ratio
        }])

    # 按日期排序并只保留最近365天
    history_df = history_df.sort_values("date").reset_index(drop=True)
    cutoff = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None)) - timedelta(days=365)
    history_df = history_df[history_df["date"] >= cutoff]

    save_ratio_history(history_df)
    return history_df


# ============================================================
# 涨跌比率计算模块
# ============================================================
def calculate_daily_return(df: pd.DataFrame) -> float:
    """
    计算最近一个完整交易日的收益率

    币安日线UTC 00:00收盘 = 北京时间08:00
    如果在08:28运行，最后一根K线是当天的（未收盘），需要排除

    Args:
        df: 包含日线数据的DataFrame

    Returns:
        最近完整交易日的收益率 (%)，如果数据不足返回None
    """
    if len(df) < 2:
        return None

    # 获取当前UTC日期（币安日线以UTC为准）
    today_utc = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None)).normalize()

    # 排除当天未收盘的K线
    df_closed = df[df["timestamp"] < today_utc]

    if len(df_closed) < 2:
        return None

    # 取最后两个完整交易日的收盘价
    # iloc[-1] = 最近完整交易日 (如1月4日)
    # iloc[-2] = 前一天 (如1月3日)
    prev_close = df_closed["close"].iloc[-2]
    curr_close = df_closed["close"].iloc[-1]

    if prev_close == 0:
        return None

    return (curr_close - prev_close) / prev_close * 100


def calculate_percentile(current_value: float, historical_values: pd.Series) -> float:
    """
    计算当前值在历史数据中的分位数

    如果90%的历史数据比当前值低，则返回90

    Args:
        current_value: 当前值
        historical_values: 历史值序列（不包含当前值）

    Returns:
        分位数 (0-100)
    """
    if len(historical_values) == 0:
        return 50.0  # 无历史数据时返回50%

    # 计算有多少历史数据小于当前值
    count_below = (historical_values < current_value).sum()
    percentile = count_below / len(historical_values) * 100

    return percentile


# ============================================================
# Telegram推送模块
# ============================================================
def get_proxies():
    """获取代理配置"""
    proxy = CONFIG.get("proxy", "")
    if proxy:
        return {"http": proxy, "https": proxy}
    return None


def send_telegram_alert(message: str, max_retries: int = 3) -> bool:
    """发送Telegram消息（带重试机制）"""
    token = CONFIG["telegram_bot_token"]
    chat_id = CONFIG["telegram_chat_id"]

    if not token or not chat_id:
        print(f"[警告] Telegram未配置，跳过发送")
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
            response = requests.post(url, json=payload, timeout=15, proxies=proxies)
            response.raise_for_status()
            print(f"[Telegram] 消息已发送")
            return True
        except Exception as e:
            print(f"[Telegram] 第{attempt}次发送失败: {e}")
            if attempt < max_retries:
                wait_time = attempt * 2
                print(f"[Telegram] {wait_time}秒后重试...")
                time.sleep(wait_time)

    print(f"[Telegram] 发送失败，已重试{max_retries}次")
    return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }

    try:
        proxies = get_proxies()
        response = requests.post(url, json=payload, timeout=15, proxies=proxies)
        response.raise_for_status()
        print(f"[Telegram] 消息已发送")
        return True
    except Exception as e:
        print(f"[Telegram] 发送失败: {e}")
        return False


def get_market_sentiment(percentile: float) -> str:
    """根据分位数判断市场情绪"""
    if percentile >= 90:
        return "极度乐观 (可能过热)"
    elif percentile >= 70:
        return "偏强"
    elif percentile >= 30:
        return "中性"
    elif percentile >= 10:
        return "偏弱"
    else:
        return "极度悲观 (可能超跌)"


def format_signal_message(data_date, rise_count: int, fall_count: int, ratio: float,
                          percentile: float, total_symbols: int, valid_symbols: int) -> str:
    """格式化涨跌比率消息"""
    data_date_str = data_date.strftime("%Y-%m-%d")
    sentiment = get_market_sentiment(percentile)

    msg = f"*涨跌比率指标 (ADL Ratio)*\n"
    msg += f"数据日期: {data_date_str} (UTC)\n\n"

    msg += f"*当日统计:*\n"
    msg += f"• 涨幅>5%: {rise_count} 个交易对\n"
    msg += f"• 跌幅>5%: {fall_count} 个交易对\n"
    msg += f"• 涨跌比率: {ratio:.2f}\n\n"

    msg += f"*分位数排名: {percentile:.1f}%* (过去{CONFIG['lookback_days']}天)\n\n"

    msg += f"市场情绪: {sentiment}\n\n"

    msg += f"扫描范围: {valid_symbols}/{total_symbols} 个有效合约"

    return msg


# ============================================================
# 主流程
# ============================================================
def scan_adl_ratio():
    """
    扫描所有合约计算涨跌比率

    Returns:
        result dict 或 None（成功）, 或 "API_FAILED"（API失败）
    """
    print("=" * 60)
    print(f"[开始] 涨跌比率扫描 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. 获取所有USDT合约
    all_symbols = get_all_usdt_futures()

    if all_symbols is None:
        print("[错误] API请求失败，无法获取交易对列表")
        return "API_FAILED"

    if len(all_symbols) == 0:
        print("[错误] 未获取到任何交易对")
        return "API_FAILED"

    # 2. 遍历所有合约，计算收益率
    rise_count = 0  # 涨幅>5%
    fall_count = 0  # 跌幅>5%
    valid_count = 0  # 有效合约数
    total = len(all_symbols)

    rise_symbols = []  # 记录涨幅>5%的交易对
    fall_symbols = []  # 记录跌幅>5%的交易对

    for idx, symbol in enumerate(all_symbols, 1):
        if idx % 50 == 0 or idx == 1:
            print(f"\n[进度] {idx}/{total}...")

        # 更新缓存
        df = update_daily_cache(symbol)

        # 检查数据是否足够（至少需要2天数据来计算收益率）
        if df.empty or len(df) < 2:
            continue

        # 计算前一日收益率
        daily_return = calculate_daily_return(df)

        if daily_return is None:
            continue

        valid_count += 1

        # 统计涨跌
        if daily_return > CONFIG["rise_threshold"]:
            rise_count += 1
            rise_symbols.append((symbol, daily_return))
        elif daily_return < CONFIG["fall_threshold"]:
            fall_count += 1
            fall_symbols.append((symbol, daily_return))

        # 限流：每20个请求休息0.5秒
        if idx % 20 == 0:
            time.sleep(0.5)

    # 计算实际统计的日期（最近完整交易日，即昨天UTC）
    today_utc = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None)).normalize()
    data_date = today_utc - timedelta(days=1)  # 实际计算的是昨天的数据

    print(f"\n[统计] 数据日期: {data_date.strftime('%Y-%m-%d')} (UTC)")
    print(f"[统计] 有效合约: {valid_count}/{total}")
    print(f"[统计] 涨幅>5%: {rise_count} 个")
    print(f"[统计] 跌幅>5%: {fall_count} 个")

    # 3. 计算涨跌比率（分母为0时用1替代）
    denominator = fall_count if fall_count > 0 else 1
    ratio = rise_count / denominator
    print(f"[统计] 涨跌比率: {ratio:.2f}")

    # 4. 保存记录（使用实际数据日期，而非运行日期）
    history_df = add_ratio_record(data_date, rise_count, fall_count, ratio)

    # 5. 计算分位数
    # 使用过去lookback_days天的数据（不包含当前记录的日期）
    history_df["date"] = pd.to_datetime(history_df["date"])

    # 获取历史数据（不包含当前记录）
    historical_ratios = history_df[history_df["date"] < data_date]["ratio"]

    # 只使用最近lookback_days天
    if len(historical_ratios) > CONFIG["lookback_days"]:
        historical_ratios = historical_ratios.tail(CONFIG["lookback_days"])

    percentile = calculate_percentile(ratio, historical_ratios)
    print(f"[统计] 分位数: {percentile:.1f}% (基于{len(historical_ratios)}天历史数据)")

    # 打印涨幅最大的几个
    if rise_symbols:
        rise_symbols.sort(key=lambda x: x[1], reverse=True)
        print(f"\n[涨幅TOP5]")
        for s, r in rise_symbols[:5]:
            print(f"  {s}: +{r:.2f}%")

    # 打印跌幅最大的几个
    if fall_symbols:
        fall_symbols.sort(key=lambda x: x[1])
        print(f"\n[跌幅TOP5]")
        for s, r in fall_symbols[:5]:
            print(f"  {s}: {r:.2f}%")

    return {
        "data_date": data_date,  # 实际数据日期
        "rise_count": rise_count,
        "fall_count": fall_count,
        "ratio": ratio,
        "percentile": percentile,
        "total_symbols": total,
        "valid_symbols": valid_count,
        "rise_symbols": rise_symbols,
        "fall_symbols": fall_symbols,
    }


def main():
    """主函数"""
    try:
        # 扫描涨跌比率
        result = scan_adl_ratio()

        if result == "API_FAILED":
            # API失败，发送错误通知
            print("[错误] 扫描失败，API请求错误")
            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            message = f"*涨跌比率扫描失败*\n时间: {now}\n\n⚠️ API请求失败，请检查网络连接"
            send_telegram_alert(message)
            return

        if result is None:
            print("[错误] 扫描失败")
            return

        print("\n" + "=" * 60)
        print("[发送Telegram通知]")
        print("=" * 60)

        # 格式化并发送消息
        message = format_signal_message(
            result["data_date"],
            result["rise_count"],
            result["fall_count"],
            result["ratio"],
            result["percentile"],
            result["total_symbols"],
            result["valid_symbols"]
        )

        send_telegram_alert(message)

        print(f"\n[完成] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        print(f"[致命错误] {e}")
        import traceback
        traceback.print_exc()

        # 发送错误通知
        error_msg = f"*涨跌比率扫描出错*\n\n错误: {str(e)}"
        send_telegram_alert(error_msg)


if __name__ == "__main__":
    main()
