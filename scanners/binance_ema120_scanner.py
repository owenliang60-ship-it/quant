#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
币安合约 EMA120 趋势扫描器
每天早上8:10扫描USDT合约，检测收盘价高于日线EMA120的标的占比

用途：衡量市场整体趋势强度
- 占比高(>60%): 多头市场，大部分币种处于上升趋势
- 占比低(<40%): 空头市场，大部分币种处于下降趋势
- 中等(40-60%): 震荡市场

Author: Claude Code
Date: 2026-01-16
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ============================================================
# 路径配置 (相对路径，支持文件夹移动)
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent  # scanners/
PROJECT_ROOT = SCRIPT_DIR.parent              # C:\CC Workspace\Quant\

# ============================================================
# 配置
# ============================================================
CONFIG = {
    # 币安合约API（无需代理）
    "base_url": "https://fapi.binance.com",

    # EMA参数
    "ema_period": 120,

    # 数据缓存 (使用完整缓存目录，包含所有币种)
    "cache_dir": str(PROJECT_ROOT / "cache" / "binance_daily_cache_full"),
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
        print(f"[错误] 获取交易对列表失败（已重试3次）")
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
            "quote_volume": float(k[7]),
        })

    df = pd.DataFrame(records)
    return df


def filter_closed_klines(df: pd.DataFrame) -> pd.DataFrame:
    """过滤掉正在进行中的日线，只保留已收盘的"""
    if df.empty:
        return df
    today_utc = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None)).normalize()
    return df[df["timestamp"] < today_utc]


def update_daily_cache(symbol: str) -> pd.DataFrame:
    """更新某个币种的日线数据缓存"""
    cache_df = load_cached_data(symbol)

    if cache_df.empty:
        # 无缓存，下载完整历史
        start_time = int((datetime.now(timezone.utc) - timedelta(days=CONFIG['history_days'])).timestamp() * 1000)
        klines = fetch_klines_by_time(symbol, start_time)

        if not klines:
            return pd.DataFrame()

        df = klines_to_dataframe(klines)
        df = filter_closed_klines(df)
        save_cached_data(symbol, df)
        return df
    else:
        # 有缓存，强制刷新最近3天数据
        today = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None)).normalize()
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

            cutoff = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None)) - timedelta(days=CONFIG['history_days'])
            df = df[df["timestamp"] >= cutoff]

            save_cached_data(symbol, df)
            return df

        return cache_df


# ============================================================
# EMA计算模块
# ============================================================
def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """
    计算EMA (Exponential Moving Average)

    Args:
        prices: 收盘价序列
        period: EMA周期

    Returns:
        EMA序列
    """
    return prices.ewm(span=period, adjust=False).mean()


def check_price_above_ema(df: pd.DataFrame, ema_period: int = 120) -> dict:
    """
    检查最新收盘价是否高于EMA

    Args:
        df: 日线数据DataFrame
        ema_period: EMA周期

    Returns:
        包含检查结果的字典
    """
    if df.empty or len(df) < ema_period:
        return None

    # 计算EMA
    ema = calculate_ema(df["close"], ema_period)

    # 获取最新值
    latest_close = df["close"].iloc[-1]
    latest_ema = ema.iloc[-1]
    latest_volume = df["quote_volume"].iloc[-1] if "quote_volume" in df.columns else 0

    # 计算价格相对EMA的偏离度
    deviation = (latest_close / latest_ema - 1) * 100

    return {
        "close": latest_close,
        "ema": latest_ema,
        "above_ema": latest_close > latest_ema,
        "deviation": deviation,  # 正数表示高于EMA，负数表示低于
        "volume": latest_volume,  # 昨日USDT交易量
        "date": df["timestamp"].iloc[-1].strftime("%Y-%m-%d")
    }


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


def format_volume(volume: float) -> str:
    """格式化交易量显示"""
    if volume >= 1e9:
        return f"{volume/1e9:.1f}B"
    elif volume >= 1e6:
        return f"{volume/1e6:.1f}M"
    elif volume >= 1e3:
        return f"{volume/1e3:.1f}K"
    else:
        return f"{volume:.0f}"


def format_summary_message(results: dict, above_list: list) -> str:
    """
    格式化摘要消息

    Args:
        results: 扫描结果统计
        above_list: 高于EMA的标的列表（已按交易量排序）

    Returns:
        格式化的摘要消息
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # 计算市场情绪
    ratio = results["above_ratio"]
    if ratio >= 60:
        sentiment = "多头市场"
        emoji = "🟢"
    elif ratio <= 40:
        sentiment = "空头市场"
        emoji = "🔴"
    else:
        sentiment = "震荡市场"
        emoji = "🟡"

    msg = f"*EMA120 趋势扫描* {emoji}\n"
    msg += f"时间: {now}\n\n"

    msg += f"📊 *市场统计*\n"
    msg += f"• 总标的数: {results['total']}\n"
    msg += f"• 高于EMA120: {results['above_count']} ({ratio:.1f}%)\n"
    msg += f"• 低于EMA120: {results['below_count']} ({100-ratio:.1f}%)\n"
    msg += f"• 市场状态: *{sentiment}*\n"

    return msg


def format_full_list_messages(above_list: list) -> list:
    """
    格式化完整标的列表消息（按交易量排序）
    由于Telegram消息限制4096字符，需要分多条发送

    Args:
        above_list: 高于EMA的标的列表（已按交易量排序）

    Returns:
        消息列表
    """
    if not above_list:
        return []

    messages = []
    current_msg = f"🚀 *高于EMA120标的* (按交易量排序)\n\n"

    for i, item in enumerate(above_list, 1):
        symbol = item['symbol'].replace('USDT', '')  # 简化显示
        deviation = item['deviation']
        volume = format_volume(item['volume'])

        line = f"{i}. {symbol} | +{deviation:.1f}% | {volume}\n"

        # Telegram消息限制4096字符，预留一些空间
        if len(current_msg) + len(line) > 3800:
            messages.append(current_msg)
            current_msg = f"🚀 *高于EMA120标的* (续)\n\n"

        current_msg += line

    if current_msg.strip():
        messages.append(current_msg)

    return messages


def format_result_message(results: dict, above_list: list, below_list: list) -> str:
    """保留旧函数兼容性，返回摘要消息"""
    return format_summary_message(results, above_list)


# ============================================================
# 主流程
# ============================================================
def scan_ema120() -> tuple:
    """
    扫描所有币种的EMA120状态

    Returns:
        (results_dict, api_failed): 结果字典和API是否失败的标志
    """
    print("=" * 50)
    print(f"[开始] EMA120趋势扫描 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    # 1. 获取所有USDT永续合约
    symbols = get_all_usdt_futures()

    if symbols is None:
        print("[错误] API请求失败，无法获取交易对列表")
        return None, True

    if len(symbols) == 0:
        print("[错误] 未获取到任何交易对")
        return None, True

    # 2. 逐个更新缓存并检查EMA状态
    above_ema = []  # 高于EMA的标的
    below_ema = []  # 低于EMA的标的
    skipped = 0
    total = len(symbols)

    for idx, symbol in enumerate(symbols, 1):
        # 进度显示 (每20个显示一次)
        if idx % 20 == 0 or idx == 1:
            print(f"\n[{idx}/{total}] 处理中...")

        # 更新缓存
        df = update_daily_cache(symbol)

        if df.empty or len(df) < CONFIG["ema_period"] + 10:
            skipped += 1
            continue

        # 检查EMA状态
        result = check_price_above_ema(df, CONFIG["ema_period"])

        if result is None:
            skipped += 1
            continue

        result["symbol"] = symbol

        if result["above_ema"]:
            above_ema.append(result)
        else:
            below_ema.append(result)

        # 限流：每20个请求休息1秒
        if idx % 20 == 0:
            time.sleep(1)

    # 3. 统计结果
    valid_count = len(above_ema) + len(below_ema)
    above_ratio = len(above_ema) / valid_count * 100 if valid_count > 0 else 0

    results = {
        "total": valid_count,
        "above_count": len(above_ema),
        "below_count": len(below_ema),
        "above_ratio": above_ratio,
        "skipped": skipped
    }

    # 按交易量排序（高于EMA的按交易量从大到小）
    above_ema.sort(key=lambda x: x["volume"], reverse=True)
    # 低于EMA的按偏离度排序（最弱的排前面）
    below_ema.sort(key=lambda x: x["deviation"])

    return (results, above_ema, below_ema), False


def main():
    """主函数"""
    try:
        # 扫描
        result, api_failed = scan_ema120()

        print("\n" + "=" * 50)
        print("[结果]")
        print("=" * 50)

        if api_failed:
            print("[错误] 扫描失败，API请求错误")
            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            message = f"*EMA120 扫描失败*\n时间: {now}\n\n⚠️ API请求失败，请检查网络连接"
            send_telegram_alert(message)
        else:
            results, above_list, below_list = result

            print(f"总标的数: {results['total']}")
            print(f"高于EMA120: {results['above_count']} ({results['above_ratio']:.1f}%)")
            print(f"低于EMA120: {results['below_count']} ({100-results['above_ratio']:.1f}%)")
            print(f"跳过(数据不足): {results['skipped']}")

            if above_list:
                print(f"\n高于EMA120标的 (按交易量排序):")
                for i, item in enumerate(above_list[:10], 1):
                    print(f"  {i}. {item['symbol']}: +{item['deviation']:.1f}% | Vol: {format_volume(item['volume'])}")
                if len(above_list) > 10:
                    print(f"  ... 共 {len(above_list)} 个")

            # 发送Telegram - 摘要消息
            summary_msg = format_summary_message(results, above_list)
            send_telegram_alert(summary_msg)

            # 发送Telegram - 完整列表（按交易量排序）
            if above_list:
                time.sleep(1)  # 避免发送过快
                list_messages = format_full_list_messages(above_list)
                for i, msg in enumerate(list_messages):
                    print(f"[Telegram] 发送列表消息 {i+1}/{len(list_messages)}...")
                    send_telegram_alert(msg)
                    if i < len(list_messages) - 1:
                        time.sleep(1)  # 消息间隔

        print(f"\n[完成] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        print(f"[致命错误] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
