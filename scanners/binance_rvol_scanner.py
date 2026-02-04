#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
币安合约 RVOL 信号扫描器
每天早上8点扫描USDT合约，检测RVOL>=3的异常成交量信号

RVOL (Relative Volume Standard Deviation) = (Volume - Mean) / StDev
当RVOL>=3时，表示当日成交量超过历史均值3个标准差，属于异常放量

Author: Claude Code
Date: 2026-01-03
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
PROJECT_ROOT = SCRIPT_DIR.parent              # 项目根目录

# ============================================================
# 配置
# ============================================================
CONFIG = {
    # 币安合约API（无需代理）
    "base_url": "https://fapi.binance.com",

    # RVOL参数
    "lookback": 120,      # 回看周期（天）
    "threshold": 4.0,     # RVOL阈值（标准差倍数）

    # 过滤条件
    "top_n": 50,  # 只扫描成交量前50

    # 数据缓存 (相对路径)
    "cache_dir": str(PROJECT_ROOT / "cache" / "binance_daily_cache"),
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
        return None  # 返回None表示API失败，区别于空列表

    symbols = []
    for s in data.get("symbols", []):
        if (s.get("quoteAsset") == "USDT" and
            s.get("contractType") == "PERPETUAL" and
            s.get("status") == "TRADING"):
            symbols.append(s["symbol"])

    print(f"[API] 获取到 {len(symbols)} 个USDT永续合约")
    return symbols


def get_24h_volume() -> dict:
    """获取所有合约24h成交量（USDT计价）"""
    url = f"{CONFIG['base_url']}/fapi/v1/ticker/24hr"

    data = api_request_with_retry(url)

    if data is None:
        print(f"[错误] 获取24h成交量失败（已重试3次）")
        return None  # 返回None表示API失败

    volumes = {}
    for item in data:
        symbol = item.get("symbol", "")
        quote_volume = float(item.get("quoteVolume", 0))
        volumes[symbol] = quote_volume

    return volumes


def get_yesterday_daily_volume(symbol: str) -> float:
    """获取某个币种昨日日线成交量（USDT计价）"""
    cache_path = get_cache_path(symbol)
    if not cache_path.exists():
        return 0

    try:
        df = pd.read_csv(cache_path, parse_dates=["timestamp"])
        if len(df) < 2:
            return 0
        df = df.sort_values("timestamp")
        # 返回最新一根日线的USDT成交量
        if "quote_volume" in df.columns:
            return float(df["quote_volume"].iloc[-1])
        return 0  # 旧缓存没有quote_volume，返回0
    except:
        return 0


def get_top_n_symbols(n: int = 50) -> list:
    """
    获取成交量前N的USDT合约

    优先使用日线成交量（更准确），如果缓存不足则用24h成交量
    """
    all_symbols = get_all_usdt_futures()

    if all_symbols is None:
        return None  # API失败

    # 方案1: 先尝试用缓存的日线成交量筛选
    # 这样可以避免24h滚动窗口导致漏掉放量币种
    cache_dir = Path(CONFIG["cache_dir"])
    symbol_volumes = []

    for s in all_symbols:
        cache_path = cache_dir / f"{s}.csv"
        if cache_path.exists():
            vol = get_yesterday_daily_volume(s)
            if vol > 0:
                symbol_volumes.append((s, vol))

    # 如果缓存数据足够（至少有100个币种有缓存），用日线成交量筛选
    if len(symbol_volumes) >= 100:
        symbol_volumes.sort(key=lambda x: x[1], reverse=True)
        top_symbols = [s[0] for s in symbol_volumes[:n]]
        print(f"[筛选] 使用日线成交量筛选前{n}: {top_symbols[:5]}...")
        return top_symbols

    # 方案2: 缓存不足，用24h成交量筛选（首次运行时）
    print("[筛选] 缓存不足，使用24h成交量筛选...")
    volumes = get_24h_volume()

    if volumes is None:
        return None  # API失败

    symbol_volumes = []
    for s in all_symbols:
        if s in volumes:
            symbol_volumes.append((s, volumes[s]))

    symbol_volumes.sort(key=lambda x: x[1], reverse=True)
    top_symbols = [s[0] for s in symbol_volumes[:n]]
    print(f"[筛选] 成交量前{n}的合约: {top_symbols[:5]}...")

    return top_symbols


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
# 数据缓存模块（复用）
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
    """
    将K线数据转换为DataFrame

    币安K线数据格式:
    k[5] = volume (基础货币成交量，如BTC个数)
    k[7] = quoteVolume (报价货币成交量，如USDT金额) <-- RVOL应该用这个
    """
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


def filter_closed_klines(df: pd.DataFrame) -> pd.DataFrame:
    """
    过滤掉正在进行中的日线，只保留已收盘的
    币安日线在 UTC 00:00 收盘，所以只保留日期 < 今天UTC日期的数据
    """
    if df.empty:
        return df
    today_utc = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None)).normalize()
    return df[df["timestamp"] < today_utc]


def update_daily_cache(symbol: str) -> pd.DataFrame:
    """
    更新某个币种的日线数据缓存
    - 如果无缓存，下载完整365天数据
    - 如果有缓存，强制刷新最近3天数据

    Bug修复说明 (2026-01-14):
    之前的逻辑只检查日期，不检查数据是否完整。
    如果前一天保存了正在进行中的日线（不完整数据），
    今天就不会更新，导致RVOL计算错误，检测不到信号。
    现在改为每天强制刷新最近3天的数据，确保使用已收盘的完整数据。
    """
    cache_df = load_cached_data(symbol)

    if cache_df.empty:
        print(f"[缓存] {symbol}: 首次下载，获取{CONFIG['history_days']}天数据...")
        start_time = int((datetime.now(timezone.utc) - timedelta(days=CONFIG['history_days'])).timestamp() * 1000)
        klines = fetch_klines_by_time(symbol, start_time)

        if not klines:
            return pd.DataFrame()

        df = klines_to_dataframe(klines)
        df = filter_closed_klines(df)  # 过滤掉进行中的日线
        save_cached_data(symbol, df)
        print(f"[缓存] {symbol}: 已保存 {len(df)} 条记录")
        return df

    else:
        # 有缓存，强制刷新最近3天数据
        today = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None)).normalize()
        refresh_start = today - timedelta(days=3)

        # 检查旧缓存是否有 quote_volume 列
        if "quote_volume" not in cache_df.columns:
            print(f"[缓存] {symbol}: 旧缓存缺少 quote_volume，重新下载完整数据...")
            cache_path = get_cache_path(symbol)
            if cache_path.exists():
                cache_path.unlink()
            return update_daily_cache(symbol)  # 递归调用，走首次下载逻辑

        # 删除缓存中最近3天的数据
        cache_df = cache_df[cache_df["timestamp"] < refresh_start]

        # 下载最近的数据
        start_time = int(refresh_start.timestamp() * 1000)
        klines = fetch_klines_by_time(symbol, start_time)

        if klines:
            new_df = klines_to_dataframe(klines)
            new_df = filter_closed_klines(new_df)  # 过滤掉进行中的日线

            df = pd.concat([cache_df, new_df], ignore_index=True)
            df = df.drop_duplicates(subset=["timestamp"], keep="last")
            df = df.sort_values("timestamp").reset_index(drop=True)

            cutoff = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None)) - timedelta(days=CONFIG['history_days'])
            df = df[df["timestamp"] >= cutoff]

            save_cached_data(symbol, df)
            return df

        return cache_df


# ============================================================
# 数据预检查模块
# ============================================================
def get_expected_latest_date():
    """
    获取预期的最新日线日期
    币安日线在 UTC 00:00 收盘，即北京时间 08:00
    """
    now_utc = datetime.now(timezone.utc)
    # 返回昨天的日期作为预期最新日期
    expected = (now_utc - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return pd.Timestamp(expected.replace(tzinfo=None))


def verify_data_freshness(symbol: str = "BTCUSDT") -> tuple:
    """
    验证数据是否已更新到最新

    Args:
        symbol: 用于检查的币种（默认BTC）

    Returns:
        (is_fresh, last_date, expected_date)
    """
    expected_date = get_expected_latest_date()
    cache_df = load_cached_data(symbol)

    if cache_df.empty:
        print(f"[预检查] {symbol} 无缓存数据")
        return False, None, expected_date

    last_date = cache_df["timestamp"].max()

    if last_date >= expected_date:
        print(f"[预检查] 数据已是最新 (缓存: {last_date.strftime('%Y-%m-%d')}, 预期: {expected_date.strftime('%Y-%m-%d')})")
        return True, last_date, expected_date
    else:
        print(f"[预检查] 数据需要更新 (缓存: {last_date.strftime('%Y-%m-%d')}, 预期: {expected_date.strftime('%Y-%m-%d')})")
        return False, last_date, expected_date


def ensure_data_updated(symbols: list) -> bool:
    """
    确保所有币种的数据已更新到最新
    如果发现数据过旧，先尝试更新一个币种来验证API可用
    同时验证缓存数据包含 quote_volume 字段

    Returns:
        True 如果数据更新成功或已是最新，False 如果更新失败
    """
    # 1. 检查 BTCUSDT 的数据是否是最新
    is_fresh, last_date, expected_date = verify_data_freshness("BTCUSDT")

    # 2. 检查是否有 quote_volume 字段（旧缓存可能没有）
    cache_df = load_cached_data("BTCUSDT")
    has_quote_volume = not cache_df.empty and "quote_volume" in cache_df.columns

    if is_fresh and has_quote_volume:
        print(f"[预检查] 数据格式正确，包含 quote_volume 字段")
        return True

    if not has_quote_volume:
        print(f"[预检查] 缓存缺少 quote_volume 字段，需要重新下载")

    # 2. 数据不是最新，尝试更新 BTCUSDT 来验证 API
    print(f"[预检查] 尝试更新 BTCUSDT 验证API...")

    # 强制更新：删除最后几天数据
    cache_df = load_cached_data("BTCUSDT")
    if not cache_df.empty:
        cutoff = cache_df["timestamp"].max() - timedelta(days=3)
        cache_df = cache_df[cache_df["timestamp"] <= cutoff]
        save_cached_data("BTCUSDT", cache_df)

    # 重新下载
    df = update_daily_cache("BTCUSDT")

    if df.empty:
        print(f"[预检查] API更新失败！")
        return False

    # 再次验证
    new_last_date = df["timestamp"].max()
    if new_last_date >= expected_date:
        print(f"[预检查] API验证成功，数据已更新到 {new_last_date.strftime('%Y-%m-%d')}")
        return True
    else:
        print(f"[预检查] 警告：API返回数据仍不是最新 (得到: {new_last_date.strftime('%Y-%m-%d')}, 预期: {expected_date.strftime('%Y-%m-%d')})")
        # 即使不是最新也继续，可能是币安数据延迟
        return True


def validate_cache_integrity(symbols: list) -> list:
    """
    校验缓存数据完整性，返回需要重新下载的币种列表

    检查项:
    1. quote_volume 列是否存在
    2. 最近3天的 quote_volume 是否有效（非空、非0）
    """
    invalid_symbols = []
    cache_dir = Path(CONFIG["cache_dir"])

    for symbol in symbols:
        cache_path = cache_dir / f"{symbol}.csv"
        if not cache_path.exists():
            continue

        try:
            df = pd.read_csv(cache_path, parse_dates=["timestamp"])

            # 检查1: quote_volume 列是否存在
            if "quote_volume" not in df.columns:
                print(f"[校验] {symbol}: 缺少 quote_volume 列")
                invalid_symbols.append(symbol)
                continue

            # 检查2: 最近3天的数据是否完整
            if len(df) >= 3:
                recent_qv = df["quote_volume"].tail(3)
                if recent_qv.isna().any() or (recent_qv == 0).any():
                    print(f"[校验] {symbol}: 最近3天 quote_volume 数据不完整")
                    invalid_symbols.append(symbol)

        except Exception as e:
            print(f"[校验] {symbol}: 读取失败 - {e}")
            invalid_symbols.append(symbol)

    return invalid_symbols


def repair_invalid_caches(invalid_symbols: list):
    """
    修复无效的缓存文件
    策略: 删除缓存文件，让 update_daily_cache 重新下载完整数据
    """
    cache_dir = Path(CONFIG["cache_dir"])

    for symbol in invalid_symbols:
        cache_path = cache_dir / f"{symbol}.csv"
        if cache_path.exists():
            print(f"[修复] 删除无效缓存: {symbol}")
            cache_path.unlink()


# ============================================================
# RVOL计算模块（新写）
# ============================================================
def calculate_rvol(volumes: pd.Series, lookback: int = 120) -> float:
    """
    计算相对成交量（标准差版本）

    RVOL = (当前成交量 - 均值) / 标准差

    公式来源: RelativeVolumeStDev by Melvin E. Dickover
    参考: Technical Analysis of Stocks & Commodities, April 2014

    Args:
        volumes: 成交量序列
        lookback: 回看周期

    Returns:
        RVOL值（几个标准差），None表示数据不足
    """
    if len(volumes) < lookback + 1:
        return None

    # 使用前lookback天计算均值和标准差（不包含当天）
    historical = volumes.iloc[-(lookback+1):-1]
    current = volumes.iloc[-1]

    avg = historical.mean()
    std = historical.std(ddof=0)  # 总体标准差

    if std == 0:
        return 0

    rvol = (current - avg) / std
    return rvol


# ============================================================
# Telegram推送模块（复用+修改）
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


def format_volume(vol: float) -> str:
    """格式化成交量显示"""
    if vol >= 1e9:
        return f"${vol/1e9:.1f}B"
    elif vol >= 1e6:
        return f"${vol/1e6:.1f}M"
    elif vol >= 1e3:
        return f"${vol/1e3:.1f}K"
    else:
        return f"${vol:.0f}"


def format_signal_message(signals: list) -> str:
    """格式化RVOL信号消息"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    msg = f"*RVOL 异常成交量信号*\n"
    msg += f"时间: {now}\n\n"
    msg += f"触发币种 ({len(signals)}个):\n"

    for i, s in enumerate(signals, 1):
        symbol = s["symbol"]
        price = s["price"]
        rvol = s["rvol"]
        volume = s["volume"]

        msg += f"{i}. *{symbol}* | ${price:,.4f} | RVOL: {rvol:.1f}σ | 成交量: {format_volume(volume)}\n"

    msg += f"\n阈值: >= {CONFIG['threshold']}σ | 回看: {CONFIG['lookback']}天\n"
    msg += f"扫描范围: 全部缓存币种"

    return msg


# ============================================================
# 主流程
# ============================================================
def get_all_cached_symbols() -> list:
    """获取所有有缓存数据的币种"""
    cache_dir = Path(CONFIG["cache_dir"])
    if not cache_dir.exists():
        return []
    symbols = [f.stem for f in cache_dir.glob("*.csv")]
    return symbols


def scan_rvol_signals() -> tuple:
    """
    扫描所有符合条件的币种

    Returns:
        (signals, api_failed): 信号列表和API是否失败的标志
    """
    print("=" * 50)
    print(f"[开始] RVOL信号扫描 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    # 0. 预检查：确保数据是最新的
    print("[步骤0] 验证数据新鲜度...")
    if not ensure_data_updated([]):
        print("[错误] 数据更新失败，无法继续扫描")
        return [], True

    # 1. 获取所有有缓存的币种（不再只扫描top N）
    # 修复：之前只扫描成交量前50，导致DASHUSDT等绝对成交量低但RVOL极高的币种被漏掉
    symbols = get_all_cached_symbols()
    print(f"[扫描范围] 所有缓存币种: {len(symbols)} 个")

    if symbols is None:
        print("[错误] API请求失败，无法获取交易对列表")
        return [], True  # API失败

    if len(symbols) == 0:
        print("[错误] 未获取到任何交易对")
        return [], True  # API异常

    # 1.5 校验并修复缓存数据完整性
    print(f"[步骤1.5] 校验缓存数据完整性...")
    invalid_symbols = validate_cache_integrity(symbols)
    if invalid_symbols:
        print(f"[校验] 发现 {len(invalid_symbols)} 个币种缓存数据不完整，正在修复...")
        repair_invalid_caches(invalid_symbols)

    # 2. 逐个更新缓存并检测信号
    signals = []
    total = len(symbols)

    for idx, symbol in enumerate(symbols, 1):
        print(f"\n[{idx}/{total}] 处理 {symbol}...")

        # 更新缓存
        df = update_daily_cache(symbol)

        if df.empty or len(df) < CONFIG["lookback"] + 10:
            print(f"  -> 数据不足，跳过")
            continue

        # 计算RVOL - 使用USDT成交量（quote_volume）
        # 如果旧缓存没有quote_volume字段，跳过该币种（需要强制刷新缓存）
        if "quote_volume" not in df.columns:
            print(f"  -> 缓存缺少quote_volume字段，跳过")
            continue

        rvol = calculate_rvol(df["quote_volume"], lookback=CONFIG["lookback"])

        if rvol is None:
            print(f"  -> 数据不足，跳过")
            continue

        # 检测信号
        if rvol >= CONFIG["threshold"]:
            signal = {
                "symbol": symbol,
                "price": df["close"].iloc[-1],
                "rvol": rvol,
                "volume": df["quote_volume"].iloc[-1],  # USDT成交量
                "timestamp": df["timestamp"].iloc[-1].strftime("%Y-%m-%d")
            }
            signals.append(signal)
            print(f"  -> 触发信号! RVOL: {rvol:.2f}σ")
        else:
            print(f"  -> RVOL: {rvol:.2f}σ")

        # 限流
        if idx % 10 == 0:
            time.sleep(1)

    return signals, False  # 成功完成扫描


def main():
    """主函数"""
    try:
        # 扫描信号
        signals, api_failed = scan_rvol_signals()

        print("\n" + "=" * 50)
        print("[结果]")
        print("=" * 50)

        if api_failed:
            # API失败，发送错误通知
            print("[错误] 扫描失败，API请求错误")
            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            message = f"*RVOL 扫描失败*\n时间: {now}\n\n⚠️ API请求失败，请检查网络连接\n\n扫描范围: 24h成交量前{CONFIG['top_n']}"
            send_telegram_alert(message)
        elif signals:
            # 按RVOL降序排序
            signals.sort(key=lambda x: x["rvol"], reverse=True)

            print(f"发现 {len(signals)} 个信号:")
            for s in signals:
                print(f"  - {s['symbol']}: RVOL {s['rvol']:.2f}σ, 成交量 {format_volume(s['volume'])}")

            # 发送Telegram
            message = format_signal_message(signals)
            send_telegram_alert(message)
        else:
            print("今日无异常成交量信号")
            # 发送无信号通知
            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            message = f"*RVOL 扫描完成*\n时间: {now}\n\n今日无异常成交量信号 (RVOL >= {CONFIG['threshold']}σ)\n\n扫描范围: 24h成交量前{CONFIG['top_n']}"
            send_telegram_alert(message)

        print(f"\n[完成] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        print(f"[致命错误] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
