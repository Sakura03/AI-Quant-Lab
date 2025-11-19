def timeframe_to_seconds(timeframe: str) -> int:
    """
        将 ccxt/自定义的时间周期字符串转换为秒数
        支持: s, m, min, h, d, w
        示例: "5m" -> 300, "1h" -> 3600
    """
    if timeframe.endswith("s"):
        amount = int(timeframe[:-1])
        return amount
    elif timeframe.endswith("m"):
        amount = int(timeframe[:-1])
        return amount * 60
    elif timeframe.endswith("min"):
        amount = int(timeframe[:-3])
        return amount * 60
    elif timeframe.endswith("h"):
        amount = int(timeframe[:-1])
        return amount * 3600
    elif timeframe.endswith("d"):
        amount = int(timeframe[:-1])
        return amount * 86400
    elif timeframe.endswith("w"):
        amount = int(timeframe[:-1])
        return amount * 604800
    else:
        raise ValueError(f"无效的时间周期: {timeframe:s}")


def format_timeframe(timeframe: str) -> str:
    """
        将 ccxt/自定义的时间周期字符串转换为中文时间
        支持: s, m, min, h, d, w
        示例: "5m" -> "5分钟", "1h" -> "1小时"
    """
    if timeframe.endswith("s"):
        return timeframe.replace("s", "秒")
    elif timeframe.endswith("m"):
        return timeframe.replace("m", "分钟")
    elif timeframe.endswith("min"):
        return timeframe.replace("min", "分钟")
    elif timeframe.endswith("h"):
        return timeframe.replace("h", "小时")
    elif timeframe.endswith("d"):
        return timeframe.replace("d", "天")
    elif timeframe.endswith("w"):
        return timeframe.replace("w", "周")
    else:
        raise ValueError(f"无效的时间周期: {timeframe:s}")


def format_time_interval(seconds: int, return_second: bool = False) -> str:
    """
        将秒数格式化为中文可读时间
        示例: 3700 -> "1小时1分钟40秒"
    """
    parts = []
    minutes = seconds // 60
    seconds = seconds % 60
    parts.append(f"{seconds:d}秒")

    hours = 0
    if minutes >= 60:
        hours = minutes // 60
        minutes = minutes % 60
    parts.append(f"{minutes:d}分钟")

    days = 0
    if hours >= 24:
        days = hours // 24
        hours = hours % 24

    if hours > 0:
        parts.append(f"{hours:d}小时")

    if days > 0:
        parts.append(f"{days:d}天")

    if not return_second:
        parts = parts[1:]

    return "".join(reversed(parts))


def format_symbol(symbol: str) -> str:
    """
        将 symbol 格式化为标准形式: XXX/USDT
        可接受输入格式:
            - "BTC"
            - "BTCUSDT"
            - "BTC/USDT"
            - "BTC/USDT:USDT"
            - "BTC-USDT"
            - "BTCUSDT:USDT"

        输出统一为:
            "BTC/USDT"
    """
    if not isinstance(symbol, str):
        raise TypeError("symbol的类型不是string")

    s = symbol.upper().strip()

    # 1. 去掉多余后缀，如 ":USDT"
    if ":" in s:
        s = s.split(":")[0]

    # 2. 替换常见分隔符为统一的 '/'
    s = s.replace("-", "/")

    # 3. 如果已经是 XXX/USDT 格式
    if "/" in s:
        base = s.split("/")[0]

    # 4. 如果是连续字符串，如 "BTCUSDT"
    elif s.endswith("USDT"):
        base = s[:-4]

    # 5. 输入只有币种，如 "BTC"
    else:
        base = s

    return f"{base:s}/USDT"
