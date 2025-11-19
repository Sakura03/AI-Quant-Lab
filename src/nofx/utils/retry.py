from functools import wraps
from typing import Tuple, Type, Any

import time


def retry(
        max_retries: int,
        delay: float,
        exceptions: Tuple[Type[BaseException], ...] = (Exception,),
        output: Any = None,
        logger_attr: str = "warning",
        raise_if_fail: bool = False,
):
    """
        通用 retry 装饰器：
        - max_retries: 最大重试次数
        - delay: 延迟
        - exceptions: 捕获的异常
        - output: 当重试次数用完时的返回值
        - logger_attr
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self = args[0]  # self是BaseClassWithLogger的派生类
            log = getattr(self, logger_attr, None)
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    is_last = i == max_retries - 1
                    if log:
                        msg = f"[重试: {i+1:d}/{max_retries:d}] {func.__name__:s} 失败: {e}"
                        if not is_last:
                            msg += f", {delay:.2f}秒后重试"
                        log(msg)
                    if is_last:
                        if raise_if_fail:
                            raise ConnectionError(f"{func.__name__:s}: 重试{max_retries:d}次均失败")
                        return output
                    time.sleep(delay)
        return wrapper
    return decorator
