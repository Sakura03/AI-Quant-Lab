from typing import Optional

import os
import os.path as osp
import logging


str_to_level = logging.getLevelNamesMapping()


def setup_logger(name: str, log_file: Optional[str] = None, level_name: str = "INFO") -> logging.Logger:
    level = str_to_level.get(level_name, logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')

    # 控制台 handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 文件 handler（可选）
    if log_file:
        os.makedirs(osp.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


class BaseClassWithLogger:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger

    def debug(self, *args, **kwargs):
        if self.logger:
            self.logger.debug(*args, **kwargs)

    def info(self, *args, **kwargs):
        if self.logger:
            self.logger.info(*args, **kwargs)

    def warning(self, *args, **kwargs):
        if self.logger:
            self.logger.warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        if self.logger:
            self.logger.error(*args, **kwargs)

    def exception(self, *args, **kwargs):
        if self.logger:
            self.logger.exception(*args, **kwargs)

    def critical(self, *args, **kwargs):
        if self.logger:
            self.logger.critical(*args, **kwargs)

    def log(self, *args, **kwargs):
        if self.logger:
            self.logger.log(*args, **kwargs)
