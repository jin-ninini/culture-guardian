import logging
import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger(log_path)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        ch = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger
