import os

import torch
from loguru import logger


def get_device():
    if (env_device := os.environ.get("DEVICE")) is not None:
        logger.info(f"Using device from environment: {env_device}")
        return env_device
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        logger.info("MPS backend detected, using MPS")
        return "mps"
    if torch.cuda.is_available():
        logger.info("CUDA detected, using CUDA")
        return "cuda"
    logger.info("No GPU detected, using CPU")
    return "cpu"


DEVICE = get_device()
