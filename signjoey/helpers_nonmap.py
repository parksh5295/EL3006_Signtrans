# coding: utf-8
"""
Collection of helper functions for the non-mapping data pipeline.
This is a separate file to avoid modifying the original helpers.py,
which contains complex logic for handling recursive includes in configs.
"""
import copy
import glob
import os
import os.path
import errno
import shutil
import random
import logging
from sys import platform
from logging import Logger
from typing import Callable, Optional
import numpy as np

import torch
from torch import nn, Tensor
# Note: We avoid a direct import of SignTranslationDataset to prevent circular dependencies.
# from signjoey.data import SignTranslationDataset 
import yaml
from signjoey.vocabulary import GlossVocabulary, TextVocabulary


def make_model_dir(model_dir: str, overwrite: bool = False, rank: int = 0) -> str:
    """
    Create a new directory for the model.
    """
    if rank == 0:
        if os.path.isdir(model_dir):
            if not overwrite:
                raise FileExistsError("Model directory exists and overwriting is disabled.")
            shutil.rmtree(model_dir)
        os.makedirs(model_dir)
    return model_dir


def make_logger(model_dir: str, mode: str = "train", rank: int = 0) -> Logger:
    """
    Create a logger for logging the training process.
    """
    logger = logging.getLogger(__name__)
    log_file = f"{mode}.log"
    
    if not logger.handlers:
        logger.setLevel(level=logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s %(message)s")
        
        if rank == 0:
            fh = logging.FileHandler(os.path.join(model_dir, log_file))
            fh.setLevel(level=logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        if platform == "linux" or platform == "darwin":
            sh = logging.StreamHandler()
            sh.setLevel(logging.INFO)
            sh.setFormatter(formatter)
            logging.getLogger("").addHandler(sh)
            
        if rank == 0:
            logger.info("Hello! This is SignJoey.")
            
    return logger


def log_cfg(cfg: dict, logger: Logger, prefix: str = "cfg"):
    """
    Write configuration to log.
    """
    for k, v in cfg.items():
        if isinstance(v, dict):
            p = ".".join([prefix, k])
            log_cfg(v, logger, prefix=p)
        else:
            p = ".".join([prefix, k])
            logger.info("{:34s} : {}".format(p, v))


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """
    Produce N identical layers.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def subsequent_mask(size: int) -> Tensor:
    """
    Mask out subsequent positions.
    """
    mask = np.triu(np.ones((1, size, size)), k=1).astype("uint8")
    return torch.from_numpy(mask) == 0


def set_seed(seed: int):
    """
    Set the random seed for modules torch, numpy and random.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_config(path: str) -> dict:
    """
    Loads and parses a YAML configuration file.
    This is the simplified version that doesn't handle includes.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found at {path}")
    with open(path, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


def get_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """
    Returns the latest checkpoint (by time) from the given directory.
    """
    list_of_files = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    latest_checkpoint = None
    if list_of_files:
        latest_checkpoint = max(list_of_files, key=os.path.getctime)
    return latest_checkpoint


def load_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer = None) -> (int, int, float):
    """
    Load model from saved checkpoint.
    """
    assert os.path.isfile(path), f"Checkpoint {path} not found"
    checkpoint = torch.load(path, map_location="cuda" if torch.cuda.is_available() else "cpu")

    # load model state from checkpoint
    model_state = checkpoint["model_state"]
    # Handle DDP-wrapped models
    is_ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)
    if is_ddp:
        model.module.load_state_dict(model_state)
    else:
        model.load_state_dict(model_state)

    if optimizer and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    step = checkpoint.get("step", 0)
    epoch = checkpoint.get("epoch", 0)
    best_ckpt_score = checkpoint.get("best_ckpt_score", -1)
    
    return step, epoch, best_ckpt_score


def freeze_params(module: nn.Module):
    """
    Freeze the parameters of this module.
    """
    for _, p in module.named_parameters():
        p.requires_grad = False 