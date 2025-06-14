import shutil
import time
from typing import List
import importlib

import torch
import torch.distributed as dist

import os

from signjoey.config import (
    load_config,
    log_cfg,
    load_checkpoint,
    make_model_dir,
    make_logger,
    set_seed,
    # SymbolicLinks,
)
# from signjoey.data import load_data, make_data_iter, SignTranslationDataset
from signjoey.vocabulary import (
    PAD_TOKEN,
    GlossVocabulary,
# ... existing code ...
def train(cfg_file: str) -> None:
    """
    Main training function.

    :param cfg_file: path to configuration file
    """
    cfg = load_config(cfg_file)

    # get DDP rank and world size
    rank, world_size = 0, 1
    if "WORLD_SIZE" in os.environ:
        ddp_setup()
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    if rank == 0:
        # set the random seed
        set_seed(seed=cfg["training"].get("random_seed", 42))

        # create logger
        model_dir = make_model_dir(
            cfg["training"]["model_dir"], overwrite=cfg["training"].get("overwrite", False)
        )
        make_logger(model_dir, mode="train")
        
        # log configuration
        log_cfg(cfg)

    # wait for the main process to create the model directory
    # and set up the logger.
    if world_size > 1:
        dist.barrier()
        
    # data
    data_cfg = cfg["data"]
    
    # Dynamically import the data loading module
    data_module_name = data_cfg.get("data_module", "data")
    print(f"Using data module: {data_module_name}")
    data_module = importlib.import_module(f"signjoey.{data_module_name}")
    
    if data_module_name == "data_nonmap":
        load_data_func = data_module.load_data_nonmap
        make_data_iter_func = data_module.make_data_iter_nonmap
        DatasetClass = data_module.SignTranslationDataset_NonMap
    else: # original "data" module
        load_data_func = data_module.load_data
        make_data_iter_func = data_module.make_data_iter
        DatasetClass = data_module.SignTranslationDataset

    train_data, dev_data, test_data, gls_vocab, txt_vocab = load_data_func(
        data_cfg=data_cfg
    )

    if rank == 0:
        # store vocabularies
        self.dev_data = dev_data
        self.test_data = test_data

        self.train_iter, self.train_sampler = make_data_iter_func(
            dataset=self.train_data,
            batch_size=self.batch_size,
            gls_vocab=self.gls_vocab,
            txt_vocab=self.txt_vocab,
        )

    trainer.train_and_validate(train_data=train_data, dev_data=dev_data)

    if rank == 0:
        # test the model
        if test_data is not None:
            trainer.testing(
                test_data=test_data,
                gls_vocab=gls_vocab,
                txt_vocab=txt_vocab,
            )
    
    if world_size > 1:
        dist.destroy_process_group()
# ... existing code ... 