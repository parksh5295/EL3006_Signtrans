# coding: utf-8
"""
Training module
"""
import argparse
import logging
import os
import shutil
import time
from typing import List

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from signjoey.model import SignModel, build_model
from signjoey.batch import Batch
from signjoey.vocabulary import (
    PAD_TOKEN,
    GlossVocabulary,
    TextVocabulary,
)
from signjoey.loss import XentLoss
from signjoey.scheduler import Scheduler, PyTorchScheduler, SignJoeyScheduler


# This is the TrainManager from signjoey/train.py.
# It's embedded here to make this a single, self-contained training script.
# pylint: disable=too-many-instance-attributes
class TrainManager:
    """
    Manages training loop, checks checkpoints, logs training progress.
    """

    def __init__(self, model: SignModel, config: dict, helpers, rank: int = 0, world_size: int = 1) -> None:
        """
        Creates a new TrainManager for a model, specified as in configuration.
        """
        train_config = config["training"]

        self.config = config
        self.model = model
        self.helpers = helpers
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = self.rank == 0
        
        self.model_dir = train_config["model_dir"]
        if self.is_main_process:
            self.tensorboard_writer = SummaryWriter(log_dir=os.path.join(self.model_dir, "runs"))
        else:
            self.tensorboard_writer = None

        self.logger = logging.getLogger(__name__)
        self.scheduler, self.optimizer = self._build_scheduler(train_config)
        
        self.shuffle = train_config["shuffle"]
        self.epochs = train_config["epochs"]
        self.batch_size = train_config["batch_size"]
        self.batch_type = train_config.get("batch_type", "sentence")
        
        self.validation_freq = train_config.get("validation_freq", 1000)
        self.logging_freq = train_config.get("logging_freq", 100)
        self.num_workers = config["data"].get("num_workers", 4)
        
        self.step = 0
        self.epoch = 0
        self.best_ckpt_score = -1
        self.best_ckpt_iteration = 0
        
        self.gls_vocab, self.txt_vocab = None, None # will be set in train()
        self.train_data, self.dev_data = None, None
        
        if self.is_main_process:
            self.valid_report_file = f"{self.model_dir}/validations.txt"
        
        self.loss = {}


    def _build_scheduler(self, train_config):
        learning_rate = train_config["learning_rate"]
        scheduler_config = train_config["scheduler"]
        optim_config = train_config["optimizer"]
        opt = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, **optim_config
        )

        if scheduler_config["type"] == "plateau":
            scheduler = PyTorchScheduler(
                optimizer=opt,
                mode="min",
                factor=scheduler_config["decrease_factor"],
                patience=scheduler_config["patience"],
            )
        elif scheduler_config["type"] == "warmup":
            model_size = self.config["model"]["encoder"]["hidden_size"]
            scheduler = SignJoeyScheduler(
                optimizer=opt,
                hidden_size=model_size,
                k=scheduler_config.get("k", 1.0),
                warmup_steps=scheduler_config["warmup_steps"],
            )
        else:  # "fixed"
            scheduler = Scheduler(optimizer=opt, learning_rate=learning_rate)
        return scheduler, opt

    def _build_loss(self, train_config):
        
        model_tasks = self.config["model"].get("tasks", [])

        recognition_loss_fun = torch.nn.CTCLoss(
            blank=self.gls_vocab.stoi[PAD_TOKEN],
            zero_infinity=True,
        ) if "recognition" in model_tasks else None

        translation_loss_fun = XentLoss(
            pad_index=self.txt_vocab.stoi[PAD_TOKEN],
            smoothing=train_config["label_smoothing"],
        ) if "translation" in model_tasks else None

        return {
            "recognition": recognition_loss_fun,
            "translation": translation_loss_fun,
        }

    def _train_step(self, batch: Batch):
        self.model.train()
        
        total_loss = 0
        
        # get recognition loss
        if self.loss.get("recognition") is not None:
            rec_loss, _ = self.model.get_loss_for_batch(batch, self.loss["recognition"], None, 1.0, 0.0)
            total_loss += rec_loss
            if self.is_main_process: self.tensorboard_writer.add_scalar("train/recognition_loss", rec_loss.item(), self.step)

        # get translation loss
        if self.loss.get("translation") is not None:
             _, trans_loss = self.model.get_loss_for_batch(batch, None, self.loss["translation"], 0.0, 1.0)
             total_loss += trans_loss
             if self.is_main_process: self.tensorboard_writer.add_scalar("train/translation_loss", trans_loss.item(), self.step)

        if self.is_main_process: 
            self.tensorboard_writer.add_scalar("train/total_loss", total_loss.item(), self.step)
            self.tensorboard_writer.add_scalar("train/learning_rate", self.optimizer.param_groups[0]['lr'], self.step)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        if self.scheduler:
             self.scheduler.step()
        return total_loss.item()

    def train_and_validate(self, train_data, dev_data, make_data_iter_func):
        self.train_data, self.dev_data = train_data, dev_data

        train_iter, _ = make_data_iter_func(
            dataset=self.train_data,
            batch_size=self.batch_size,
            batch_type=self.batch_type,
            gls_vocab=self.gls_vocab,
            txt_vocab=self.txt_vocab,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            use_ddp=(self.world_size > 1),
            rank=self.rank,
        )

        for epoch_no in range(self.epochs):
            self.epoch = epoch_no
            self.logger.info("EPOCH %d", epoch_no + 1)
            
            if self.world_size > 1 and hasattr(train_iter.sampler, 'set_epoch'):
                train_iter.sampler.set_epoch(epoch_no)
            
            for i, batch in enumerate(iter(train_iter)):
                self.step += 1
                loss = self._train_step(batch)
                if self.step % self.logging_freq == 0 and self.is_main_process:
                    self.logger.info(
                        "Epoch %d, Step %d: loss %f, lr %f",
                        epoch_no + 1,
                        self.step,
                        loss,
                        self.optimizer.param_groups[0]['lr']
                    )
                if self.step % self.validation_freq == 0:
                    self._validate(dev_data)
            
            if self.is_main_process:
                self._save_checkpoint(name="latest")

    def _validate(self, dev_data):
        # Placeholder for validation logic
        self.logger.info("Validation step (not implemented).")
        pass
        
    def _save_checkpoint(self, name: str):
        if not self.is_main_process: return
        model_path = f"{self.model_dir}/{name}.ckpt"
        model_state = self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict()
        state = {
            "step": self.step,
            "epoch": self.epoch,
            "best_ckpt_score": self.best_ckpt_score,
            "best_ckpt_iteration": self.best_ckpt_iteration,
            "model_state": model_state,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
        }
        torch.save(state, model_path)
        self.logger.info("Saved checkpoint %s.", model_path)


def ddp_setup():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def train(cfg_file: str) -> None:
    # --- Always use the unified helpers module ---
    from signjoey import helpers
    from signjoey.data import load_data, make_data_iter
    from signjoey.data_nonmap import load_data_nonmap, make_data_iter_nonmap
    
    cfg = helpers.load_config(cfg_file)
    
    # Determine which data loading function to use
    data_module_name = cfg["data"].get("data_module", "data")
    if data_module_name == "data_nonmap":
        load_data_func = load_data_nonmap
        make_data_iter_func = make_data_iter_nonmap
    else:
        load_data_func = load_data
        make_data_iter_func = make_data_iter

    rank, world_size = 0, 1
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        ddp_setup()
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    model_dir = helpers.make_model_dir(
        cfg["training"]["model_dir"], overwrite=cfg["training"].get("overwrite", False), rank=rank
    )
    logger = helpers.make_logger(model_dir, mode="train", rank=rank)
    if rank == 0:
        helpers.log_cfg(cfg, logger)

    helpers.set_seed(seed=cfg["training"].get("random_seed", 42))
    
    # Load data
    data_cfg = cfg["data"]
    train_data, dev_data, test_data, gls_vocab, txt_vocab = load_data_func(data_cfg=data_cfg)

    # Build model
    model = build_model(
        cfg=cfg,
        gls_vocab=gls_vocab,
        txt_vocab=txt_vocab,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "LOCAL_RANK" in os.environ:
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    model.to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])], find_unused_parameters=True)

    # Trainer
    trainer = TrainManager(model=model, config=cfg, helpers=helpers, rank=rank, world_size=world_size)
    trainer.gls_vocab = gls_vocab
    trainer.txt_vocab = txt_vocab
    
    # Re-build loss function with correct vocab
    trainer.loss = trainer._build_loss(cfg["training"])

    # Load checkpoint
    if cfg["training"].get("load_model"):
        helpers.load_checkpoint(cfg["training"]["load_model"], trainer.model, trainer.optimizer)

    trainer.train_and_validate(train_data, dev_data, make_data_iter_func)

    if test_data is not None and rank == 0:
        logger.info("Testing not implemented.")
        pass

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SignJoey")
    parser.add_argument("config", default="configs/new_config.yaml", type=str, nargs="?", help="Training config file.")
    args = parser.parse_args()
    train(cfg_file=args.config)