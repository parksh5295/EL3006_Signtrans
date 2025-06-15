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
import importlib

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from signjoey.model import SignModel, build_model
from signjoey.batch import Batch
from signjoey.helpers import (
    log_data_info,
    load_config,
    log_cfg,
    load_checkpoint,
    make_model_dir,
    make_logger,
    set_seed,
)
from signjoey.vocabulary import (
    PAD_TOKEN,
    GlossVocabulary,
    TextVocabulary,
    build_vocab,
)
from signjoey.loss import XentLoss
from signjoey.prediction import validate_on_data
from signjoey.scheduler import Scheduler, PyTorchScheduler, SignJoeyScheduler
from signjoey.metrics import wer_single

# This is the TrainManager from signjoey/train.py.
# I'm putting it here to make this a single, self-contained training script.
# pylint: disable=too-many-instance-attributes
class TrainManager:
    """
    Manages training loop, checks checkpoints, logs training progress.
    """

    def __init__(self, model: SignModel, config: dict, rank: int = 0, world_size: int = 1) -> None:
        """
        Creates a new TrainManager for a model, specified as in configuration.
        """
        train_config = config["training"]

        self.config = config
        self.model = model
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
        self.batch_multiplier = train_config.get("batch_multiplier", 1)
        
        self.validation_freq = train_config.get("validation_freq", 1000)
        self.logging_freq = train_config.get("logging_freq", 100)
        self.num_valid_log = train_config.get("num_valid_log", 5) if self.is_main_process else 0
        self.num_workers = config["data"].get("num_workers", 4)
        
        self.fp16 = train_config.get("fp16", False)
        self.scaler = torch.cuda.amp.GradScaler() if self.fp16 else None
        
        self.step = 0
        self.epoch = 0
        self.best_ckpt_score = -1
        self.best_ckpt_iteration = 0
        
        self.gls_vocab, self.txt_vocab = None, None # will be set in train()
        self.train_data, self.dev_data = None, None
        
        if self.is_main_process:
            self.valid_report_file = f"{self.model_dir}/validations.txt"
        
        self.loss = self._build_loss(train_config)


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
                k=scheduler_config["k"],
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
        rec_loss, trans_loss = 0, 0
        
        if self.loss["recognition"] is not None:
            rec_loss, _ = self.model.get_loss_for_batch(batch, self.loss["recognition"], None, 1.0, 0.0)
            total_loss += rec_loss
            if self.is_main_process: self.tensorboard_writer.add_scalar("train/recognition_loss", rec_loss.item(), self.step)

        if self.loss["translation"] is not None:
             _, trans_loss = self.model.get_loss_for_batch(batch, None, self.loss["translation"], 0.0, 1.0)
             total_loss += trans_loss
             if self.is_main_process: self.tensorboard_writer.add_scalar("train/translation_loss", trans_loss.item(), self.step)

        if self.is_main_process: self.tensorboard_writer.add_scalar("train/total_loss", total_loss.item(), self.step)

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
        # TODO: Validation logic from signjoey/train.py
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
    cfg = load_config(cfg_file)

    rank, world_size = 0, 1
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        ddp_setup()
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    model_dir = make_model_dir(
        cfg["training"]["model_dir"], overwrite=cfg["training"].get("overwrite", False), rank=rank
    )
    make_logger(model_dir, mode="train", rank=rank)
    if rank == 0:
        log_cfg(cfg)

    set_seed(seed=cfg["training"].get("random_seed", 42))
    
    # Dynamic data loading
    data_cfg = cfg["data"]
    data_module_name = data_cfg.get("data_module", "data_nonmap")
    if rank == 0: print(f"Using data module: {data_module_name}")
    data_module = importlib.import_module(f"signjoey.{data_module_name}")
    
    load_data_func = getattr(data_module, 'load_data_nonmap' if data_module_name == 'data_nonmap' else 'load_data')
    make_data_iter_func = getattr(data_module, 'make_data_iter_nonmap' if data_module_name == 'data_nonmap' else 'make_data_iter')
    
    train_data, dev_data, test_data, gls_vocab, txt_vocab = load_data_func(data_cfg=data_cfg)

    # Build model
    model = build_model(
        cfg=cfg["model"],
        gls_vocab=gls_vocab,
        txt_vocab=txt_vocab,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "LOCAL_RANK" in os.environ:
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    model.to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])

    # Trainer
    trainer = TrainManager(model=model, config=cfg, rank=rank, world_size=world_size)
    trainer.gls_vocab = gls_vocab
    trainer.txt_vocab = txt_vocab
    
    # Re-build loss function with correct vocab
    trainer.loss = trainer._build_loss(cfg)

    # Load checkpoint
    if cfg["training"].get("load_model"):
        load_checkpoint(cfg["training"]["load_model"], model, trainer.optimizer)

    trainer.train_and_validate(train_data, dev_data, make_data_iter_func)

    if test_data is not None and rank == 0:
        # TODO: Testing logic
        trainer.logger.info("Testing not implemented.")
        pass

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SignJoey")
    parser.add_argument("config", default="configs/default.yaml", type=str, nargs="?", help="Training config file.")
    args = parser.parse_args()
    train(cfg_file=args.config) 