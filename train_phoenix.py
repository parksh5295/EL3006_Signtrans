# coding: utf-8
"""
Training module for PHOENIX-2014T dataset.
"""
import argparse
import logging
import os
import shutil
import time
from typing import List
import yaml

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

# --- Phoenix Specific Imports ---
from signjoey.helpers_nonmap import load_config, make_logger, log_config, load_checkpoint
from signjoey.data_phoenix import load_data_phoenix, make_data_iter_phoenix
# --- End Phoenix Specific Imports ---

from signjoey.model import SignModel, build_model
from signjoey.batch import Batch
from signjoey.vocabulary import (
    PAD_TOKEN,
    GlossVocabulary,
    TextVocabulary,
)
from signjoey.loss import XentLoss
from signjoey.scheduler import Scheduler, PyTorchScheduler, SignJoeyScheduler


class TrainManager:
    """
    Manages training loop, checks checkpoints, logs training progress.
    """
    def __init__(self, model: SignModel, config: dict, rank: int = 0, world_size: int = 1) -> None:
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
        
        self.validation_freq = train_config.get("validation_freq", 1000)
        self.logging_freq = train_config.get("logging_freq", 100)
        self.num_workers = config["data"].get("num_workers", 4)
        
        self.step = 0
        self.epoch = 0
        self.best_ckpt_score = -1
        self.best_ckpt_iteration = 0
        
        self.gls_vocab, self.txt_vocab = None, None
        self.train_data, self.dev_data = None, None
        
        if self.is_main_process:
            self.valid_report_file = f"{self.model_dir}/validations.txt"
        
        self.loss = {}

    def _build_scheduler(self, train_config):
        learning_rate = train_config["learning_rate"]
        scheduler_config = train_config["scheduler"]
        optim_config = train_config.get("optimizer", {})
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
        else:
            scheduler = Scheduler(optimizer=opt, learning_rate=learning_rate)
        return scheduler, opt

    def _build_loss(self, train_config):
        model_tasks = self.config["model"].get("tasks", ["translation"])

        recognition_loss_fun = torch.nn.CTCLoss(
            blank=self.gls_vocab.stoi[PAD_TOKEN],
            zero_infinity=True,
        ) if "recognition" in model_tasks else None

        translation_loss_fun = XentLoss(
            pad_index=self.txt_vocab.stoi[PAD_TOKEN],
            smoothing=train_config.get("label_smoothing", 0.0),
        ) if "translation" in model_tasks else None

        return {
            "recognition": recognition_loss_fun,
            "translation": translation_loss_fun,
        }

    def _train_step(self, batch: Batch):
        self.model.train()
        
        total_loss = 0
        
        if self.loss.get("recognition") is not None:
            rec_loss, _ = self.model.get_loss_for_batch(batch, self.loss["recognition"], None, 1.0, 0.0)
            total_loss += rec_loss
            if self.is_main_process: self.tensorboard_writer.add_scalar("train/recognition_loss", rec_loss.item(), self.step)

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

    def train_and_validate(self, train_data, dev_data):
        self.train_data, self.dev_data = train_data, dev_data

        train_iter = make_data_iter_phoenix(
            dataset=self.train_data,
            batch_size=self.batch_size,
            gls_vocab=self.gls_vocab,
            txt_vocab=self.txt_vocab,
            shuffle=self.shuffle,
            use_ddp=(self.world_size > 1),
            rank=self.rank,
            world_size=self.world_size
        )

        for epoch_no in range(self.epochs):
            self.epoch = epoch_no
            if self.is_main_process: self.logger.info("EPOCH %d", epoch_no + 1)
            
            if self.world_size > 1 and hasattr(train_iter.sampler, 'set_epoch'):
                train_iter.sampler.set_epoch(epoch_no)
            
            for batch in iter(train_iter):
                self.step += 1
                loss = self._train_step(batch)
                if self.step % self.logging_freq == 0 and self.is_main_process:
                    self.logger.info(
                        "Epoch %d, Step %d: loss %f, lr %f",
                        epoch_no + 1, self.step, loss, self.optimizer.param_groups[0]['lr']
                    )
                if self.step % self.validation_freq == 0:
                    self._validate()
            
            if self.is_main_process:
                self._save_checkpoint(name="latest")

    def _validate(self):
        if self.is_main_process:
             self.logger.info("Validation step (placeholder).")
        pass
        
    def _save_checkpoint(self, name: str):
        if not self.is_main_process: return
        model_path = f"{self.model_dir}/{name}.ckpt"
        model_state = self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict()
        state = {
            "step": self.step, "epoch": self.epoch,
            "best_ckpt_score": self.best_ckpt_score,
            "best_ckpt_iteration": self.best_ckpt_iteration,
            "model_state": model_state, "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
        }
        torch.save(state, model_path)
        if self.is_main_process: self.logger.info("Saved checkpoint %s.", model_path)


def ddp_setup():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def train_phoenix(cfg_file: str) -> None:
    cfg = load_config(cfg_file)
    
    rank, world_size = 0, 1
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        ddp_setup()
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    if rank == 0:
        make_logger(train_config=cfg["training"], model_dir=cfg["training"]["model_dir"], rank=rank)
        log_config(cfg)

    data_cfg = cfg["data"]
    train_data, dev_data, _, gls_vocab, txt_vocab = load_data_phoenix(data_cfg=data_cfg)

    model = build_model(cfg["model"], gls_vocab=gls_vocab, txt_vocab=txt_vocab)
    
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    trainer = TrainManager(model=model, config=cfg, rank=rank, world_size=world_size)
    trainer.gls_vocab = gls_vocab
    trainer.txt_vocab = txt_vocab
    trainer.loss = trainer._build_loss(cfg["training"])
    
    trainer.train_and_validate(train_data, dev_data)


def main():
    parser = argparse.ArgumentParser("Phoenix-Trainer")
    parser.add_argument(
        "config",
        default="configs/phoenix_config.yaml",
        type=str,
        help="Training configuration file for PHOENIX-2014T.",
    )
    args = parser.parse_args()
    train_phoenix(cfg_file=args.config)


if __name__ == "__main__":
    main() 