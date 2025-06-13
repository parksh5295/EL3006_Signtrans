import math
from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch.optim.lr_scheduler import _LRScheduler


class Scheduler(ABC):
    """Base class for all schedulers."""

    @abstractmethod
    def step(self, step: int = None, epoch: int = None, **kwargs) -> None:
        """Update the learning rate at the given step/epoch."""
        pass

    @abstractmethod
    def get_last_lr(self) -> list:
        """Return last computed learning rate by current scheduler."""
        pass


class PyTorchScheduler(Scheduler):
    """Wrapper for PyTorch schedulers."""

    def __init__(self, scheduler: _LRScheduler):
        self.scheduler = scheduler

    def step(self, step: int = None, epoch: int = None, **kwargs) -> None:
        if step is not None:
            self.scheduler.step(step)
        elif epoch is not None:
            self.scheduler.step(epoch)
        else:
            self.scheduler.step()

    def get_last_lr(self) -> list:
        return self.scheduler.get_last_lr()


class SignJoeyScheduler(Scheduler):
    """Learning rate scheduler for SignJoey model."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        hidden_size: int,
        warmup_steps: int,
        peak_lr: float = 1e-3,
        **kwargs
    ):
        self.optimizer = optimizer
        self.hidden_size = hidden_size
        self.warmup_steps = warmup_steps
        self.peak_lr = peak_lr
        self._step = 0

    def step(self, step: int = None, epoch: int = None, **kwargs) -> None:
        if step is not None:
            self._step = step
        else:
            self._step += 1

        # Calculate learning rate
        if self._step < self.warmup_steps:
            lr = self.peak_lr * (self._step / self.warmup_steps)
        else:
            lr = self.peak_lr * (self.warmup_steps ** 0.5) * (self._step ** -0.5)

        # Update learning rate for all parameter groups
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_last_lr(self) -> list:
        return [param_group["lr"] for param_group in self.optimizer.param_groups] 