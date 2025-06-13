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
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
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
    SymbolicLinks,
)
from signjoey.data import load_data, make_data_iter, SignTranslationDataset
from signjoey.vocabulary import GlossVocabulary, TextVocabulary
from signjoey.loss import XentLoss
from signjoey.prediction import validate_on_data
from signjoey.scheduler import Scheduler, PyTorchScheduler, SignJoeyScheduler
from signjoey.metrics import wer_single

# pylint: disable=too-many-instance-attributes
class TrainManager:
    """
    Manages training loop, checks checkpoints, logs training progress.
    """

    def __init__(self, model: SignModel, config: dict) -> None:
        """
        Creates a new TrainManager for a model, specified as in configuration.

        :param model: torch module defining the model
        :param config: dictionary containing the training configurations
        """
        train_config = config["training"]

        self.config = config
        self.model = model

        # files for logging and storing
        self.model_dir = train_config["model_dir"]
        self.logging_freq = train_config.get("logging_freq", 100)
        self.valid_report_file = f"{self.model_dir}/validations.txt"

        # training
        self.shuffle = train_config["shuffle"]
        self.epochs = train_config["epochs"]
        self.batch_size = train_config["batch_size"]
        self.batch_type = train_config.get("batch_type", "sentence")
        self.batch_multiplier = train_config.get("batch_multiplier", 1)

        # optimization
        self.learning_rate_min = train_config.get("learning_rate_min", 1.0e-8)
        self.clip_grad_fun = torch.nn.utils.clip_grad_norm_
        self.clip_grad_opt = train_config.get("clipping_threshold", 1.0)

        # validation
        self.validation_freq = train_config.get("validation_freq", 1000)
        self.num_valid_log = train_config.get("num_valid_log", 5)

        # logging
        self.logger = logging.getLogger(__name__)

        # tensorboard
        self.tensorboard_writer = SummaryWriter(log_dir=os.path.join(self.model_dir, "runs"))

        # scheduler
        self.scheduler, self.optimizer = self._build_scheduler(train_config)

        # loss
        self.loss = self._build_loss(train_config)

        # data & vocab
        self.train_data, self.dev_data, self.test_data = None, None, None
        self.gls_vocab, self.txt_vocab = None, None

        # model parameters
        self.fp16 = train_config.get("fp16", False)
        if self.fp16:
            self.scaler = torch.cuda.amp.GradScaler()
        self.last_best_ckpt = ""
        self.last_latest_ckpt = ""
        self.step = 0
        self.epoch = 0
        self.best_ckpt_iteration = 0
        self.best_ckpt_score = -1
        self.best_ckpt_file = ""

    def _build_scheduler(self, train_config):
        # build scheduler
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
        # build loss
        if "translation_loss_weight" not in train_config:
            train_config["translation_loss_weight"] = 1.0
        if "recognition_loss_weight" not in train_config:
            train_config["recognition_loss_weight"] = 0.0

        if self.model.do_recognition:
            recognition_loss_fun = torch.nn.CTCLoss(
                blank=self.model.gls_vocab.stoi[self.model.gls_vocab.pad_token],
                zero_infinity=True,
            )
        else:
            recognition_loss_fun = None

        if self.model.do_translation:
            translation_loss_fun = XentLoss(
                pad_index=self.model.txt_vocab.stoi[self.model.txt_vocab.pad_token],
                smoothing=train_config["label_smoothing"],
            )
        else:
            translation_loss_fun = None

        loss_function = {
            "recognition": recognition_loss_fun,
            "translation": translation_loss_fun,
        }

        loss_weight = {
            "recognition": train_config["recognition_loss_weight"],
            "translation": train_config["translation_loss_weight"],
        }
        return {"func": loss_function, "weight": loss_weight}

    def _log_examples(
        self,
        data: SignTranslationDataset,
        batch_size: int,
        gls_vocab: GlossVocabulary,
        txt_vocab: TextVocabulary,
        n: int,
        data_name: str,
    ) -> None:
        """
        forward an example from the data and log it to the console
        """
        data_iter = make_data_iter(
            dataset=data,
            batch_size=batch_size,
            batch_type="sentence",
            shuffle=False,
            gls_vocab=gls_vocab,
            txt_vocab=txt_vocab,
            level=self.config["data"]["level"],
        )
        for i, batch in enumerate(iter(data_iter)):
            if i < n:
                self.logger.info(f"Example #{i+1} from {data_name}:")
                if self.model.do_recognition:
                    gls_hyp, _ = self.model.run_batch(
                        batch, recognition_beam_size=1, translation_beam_size=0,
                    )
                    self.logger.info(
                        f"\tGloss Reference: "
                        f'{" ".join(batch.gls_text if hasattr(batch, "gls_text") else batch.gls)}'
                    )
                    self.logger.info(f"\tGloss Hypothesis: {gls_hyp[0]}")
                if self.model.do_translation:
                    _, txt_hyp, _ = self.model.run_batch(
                        batch, recognition_beam_size=0, translation_beam_size=1,
                    )
                    self.logger.info(f"\tText Reference: {batch.txt[0]}")
                    self.logger.info(f"\tText Hypothesis: {txt_hyp[0]}")
            else:
                break

    def _save_checkpoint(self, name: str) -> None:
        """
        Save the model's current parameters and the training state to a
        checkpoint file.

        :param name: checkpoint name
        """
        model_path = f"{self.model_dir}/{name}.ckpt"
        state = {
            "step": self.step,
            "epoch": self.epoch,
            "best_ckpt_score": self.best_ckpt_score,
            "best_ckpt_iteration": self.best_ckpt_iteration,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict()
            if self.scheduler is not None
            else None,
            "amp_scaler_state": self.scaler.state_dict() if self.fp16 else None,
        }
        torch.save(state, model_path)
        self.logger.info(f"Saved checkpoint {model_path}.")

    def _train_step(self, batch: Batch):
        """
        Train the model on one batch.
        """

        # reactivate training mode
        self.model.train()

        # get recognition loss
        if self.loss["weight"]["recognition"] > 0.0:
            # get loss
            if self.fp16:
                with torch.cuda.amp.autocast():
                    recognition_loss, _ = self.model.get_loss_for_batch(
                        batch=batch,
                        recognition_loss_function=self.loss["func"]["recognition"],
                        translation_loss_function=None,
                        recognition_loss_weight=self.loss["weight"]["recognition"],
                        translation_loss_weight=0.0,
                    )
            else:
                recognition_loss, _ = self.model.get_loss_for_batch(
                    batch=batch,
                    recognition_loss_function=self.loss["func"]["recognition"],
                    translation_loss_function=None,
                    recognition_loss_weight=self.loss["weight"]["recognition"],
                    translation_loss_weight=0.0,
                )
        else:
            recognition_loss = torch.Tensor([0.0])
            if self.model.is_cuda:
                recognition_loss = recognition_loss.cuda()

        # get translation loss
        if self.loss["weight"]["translation"] > 0.0:
            if self.fp16:
                with torch.cuda.amp.autocast():
                    _, translation_loss = self.model.get_loss_for_batch(
                        batch=batch,
                        recognition_loss_function=None,
                        translation_loss_function=self.loss["func"]["translation"],
                        recognition_loss_weight=0.0,
                        translation_loss_weight=self.loss["weight"]["translation"],
                    )
            else:
                _, translation_loss = self.model.get_loss_for_batch(
                    batch=batch,
                    recognition_loss_function=None,
                    translation_loss_function=self.loss["func"]["translation"],
                    recognition_loss_weight=0.0,
                    translation_loss_weight=self.loss["weight"]["translation"],
                )
        else:
            translation_loss = torch.Tensor([0.0])
            if self.model.is_cuda:
                translation_loss = translation_loss.cuda()

        # combine losses
        total_loss = recognition_loss + translation_loss
        total_loss = total_loss / self.batch_multiplier

        # accumulate gradients
        if self.fp16:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        if self.step % self.batch_multiplier == 0:
            # clip gradients
            if self.clip_grad_opt > 0:
                if self.fp16:
                    self.scaler.unscale_(self.optimizer)
                    self.clip_grad_fun(self.model.parameters(), self.clip_grad_opt)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.clip_grad_fun(self.model.parameters(), self.clip_grad_opt)
                    self.optimizer.step()

            # zero gradients
            self.optimizer.zero_grad()

            # update learning rate
            if self.scheduler is not None:
                self.scheduler.step(self.step)

        # log losses to tensorboard
        self.tensorboard_writer.add_scalar("train/total_loss", total_loss.item(), self.step)
        self.tensorboard_writer.add_scalar("train/recognition_loss", recognition_loss.item(), self.step)
        self.tensorboard_writer.add_scalar("train/translation_loss", translation_loss.item(), self.step)
        self.tensorboard_writer.add_scalar("train/learning_rate", self.scheduler.learning_rate, self.step)

        # returns loss, gloss_loss, translation_loss
        return total_loss, recognition_loss, translation_loss

    def train_and_validate(self, train_data: SignTranslationDataset, dev_data: SignTranslationDataset) -> None:
        """
        Train the model and validate it from time to time on the dev set.
        """
        self.train_data = train_data

        train_iter = make_data_iter(
            train_data,
            batch_size=self.batch_size,
            batch_type=self.batch_type,
            shuffle=self.shuffle,
            gls_vocab=self.gls_vocab,
            txt_vocab=self.txt_vocab,
            level=self.config["data"]["level"],
        )
        self.epoch = 1
        self.logger.info(
            f"Train stats: {len(self.train_data)} sentences, "
            f"{len(self.gls_vocab)} glosses, {len(self.txt_vocab)} words."
        )

        log_data_info(
            self.train_data, self.dev_data, self.gls_vocab, self.txt_vocab
        )

        self._log_examples(
            data=dev_data.samples,
            batch_size=1,
            gls_vocab=self.gls_vocab,
            txt_vocab=self.txt_vocab,
            n=self.num_valid_log,
            data_name="dev",
        )

        self.logger.info("*" * 40)
        self.logger.info(f"Starting Epoch {self.epoch}")
        self.logger.info("*" * 40)

        epoch_start_time = time.time()
        for batch in iter(train_iter):
            self.step += 1

            self._train_step(batch)

            if self.step % self.logging_freq == 0:
                elapsed_time = time.time() - epoch_start_time
                self.logger.info(
                    f"[Epoch {self.epoch:02d} Step: {self.step:08d}] "
                    f"LR: {self.scheduler.learning_rate:1.8f}, "
                    f"Elapsed Time: {elapsed_time:04.2f}[s]"
                )
                epoch_start_time = time.time()

            if self.step % self.validation_freq == 0:
                self._validate(dev_data)

            if self.scheduler.finished(self.step):
                self.logger.info("Training Finished.")
                self._save_checkpoint(name="final")
                break

        self.tensorboard_writer.close()
        self.logger.info("Finished training.")

    def _validate(self, dev_data):
        valid_start_time = time.time()
        (
            valid_scores,
            valid_recognition_results,
            valid_translation_results,
            valid_attention_scores,
        ) = validate_on_data(
            model=self.model,
            data=dev_data,
            batch_size=self.config["training"]["eval_batch_size"],
            level=self.config["data"]["level"],
            gls_vocab=self.gls_vocab,
            txt_vocab=self.txt_vocab,
            recognition_beam_sizes=[1],
            translation_beam_sizes=[1],
        )
        valid_duration = time.time() - valid_start_time
        self.logger.info(f"Validation took {valid_duration:.2f}s.")

        # log validation results to tensorboard
        self.tensorboard_writer.add_scalar("validation/loss", valid_scores["valid_loss"], self.step)
        if "wer_1" in valid_scores:
            self.tensorboard_writer.add_scalar("validation/wer", valid_scores["wer_1"], self.step)
        if "bleu_4" in valid_scores:
            self.tensorboard_writer.add_scalar("validation/bleu", valid_scores["bleu_4"], self.step)
        if "rouge" in valid_scores:
            self.tensorboard_writer.add_scalar("validation/rouge", valid_scores["rouge"], self.step)

        # log validation results to file
        with open(self.valid_report_file, "a") as opened_file:
            opened_file.write(
                f"Step: {self.step}, "
                f"WER-1: {valid_scores['wer_1']:.5f}, "
                f"BLEU-4: {valid_scores['bleu_4']:.5f}, "
                f"ROUGE: {valid_scores['rouge']:.5f}\n"
            )

        # scheduler
        if self.scheduler is not None and self.scheduler.type == "plateau":
            self.scheduler.step(metrics=valid_scores["valid_loss"])

        # save checkpoints
        ckpt_score = (
            -valid_scores["wer_1"]
            if "wer_1" in valid_scores
            else valid_scores["bleu_4"]
        )

        if ckpt_score > self.best_ckpt_score:
            self.best_ckpt_score = ckpt_score
            self.best_ckpt_iteration = self.step
            self.logger.info(
                f"Best validation result ({ckpt_score:.5f}) at step {self.step}."
            )
            self.last_best_ckpt = f"{self.model_dir}/best.ckpt"
            self._save_checkpoint(name="best")

        self.last_latest_ckpt = f"{self.model_dir}/latest.ckpt"
        self._save_checkpoint(name="latest")

        # log examples
        self._log_examples(
            data=dev_data.samples,
            batch_size=1,
            gls_vocab=self.gls_vocab,
            txt_vocab=self.txt_vocab,
            n=self.num_valid_log,
            data_name="dev",
        )

    def testing(
        self,
        test_data: SignTranslationDataset,
        gls_vocab: GlossVocabulary,
        txt_vocab: TextVocabulary,
    ):
        """
        final testing on the test set
        """
        # test configuration
        test_config = self.config["testing"]
        ckpts_to_test = test_config["ckpts"]

        for ckpt in ckpts_to_test:
            # load checkpoint
            model_path = f"{self.model_dir}/{ckpt}"
            if not os.path.exists(model_path):
                self.logger.warning(f"Checkpoint {model_path} does not exist.")
                continue

            load_checkpoint(model_path, self.model)

            # run validation
            (
                test_scores,
                test_recognition_results,
                test_translation_results,
                test_attention_scores,
            ) = validate_on_data(
                model=self.model,
                data=test_data,
                batch_size=self.config["training"]["eval_batch_size"],
                level=self.config["data"]["level"],
                gls_vocab=gls_vocab,
                txt_vocab=txt_vocab,
                recognition_beam_sizes=test_config["recognition_beam_sizes"],
                translation_beam_sizes=test_config["translation_beam_sizes"],
            )

            # save results
            self.logger.info(test_scores)
            test_name = ckpt.replace(".ckpt", "")
            for beam_size in test_config["recognition_beam_sizes"]:
                test_recognition_results.to_file(
                    f"{self.model_dir}/{test_name}.{beam_size}.rec.txt"
                )
            for beam_size in test_config["translation_beam_sizes"]:
                test_translation_results.to_file(
                    f"{self.model_dir}/{test_name}.{beam_size}.trans.txt"
                )


def train(cfg_file: str) -> None:
    """
    Main training function.

    :param cfg_file: path to configuration file
    """
    cfg = load_config(cfg_file)

    # make logger
    model_dir = make_model_dir(
        cfg["training"]["model_dir"], overwrite=cfg["training"].get("overwrite", False),
    )
    make_logger(model_dir, mode="train")
    logger = logging.getLogger(__name__)

    # copy config to model directory
    shutil.copy2(cfg_file, f"{model_dir}/config.yaml")

    # set the random seed
    set_seed(seed=cfg["training"].get("random_seed", 42))

    # load the data
    train_data, dev_data, test_data, gls_vocab, txt_vocab = load_data(
        data_cfg=cfg["data"]
    )

    # build model
    model = build_model(
        cfg=cfg, gls_vocab=gls_vocab, txt_vocab=txt_vocab,
    )
    logger.info(f"Number of trainable parameters: {model.count_params()}")

    # a trained model asks for the fields from the data
    # if model.do_recognition:
    #     model.gls_vocab = gls_vocab
    # if model.do_translation:
    #     model.txt_vocab = txt_vocab

    # create a training manager
    trainer = TrainManager(model=model, config=cfg)

    # load checkpoint if specified
    if "load_model" in cfg["training"]:
        trainer.step, trainer.epoch, trainer.best_ckpt_score = load_checkpoint(
            cfg["training"]["load_model"], trainer.model, trainer.optimizer
        )
        trainer.scheduler.set_step(trainer.step)

    # Pre-training validation
    if cfg["training"].get("validate_pre_train", False):
        logger.info("Validation before training begins...")
        trainer._validate(dev_data)

    # train the model
    trainer.train_and_validate(train_data=train_data, dev_data=dev_data)

    # test the model
    if test_data is not None:
        trainer.testing(test_data=test_data, gls_vocab=gls_vocab, txt_vocab=txt_vocab)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Joey-NMT")
    parser.add_argument(
        "config",
        default="configs/default.yaml",
        type=str,
        help="Training configuration file.",
    )
    args = parser.parse_args()
    train(cfg_file=args.config) 