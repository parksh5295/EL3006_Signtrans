#!/usr/bin/env python
import torch

torch.backends.cudnn.deterministic = True

import argparse
import numpy as np
import os
import shutil
import time
import queue
from typing import List, Optional, Tuple, Dict

from signjoey.model import build_model
from signjoey.batch import Batch
from signjoey.helpers import (
    log_data_info,
    load_config,
    log_cfg,
    load_checkpoint,
    make_model_dir,
    make_logger,
    set_seed,
    symlink_update,
)
from signjoey.model import SignModel
from signjoey.prediction import validate_on_data
from signjoey.loss import XentLoss
from signjoey.data import load_data, make_data_iter, SignTranslationDataset
from signjoey.builders import build_optimizer, build_scheduler, build_gradient_clipper
from signjoey.prediction import test
from signjoey.metrics import wer_single
from signjoey.vocabulary import SIL_TOKEN
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


# pylint: disable=too-many-instance-attributes
class TrainManager:
    """ Manages training loop, validations, learning rate scheduling
    and early stopping."""

    def __init__(self, model: SignModel, config: dict) -> None:
        """
        Creates a new TrainManager for a model, specified as in configuration.

        :param model: torch module defining the model
        :param config: dictionary containing the training configurations
        """
        train_config = config["training"]

        # files for logging and storing
        self.model_dir = make_model_dir(
            train_config["model_dir"], overwrite=train_config.get("overwrite", False)
        )
        self.logger = make_logger(model_dir=self.model_dir)
        self.logging_freq = train_config.get("logging_freq", 100)
        self.valid_report_file = "{}/validations.txt".format(self.model_dir)
        self.tb_writer = SummaryWriter(log_dir=self.model_dir + "/tensorboard/")

        # input
        self.feature_size = (
            sum(config["data"]["feature_size"])
            if isinstance(config["data"]["feature_size"], list)
            else config["data"]["feature_size"]
        )
        self.dataset_version = config["data"].get("version", "phoenix_2014_trans")

        # model
        self.model = model
        self.txt_pad_index = self.model.txt_pad_index
        self.txt_bos_index = self.model.txt_bos_index
        self._log_parameters_list()
        # Check if we are doing only recognition or only translation or both
        self.do_recognition = (
            config["training"].get("recognition_loss_weight", 1.0) > 0.0
        )
        self.do_translation = (
            config["training"].get("translation_loss_weight", 1.0) > 0.0
        )

        # Get Recognition and Translation specific parameters
        if self.do_recognition:
            self._get_recognition_params(train_config=train_config)
        if self.do_translation:
            self._get_translation_params(train_config=train_config)

        # optimization
        self.last_best_lr = train_config.get("learning_rate", -1)
        self.learning_rate_min = train_config.get("learning_rate_min", 1.0e-8)
        self.clip_grad_fun = build_gradient_clipper(config=train_config)
        self.optimizer = build_optimizer(
            config=train_config, parameters=model.parameters()
        )
        self.batch_multiplier = train_config.get("batch_multiplier", 1)

        # validation & early stopping
        self.validation_freq = train_config.get("validation_freq", 100)
        self.num_valid_log = train_config.get("num_valid_log", 5)
        self.ckpt_queue = queue.Queue(maxsize=train_config.get("keep_last_ckpts", 5))
        self.eval_metric = train_config.get("eval_metric", "bleu")
        if self.eval_metric not in ["bleu", "chrf", "wer", "rouge"]:
            raise ValueError(
                "Invalid setting for 'eval_metric': {}".format(self.eval_metric)
            )
        self.early_stopping_metric = train_config.get(
            "early_stopping_metric", "eval_metric"
        )

        # if we schedule after BLEU/chrf, we want to maximize it, else minimize
        # early_stopping_metric decides on how to find the early stopping point:
        # ckpts are written when there's a new high/low score for this metric
        if self.early_stopping_metric in [
            "ppl",
            "translation_loss",
            "recognition_loss",
        ]:
            self.minimize_metric = True
        elif self.early_stopping_metric == "eval_metric":
            if self.eval_metric in ["bleu", "chrf", "rouge"]:
                assert self.do_translation
                self.minimize_metric = False
            else:  # eval metric that has to get minimized (not yet implemented)
                self.minimize_metric = True
        else:
            raise ValueError(
                "Invalid setting for 'early_stopping_metric': {}".format(
                    self.early_stopping_metric
                )
            )

        # data_augmentation parameters
        self.frame_subsampling_ratio = config["data"].get(
            "frame_subsampling_ratio", None
        )
        self.random_frame_subsampling = config["data"].get(
            "random_frame_subsampling", None
        )
        self.random_frame_masking_ratio = config["data"].get(
            "random_frame_masking_ratio", None
        )

        # learning rate scheduling
        self.scheduler, self.scheduler_step_at = build_scheduler(
            config=train_config,
            scheduler_mode="min" if self.minimize_metric else "max",
            optimizer=self.optimizer,
            hidden_size=config["model"]["encoder"]["hidden_size"],
        )

        # data & batch handling
        self.level = config["data"]["level"]
        if self.level not in ["word", "bpe", "char"]:
            raise ValueError("Invalid segmentation level': {}".format(self.level))

        self.shuffle = train_config.get("shuffle", True)
        self.epochs = train_config["epochs"]
        self.batch_size = train_config["batch_size"]
        self.batch_type = train_config.get("batch_type", "sentence")
        self.eval_batch_size = train_config.get("eval_batch_size", self.batch_size)
        self.eval_batch_type = train_config.get("eval_batch_type", self.batch_type)

        self.use_cuda = train_config["use_cuda"]
        if self.use_cuda:
            self.model.cuda()
            if self.do_translation:
                self.translation_loss_function.cuda()
            if self.do_recognition:
                self.recognition_loss_function.cuda()

        # initialize training statistics
        self.steps = 0
        # stop training if this flag is True by reaching learning rate minimum
        self.stop = False
        self.total_txt_tokens = 0
        self.total_gls_tokens = 0
        self.best_ckpt_iteration = 0
        # initial values for best scores
        self.best_ckpt_score = np.inf if self.minimize_metric else -np.inf
        self.best_all_ckpt_scores = {}
        # comparison function for scores
        self.is_best = (
            lambda score: score < self.best_ckpt_score
            if self.minimize_metric
            else score > self.best_ckpt_score
        )

        # model parameters
        if "load_model" in train_config.keys():
            model_load_path = train_config["load_model"]
            self.logger.info("Loading model from %s", model_load_path)
            reset_best_ckpt = train_config.get("reset_best_ckpt", False)
            reset_scheduler = train_config.get("reset_scheduler", False)
            reset_optimizer = train_config.get("reset_optimizer", False)
            self.init_from_checkpoint(
                model_load_path,
                reset_best_ckpt=reset_best_ckpt,
                reset_scheduler=reset_scheduler,
                reset_optimizer=reset_optimizer,
            )

    def _get_recognition_params(self, train_config) -> None:
        # NOTE (Cihan): The blank label is the silence index in the gloss vocabulary.
        #   There is an assertion in the GlossVocabulary class's __init__.
        #   This is necessary to do TensorFlow decoding, as it is hardcoded
        #   Currently it is hardcoded as 0.
        self.gls_silence_token = self.model.gls_vocab.stoi[SIL_TOKEN]
        assert self.gls_silence_token == 0

        self.recognition_loss_function = torch.nn.CTCLoss(
            blank=self.gls_silence_token, zero_infinity=True
        )
        self.recognition_loss_weight = train_config.get("recognition_loss_weight", 1.0)
        self.eval_recognition_beam_size = train_config.get(
            "eval_recognition_beam_size", 1
        )

    def _get_translation_params(self, train_config) -> None:
        self.label_smoothing = train_config.get("label_smoothing", 0.0)
        self.translation_loss_function = XentLoss(
            pad_index=self.txt_pad_index, smoothing=self.label_smoothing
        )
        self.translation_normalization_mode = train_config.get(
            "translation_normalization", "batch"
        )
        if self.translation_normalization_mode not in ["batch", "tokens"]:
            raise ValueError(
                "Invalid normalization {}.".format(self.translation_normalization_mode)
            )
        self.translation_loss_weight = train_config.get("translation_loss_weight", 1.0)
        self.eval_translation_beam_size = train_config.get(
            "eval_translation_beam_size", 1
        )
        self.eval_translation_beam_alpha = train_config.get(
            "eval_translation_beam_alpha", -1
        )
        self.translation_max_output_length = train_config.get(
            "translation_max_output_length", None
        )

    def _save_checkpoint(self) -> None:
        """
        Save the model's current parameters and the training state to a
        checkpoint.

        The training state contains the total number of training steps,
        the total number of training tokens,
        the best checkpoint score and iteration so far,
        and optimizer and scheduler states.

        """
        model_path = "{}/{}.ckpt".format(self.model_dir, self.steps)
        state = {
            "steps": self.steps,
            "total_txt_tokens": self.total_txt_tokens if self.do_translation else 0,
            "total_gls_tokens": self.total_gls_tokens if self.do_recognition else 0,
            "best_ckpt_score": self.best_ckpt_score,
            "best_all_ckpt_scores": self.best_all_ckpt_scores,
            "best_ckpt_iteration": self.best_ckpt_iteration,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict()
            if self.scheduler is not None
            else None,
        }
        torch.save(state, model_path)
        if self.ckpt_queue.full():
            to_delete = self.ckpt_queue.get()  # delete oldest ckpt
            try:
                os.remove(to_delete)
            except FileNotFoundError:
                self.logger.warning(
                    "Wanted to delete old checkpoint %s but " "file does not exist.",
                    to_delete,
                )

        self.ckpt_queue.put(model_path)

        # create/modify symbolic link for best checkpoint
        symlink_update(
            "{}.ckpt".format(self.steps), "{}/best.ckpt".format(self.model_dir)
        )

    def init_from_checkpoint(
        self,
        path: str,
        reset_best_ckpt: bool = False,
        reset_scheduler: bool = False,
        reset_optimizer: bool = False,
    ) -> None:
        """
        Initialize the trainer from a given checkpoint file.

        This checkpoint file contains not only model parameters, but also
        scheduler and optimizer states, see `self._save_checkpoint`.

        :param path: path to checkpoint
        :param reset_best_ckpt: reset tracking of the best checkpoint,
                                use for domain adaptation with a new dev
                                set or when using a new metric for fine-tuning.
        :param reset_scheduler: reset the learning rate scheduler, and do not
                                use the one stored in the checkpoint.
        :param reset_optimizer: reset the optimizer, and do not use the one
                                stored in the checkpoint.
        """
        model_checkpoint = load_checkpoint(path=path, use_cuda=self.use_cuda)

        # restore model and optimizer parameters
        self.model.load_state_dict(model_checkpoint["model_state"])

        if not reset_optimizer:
            self.optimizer.load_state_dict(model_checkpoint["optimizer_state"])
        if not reset_scheduler and self.scheduler is not None:
                self.scheduler.load_state_dict(model_checkpoint["scheduler_state"])
        self.steps = model_checkpoint["steps"]
        if not reset_best_ckpt:
            self.best_ckpt_score = model_checkpoint["best_ckpt_score"]
            # self.best_all_ckpt_scores = model_checkpoint["best_all_ckpt_scores"]
            self.best_ckpt_iteration = model_checkpoint["best_ckpt_iteration"]
        if self.use_cuda:
            self.model.cuda()

    def train_and_validate(self, train_data: SignTranslationDataset, valid_data: SignTranslationDataset) -> None:
        train_iter = make_data_iter(
            dataset=train_data,
            batch_size=self.batch_size,
            gls_vocab=self.model.gls_vocab,
            txt_vocab=self.model.txt_vocab,
            sgn_dim=self.feature_size,
            level=self.level,
            train=True,
            shuffle=self.shuffle,
            frame_subsampling_ratio=self.frame_subsampling_ratio,
        )
        # epoch_no = None
        self.model.train()

        for epoch_no in range(self.epochs):
            self.logger.info("EPOCH %d", epoch_no + 1)

            if self.scheduler is not None and self.scheduler_step_at == "epoch":
                self.scheduler.step(epoch=epoch_no)

            self.model.train()
            start_time = time.time()
            
            for i, batch in enumerate(iter(train_iter)):
                batch_recognition_loss, batch_translation_loss = self._train_batch(
                    batch=batch, update=(i + 1) % self.batch_multiplier == 0
                )
                
                if self.steps % self.validation_freq == 0:
                    self.model.eval()
                    valid_scores, _, _, _, _, _, _, _ = validate_on_data(
                        model=self.model, data=valid_data, batch_size=self.eval_batch_size,
                        use_cuda=self.use_cuda, sgn_dim=self.feature_size, level=self.level,
                        gls_vocab=self.model.gls_vocab, txt_vocab=self.model.txt_vocab,
                        do_recognition=self.do_recognition, recognition_loss_function=self.recognition_loss_function,
                        recognition_loss_weight=self.recognition_loss_weight,
                        do_translation=self.do_translation, translation_loss_function=self.translation_loss_function,
                        translation_loss_weight=self.translation_loss_weight,
                        translation_max_output_length=self.translation_max_output_length,
                        txt_pad_index=self.txt_pad_index,
                        recognition_beam_size=self.eval_recognition_beam_size,
                        translation_beam_size=self.eval_translation_beam_size,
                        translation_beam_alpha=self.eval_translation_beam_alpha,
                    )
                    self.model.train()

                    if self.early_stopping_metric == "eval_metric":
                        ckpt_score = valid_scores[self.eval_metric]
                    elif self.early_stopping_metric == "ppl":
                        ckpt_score = valid_scores[-1]
                    elif self.early_stopping_metric == "translation_loss":
                        ckpt_score = valid_scores[1]
                    else:
                        ckpt_score = valid_scores[0]

                    new_best = False
                    if self.is_best(ckpt_score):
                        self.best_ckpt_score = ckpt_score
                        self.best_ckpt_iteration = self.steps
                        self.best_all_ckpt_scores = valid_scores
                        self.logger.info(
                            "Hooray! New best validation result-[%s]!",
                            self.early_stopping_metric,
                        )
                        if self.ckpt_queue.full():
                            to_delete = self.ckpt_queue.get()
                            try:
                                os.remove(to_delete)
                            except FileNotFoundError:
                                self.logger.warning("Wanted to delete old checkpoint %s but file does not exist.", to_delete)
                        checkpoint_path = "{}/{}.ckpt".format(self.model_dir, self.steps)
                            self._save_checkpoint()
                        self.ckpt_queue.put(checkpoint_path)
                        new_best = True
                        symlink_update("{}.ckpt".format(self.steps), "{}/best.ckpt".format(self.model_dir))

                    if self.scheduler is not None and self.scheduler_step_at == "validation":
                        self.scheduler.step(ckpt_score)
                        # now_lr = self.scheduler.optimizer.param_groups[0]["lr"]

                    self._add_report(
                        valid_scores=valid_scores,
                        valid_recognition_loss=valid_scores[0],
                        valid_translation_loss=valid_scores[1],
                        valid_ppl=valid_scores[-1],
                        eval_metric=self.eval_metric,
                        new_best=new_best,
                    )
            
            self.logger.info(f"Epoch {epoch_no + 1} finished in {time.time() - start_time:.2f}s")
        
        self.logger.info("Training ended.")

    def _train_batch(self, batch: Batch, update: bool = True) -> (Tensor, Tensor):
        """
        Train the model on one batch: Compute the loss, make a gradient step.

        :param batch: training batch
        :param update: if False, only store gradient. if True also make update
        :return normalized_recognition_loss: Normalized recognition loss
        :return normalized_translation_loss: Normalized translation loss
        """

        recognition_loss, translation_loss = self.model.get_loss_for_batch(
            batch=batch,
            recognition_loss_function=self.recognition_loss_function
            if self.do_recognition
            else None,
            translation_loss_function=self.translation_loss_function
            if self.do_translation
            else None,
            recognition_loss_weight=self.recognition_loss_weight
            if self.do_recognition
            else None,
            translation_loss_weight=self.translation_loss_weight
            if self.do_translation
            else None,
        )

        # normalize translation loss
        if self.do_translation:
            if self.translation_normalization_mode == "batch":
                txt_normalization_factor = batch.num_seqs
            elif self.translation_normalization_mode == "tokens":
                txt_normalization_factor = batch.num_txt_tokens
            else:
                raise NotImplementedError("Only normalize by 'batch' or 'tokens'")

            # division needed since loss.backward sums the gradients until updated
            normalized_translation_loss = translation_loss / (
                txt_normalization_factor * self.batch_multiplier
            )
        else:
            normalized_translation_loss = 0

        # TODO (Cihan): Add Gloss Token normalization (?)
        #   I think they are already being normalized by batch
        #   I need to think about if I want to normalize them by token.
        if self.do_recognition:
            normalized_recognition_loss = recognition_loss / self.batch_multiplier
        else:
            normalized_recognition_loss = 0

        total_loss = normalized_recognition_loss + normalized_translation_loss
        # compute gradients
        total_loss.backward()

        if self.clip_grad_fun is not None:
            # clip gradients (in-place)
            self.clip_grad_fun(params=self.model.parameters())

        if update:
            # make gradient step
            self.optimizer.step()
            self.optimizer.zero_grad()

            # increment step counter
            self.steps += 1

        # increment token counter
        if self.do_recognition:
            self.total_gls_tokens += batch.num_gls_tokens
        if self.do_translation:
            self.total_txt_tokens += batch.num_txt_tokens

        return normalized_recognition_loss, normalized_translation_loss

    def _add_report(
        self,
        valid_scores: Dict,
        valid_recognition_loss: float,
        valid_translation_loss: float,
        valid_ppl: float,
        eval_metric: str,
        new_best: bool = False,
    ) -> None:
        """
        Append a one-line report to validation logging file.

        :param valid_scores: Dictionary of validation scores
        :param valid_recognition_loss: validation loss (sum over whole validation set)
        :param valid_translation_loss: validation loss (sum over whole validation set)
        :param valid_ppl: validation perplexity
        :param eval_metric: evaluation metric, e.g. "bleu"
        :param new_best: whether this is a new best model
        """
        current_lr = -1
        # ignores other param groups for now
        for param_group in self.optimizer.param_groups:
            current_lr = param_group["lr"]

        if new_best:
            self.last_best_lr = current_lr

        if current_lr < self.learning_rate_min:
            self.stop = True

        with open(self.valid_report_file, "a", encoding="utf-8") as opened_file:
            opened_file.write(
                "Steps: {}\t"
                "Recognition Loss: {:.5f}\t"
                "Translation Loss: {:.5f}\t"
                "PPL: {:.5f}\t"
                "Eval Metric: {}\t"
                "WER {:.2f}\t(DEL: {:.2f},\tINS: {:.2f},\tSUB: {:.2f})\t"
                "BLEU-4 {:.2f}\t(BLEU-1: {:.2f},\tBLEU-2: {:.2f},\tBLEU-3: {:.2f},\tBLEU-4: {:.2f})\t"
                "CHRF {:.2f}\t"
                "ROUGE {:.2f}\t"
                "LR: {:.8f}\t{}\n".format(
                    self.steps,
                    valid_recognition_loss if self.do_recognition else -1,
                    valid_translation_loss if self.do_translation else -1,
                    valid_ppl if self.do_translation else -1,
                    eval_metric,
                    # WER
                    valid_scores["wer"] if self.do_recognition else -1,
                    valid_scores["wer_scores"]["del_rate"]
                    if self.do_recognition
                    else -1,
                    valid_scores["wer_scores"]["ins_rate"]
                    if self.do_recognition
                    else -1,
                    valid_scores["wer_scores"]["sub_rate"]
                    if self.do_recognition
                    else -1,
                    # BLEU
                    valid_scores["bleu"] if self.do_translation else -1,
                    valid_scores["bleu_scores"]["bleu1"] if self.do_translation else -1,
                    valid_scores["bleu_scores"]["bleu2"] if self.do_translation else -1,
                    valid_scores["bleu_scores"]["bleu3"] if self.do_translation else -1,
                    valid_scores["bleu_scores"]["bleu4"] if self.do_translation else -1,
                    # Other
                    valid_scores["chrf"] if self.do_translation else -1,
                    valid_scores["rouge"] if self.do_translation else -1,
                    current_lr,
                    "*" if new_best else "",
                )
            )

    def _log_parameters_list(self) -> None:
        """
        Write all model parameters (name, shape) to the log.
        """
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info("Total params: %d", n_params)
        trainable_params = [
            n for (n, p) in self.model.named_parameters() if p.requires_grad
        ]
        self.logger.info("Trainable parameters: %s", sorted(trainable_params))
        assert trainable_params

    def _log_examples(
        self,
        sequences: List[str],
        gls_references: List[str],
        gls_hypotheses: List[str],
        txt_references: List[str],
        txt_hypotheses: List[str],
    ) -> None:
        """
        Log `self.num_valid_log` number of samples from valid.

        :param sequences: sign video sequence names (list of strings)
        :param txt_hypotheses: decoded txt hypotheses (list of strings)
        :param txt_references: decoded txt references (list of strings)
        :param gls_hypotheses: decoded gls hypotheses (list of strings)
        :param gls_references: decoded gls references (list of strings)
        """

        if self.do_recognition:
            assert len(gls_references) == len(gls_hypotheses)
            num_sequences = len(gls_hypotheses)
        if self.do_translation:
            assert len(txt_references) == len(txt_hypotheses)
            num_sequences = len(txt_hypotheses)

        rand_idx = np.sort(np.random.permutation(num_sequences)[: self.num_valid_log])
        self.logger.info("Logging Recognition and Translation Outputs")
        self.logger.info("=" * 120)
        for ri in rand_idx:
            self.logger.info("Logging Sequence: %s", sequences[ri])
            if self.do_recognition:
                gls_res = wer_single(r=gls_references[ri], h=gls_hypotheses[ri])
                self.logger.info(
                    "\tGloss Reference :\t%s", gls_res["alignment_out"]["align_ref"]
                )
                self.logger.info(
                    "\tGloss Hypothesis:\t%s", gls_res["alignment_out"]["align_hyp"]
                )
                self.logger.info(
                    "\tGloss Alignment :\t%s", gls_res["alignment_out"]["alignment"]
                )
            if self.do_recognition and self.do_translation:
                self.logger.info("\t" + "-" * 116)
            if self.do_translation:
                txt_res = wer_single(r=txt_references[ri], h=txt_hypotheses[ri])
                self.logger.info(
                    "\tText Reference  :\t%s", txt_res["alignment_out"]["align_ref"]
                )
                self.logger.info(
                    "\tText Hypothesis :\t%s", txt_res["alignment_out"]["align_hyp"]
                )
                self.logger.info(
                    "\tText Alignment  :\t%s", txt_res["alignment_out"]["alignment"]
                )
            self.logger.info("=" * 120)

    def _store_outputs(
        self, tag: str, sequence_ids: List[str], hypotheses: List[str], sub_folder=None
    ) -> None:
        """
        Write current validation outputs to file in `self.model_dir.`

        :param hypotheses: list of strings
        """
        if sub_folder:
            out_folder = os.path.join(self.model_dir, sub_folder)
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            current_valid_output_file = "{}/{}.{}".format(out_folder, self.steps, tag)
        else:
            out_folder = self.model_dir
            current_valid_output_file = "{}/{}".format(out_folder, tag)

        with open(current_valid_output_file, "w", encoding="utf-8") as opened_file:
            for seq, hyp in zip(sequence_ids, hypotheses):
                opened_file.write("{}|{}\n".format(seq, hyp))


def train(cfg_file: str) -> None:
    """
    Main training function. After training, also test on test data if given.

    :param cfg_file: path to configuration yaml file
    """
    cfg = load_config(cfg_file)

    # set the random seed
    set_seed(seed=cfg["training"].get("random_seed", 42))

    train_data, dev_data, test_data, gls_vocab, txt_vocab = load_data(
        data_cfg=cfg["data"]
    )

    # build model and load parameters into it
    do_recognition = cfg["training"].get("recognition_loss_weight", 1.0) > 0.0
    do_translation = cfg["training"].get("translation_loss_weight", 1.0) > 0.0
    model = build_model(
        cfg=cfg["model"],
        gls_vocab=gls_vocab,
        txt_vocab=txt_vocab,
        sgn_dim=sum(cfg["data"]["feature_size"])
        if isinstance(cfg["data"]["feature_size"], list)
        else cfg["data"]["feature_size"],
        do_recognition=do_recognition,
        do_translation=do_translation,
    )

    # for training management, e.g. early stopping and model selection
    trainer = TrainManager(model=model, config=cfg)

    # store copy of original training config in model dir
    shutil.copy2(cfg_file, trainer.model_dir + "/config.yaml")

    # log all entries of config
    log_cfg(cfg, trainer.logger)

    log_data_info(
        train_data=train_data,
        valid_data=dev_data,
        test_data=test_data,
        gls_vocab=gls_vocab,
        txt_vocab=txt_vocab,
        logging_function=trainer.logger.info,
    )

    trainer.logger.info(str(model))

    # store the vocabs
    gls_vocab_file = "{}/gls.vocab".format(cfg["training"]["model_dir"])
    gls_vocab.to_file(gls_vocab_file)
    txt_vocab_file = "{}/txt.vocab".format(cfg["training"]["model_dir"])
    txt_vocab.to_file(txt_vocab_file)

    # train the model
    trainer.train_and_validate(train_data=train_data, valid_data=dev_data)
    # Delete to speed things up as we don't need training data anymore
    del train_data, dev_data, test_data

    # predict with the best model on validation and test
    # (if test data is available)
    ckpt = "{}/{}.ckpt".format(trainer.model_dir, trainer.best_ckpt_iteration)
    output_name = "best.IT_{:08d}".format(trainer.best_ckpt_iteration)
    output_path = os.path.join(trainer.model_dir, output_name)
    logger = trainer.logger
    del trainer
    test(cfg_file, ckpt=ckpt, output_path=output_path, logger=logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Joey-NMT")
    parser.add_argument(
        "config",
        default="configs/default.yaml",
        type=str,
        help="Training configuration file (yaml).",
    )
    parser.add_argument(
        "--gpu_id", type=str, default="0", help="gpu to run your job on"
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    train(cfg_file=args.config)
