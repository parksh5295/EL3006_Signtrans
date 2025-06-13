#!/usr/bin/env python
import torch

torch.backends.cudnn.deterministic = True

import logging
import os
import numpy as np
import pickle as pickle
import time
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from typing import List, Optional, Tuple
from torchtext.data import Dataset
from signjoey.loss import XentLoss
from signjoey.helpers import (
    bpe_postprocess,
    load_config,
    get_latest_checkpoint,
    load_checkpoint,
    make_logger,
)
from signjoey.metrics import bleu, chrf, rouge, wer_list
from signjoey.model import build_model, SignModel
from signjoey.batch import Batch
from signjoey.data import load_data, make_data_iter, SignTranslationDataset
from signjoey.vocabulary import PAD_TOKEN, SIL_TOKEN, Vocabulary, GlossVocabulary, TextVocabulary
from signjoey.phoenix_utils.phoenix_cleanup import (
    clean_phoenix_2014,
    clean_phoenix_2014_trans,
)


# pylint: disable=too-many-arguments,too-many-locals,no-member
def validate_on_data(
    model: SignModel,
    data: SignTranslationDataset,
    batch_size: int,
    batch_type: str,
    use_cuda: bool,
    sgn_dim: int,
    level: str,
    gls_vocab: GlossVocabulary,
    txt_vocab: TextVocabulary,
    do_recognition: bool,
    recognition_loss_function: torch.nn.Module,
    recognition_loss_weight: int,
    do_translation: bool,
    translation_loss_function: torch.nn.Module,
    translation_loss_weight: int,
    translation_max_output_length: int,
    # level: str,
    txt_pad_index: int,
    recognition_beam_size: int = 1,
    translation_beam_size: int = 1,
    translation_beam_alpha: int = -1,
    # batch_type: str = "sentence",
    dataset_version: str = "phoenix_2014_trans",
) -> (
    dict,
    float,
    float,
    float,
    List[str],
    List[str],
    List[str],
    List[str],
    List[str],
):
    """
    Generate translations for the given data.
    If `loss_function` is not None and references are given,
    also compute the loss.

    :param model: model module
    :param data: dataset for validation
    :param batch_size: validation batch size
    :param batch_type: validation batch type (sentence or token)
    :param use_cuda: if True, use CUDA
    :param sgn_dim: Feature dimension of sgn frames
    :param level: segmentation level, one of "char", "bpe", "word"
    :param gls_vocab: Gloss vocabulary
    :param txt_vocab: Text vocabulary
    :param do_recognition: Flag for predicting glosses
    :param recognition_loss_function: Recognition loss function (CTC)
    :param recognition_loss_weight: CTC loss weight
    :param do_translation: Flag for predicting text
    :param translation_loss_function: Translation loss function (XEntropy)
    :param translation_loss_weight: Translation loss weight
    :param translation_max_output_length: Maximum length for generated hypotheses
    :param txt_pad_index: Txt padding token index
    :param recognition_beam_size: Beam size for recognition (CTC)
    :param translation_beam_size: Beam size for translation
    :param translation_beam_alpha: Beam search alpha for length penalty (translation)
    :param dataset_version: phoenix_2014 or phoenix_2014_trans

    :return:
        - scores: a dictionary of scores {metric: value}
        - valid_rec_loss: validation recognition loss
        - valid_tr_loss: validation translation loss
        - ppl: validation perplexity
        - sequences: sequence names
        - gls_hyp: gloss hypotheses
        - gls_ref: gloss references
        - txt_hyp: text hypotheses
        - txt_ref: text references
    """
    valid_iter = make_data_iter(
        dataset=data,
        batch_size=batch_size,
        batch_type=batch_type,
        gls_vocab=gls_vocab,
        txt_vocab=txt_vocab,
        sgn_dim=sgn_dim,
        level=level,
        shuffle=False,
        train=False,
    )

    # disable dropout
    model.eval()
    # don't track gradients during validation
    with torch.no_grad():
        '''
        all_gls_outputs = []
        all_txt_outputs = []
        all_attention_scores = []
        total_recognition_loss = 0
        total_translation_loss = 0
        total_num_txt_tokens = 0
        total_num_gls_tokens = 0
        total_num_seqs = 0
        for valid_batch in iter(valid_iter):
            batch = Batch(
                is_train=False,
                torch_batch=valid_batch,
                txt_pad_index=txt_pad_index,
                sgn_dim=sgn_dim,
                use_cuda=use_cuda,
                frame_subsampling_ratio=frame_subsampling_ratio,
            )
            sort_reverse_index = batch.sort_by_sgn_lengths()
        '''
        all_gls_hyp, all_txt_hyp = [], []
        all_gls_ref, all_txt_ref = [], []
        all_sequences = []
        total_rec_loss, total_tr_loss = 0, 0
        total_num_txt, total_num_gls = 0, 0

        '''
        batch_recognition_loss, batch_translation_loss = model.get_loss_for_batch(
            batch=batch,
            recognition_loss_function=recognition_loss_function
            if do_recognition
            else None,
            translation_loss_function=translation_loss_function
            if do_translation
            else None,
            recognition_loss_weight=recognition_loss_weight
            if do_recognition
            else None,
            translation_loss_weight=translation_loss_weight
            if do_translation
            else None,
        )
        '''

        for batch in valid_iter:
            sort_reverse_index = batch.sort_by_sgn_lengths()
            
            rec_loss, tr_loss = model.get_loss_for_batch(
                batch,
                recognition_loss_function,
                translation_loss_function,
                recognition_loss_weight,
                translation_loss_weight,
            )
            if do_recognition:
                total_rec_loss += rec_loss
                total_num_gls += batch.num_gls_tokens
            if do_translation:
                total_tr_loss += tr_loss
                total_num_txt += batch.num_txt_tokens
            '''
            (
                batch_gls_predictions,
                batch_txt_predictions,
                batch_attention_scores,
            ) = model.run_batch(
                batch=batch,
                recognition_beam_size=recognition_beam_size if do_recognition else None,
                translation_beam_size=translation_beam_size if do_translation else None,
                translation_beam_alpha=translation_beam_alpha
                if do_translation
                else None,
                translation_max_output_length=translation_max_output_length
                if do_translation
                else None,
            )
            '''
            
            gls_hyp, txt_hyp, _ = model.run_batch(
                batch,
                recognition_beam_size,
                translation_beam_size,
                translation_beam_alpha,
                translation_max_output_length,
            )

            # Re-sort to match the original order
            gls_hyp = [gls_hyp[i] for i in sort_reverse_index]
            txt_hyp = [txt_hyp[i] for i in sort_reverse_index]
            batch.sequence = [batch.sequence[i] for i in sort_reverse_index]

            all_sequences.extend(batch.sequence)

            if do_recognition:
                all_gls_hyp.extend(model.gls_vocab.arrays_to_sentences(gls_hyp))
                all_gls_ref.extend(model.gls_vocab.arrays_to_sentences(batch.gls.cpu().numpy()))

            if do_translation:
                '''
                all_txt_outputs.extend(batch_txt_predictions[sort_reverse_index])
                all_attention_scores.extend(
                    batch_attention_scores[sort_reverse_index]
                    if batch_attention_scores is not None
                    else []
                )
                '''
                all_txt_hyp.extend(model.txt_vocab.arrays_to_sentences(txt_hyp))
                all_txt_ref.extend(model.txt_vocab.arrays_to_sentences(batch.txt.cpu().numpy()))

    scores = {}
    if do_recognition:
        # Phoenix cleanup
        cln_fn = clean_phoenix_2014_trans if dataset_version == "phoenix_2014_trans" else clean_phoenix_2014
        cleaned_gls_ref = [cln_fn(" ".join(t)) for t in all_gls_ref]
        cleaned_gls_hyp = [cln_fn(" ".join(t)) for t in all_gls_hyp]
        scores.update(wer_list(references=cleaned_gls_ref, hypotheses=cleaned_gls_hyp))

    if do_translation:
        join_char = " " if level in ["word", "bpe"] else ""
        txt_ref = [join_char.join(t) for t in all_txt_ref]
        txt_hyp = [join_char.join(t) for t in all_txt_hyp]
        if level == "bpe":
            txt_ref = [bpe_postprocess(v) for v in txt_ref]
            txt_hyp = [bpe_postprocess(v) for v in txt_hyp]
        scores["bleu"] = bleu(references=txt_ref, hypotheses=txt_hyp)
        scores["chrf"] = chrf(references=txt_ref, hypotheses=txt_hyp)
        scores["rouge"] = rouge(references=txt_ref, hypotheses=txt_hyp)

    valid_rec_loss = total_rec_loss / total_num_gls if do_recognition and total_num_gls > 0 else -1
    valid_tr_loss = total_tr_loss / total_num_txt if do_translation and total_num_txt > 0 else -1
    ppl = torch.exp(torch.tensor(valid_tr_loss)) if do_translation and valid_tr_loss > -1 else -1

    return (
        scores,
        valid_rec_loss,
        valid_tr_loss,
        ppl,
        all_sequences,
        all_gls_hyp,
        all_gls_ref,
        all_txt_hyp,
        all_txt_ref,
    )


def _write_to_file(file_path: str, sequence_ids: List[str], hypotheses: List[str]):
    with open(file_path, mode="w", encoding="utf-8") as out_file:
        for seq, hyp in zip(sequence_ids, hypotheses):
            out_file.write(seq + "|" + " ".join(hyp) + "\n")


# pylint: disable-msg=logging-too-many-args
def test(
    cfg_file, ckpt: str, output_path: str = None, logger: logging.Logger = None
) -> None:
    """
    Main test function. Handles loading a model from checkpoint, generating
    translations and storing them and metrics to disk.
    :param cfg_file: path to configuration file
    :param ckpt: path to checkpoint to load
    :param output_path: path to output
    :param logger: log output to this logger (otherwise create new logger)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            FORMAT = "%(asctime)-15s - %(message)s"
            logging.basicConfig(format=FORMAT, level=logging.DEBUG)

    cfg = load_config(cfg_file)
    '''
    test_data_cfg = cfg["data"]["test"]

    # load the data
    train_data, dev_data, test_data, gls_vocab, txt_vocab = load_data(
        data_cfg=cfg["data"]
    )
    data_to_test = {"dev": dev_data, "test": test_data}

    # build model and load parameters
    model = build_model(
        cfg=cfg["model"],
        gls_vocab=gls_vocab,
        txt_vocab=txt_vocab,
        sgn_dim=sum(cfg["data"]["feature_size"]) if isinstance(cfg["data"]["feature_size"], list) else cfg["data"]["feature_size"],
        do_recognition=cfg["training"]["recognition_loss_weight"] > 0,
        do_translation=cfg["training"]["translation_loss_weight"] > 0,
    )
    '''

    # when checkpoint is not specified, take latest (best) from model dir
    if ckpt is None:
        model_dir = cfg["training"]["model_dir"]
        ckpt = get_latest_checkpoint(model_dir)
        if ckpt is None:
            raise FileNotFoundError(
                "No checkpoint found in directory {}.".format(model_dir)
            )

    # batch_size = cfg["training"]["batch_size"]
    # batch_type = cfg["training"].get("batch_type", "sentence")
    use_cuda = cfg["training"].get("use_cuda", False)
    # level = cfg["data"]["level"]

    # load the data
    _, dev_data, test_data, gls_vocab, txt_vocab = load_data(
        data_cfg=cfg["data"]
    )
    
    # build model
    do_recognition = cfg["training"]["recognition_loss_weight"] > 0
    do_translation = cfg["training"]["translation_loss_weight"] > 0
    model = build_model(
        cfg=cfg["model"],
        gls_vocab=gls_vocab,
        txt_vocab=txt_vocab,
        sgn_dim=sum(cfg["data"]["feature_size"]) if isinstance(cfg["data"]["feature_size"], list) else cfg["data"]["feature_size"],
        do_recognition=do_recognition,
        do_translation=do_translation,
    )

    # load model state from disk
    model_checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda)
    model.load_state_dict(model_checkpoint["model_state"])

    if use_cuda:
        model.cuda()

    # Get loss functions
    recognition_loss_function = (
        nn.CTCLoss(blank=gls_vocab.stoi[SIL_TOKEN], zero_infinity=True)
        if do_recognition
        else None
    )
    translation_loss_function = (
        XentLoss(pad_index=txt_vocab.stoi[PAD_TOKEN], smoothing=0.0)
        if do_translation
        else None
        )
        if use_cuda:
        if recognition_loss_function:
            recognition_loss_function.cuda()
        if translation_loss_function:
            translation_loss_function.cuda()

    # Get Hyper-parameters for testing
    testing_cfg = cfg.get("testing", {})
    
    # Frame subsampling
    frame_subsampling_ratio = testing_cfg.get("frame_subsampling_ratio")
    if frame_subsampling_ratio:
        if dev_data:
            dev_data.frame_subsampling_ratio = frame_subsampling_ratio
        if test_data:
            test_data.frame_subsampling_ratio = frame_subsampling_ratio

    # Batching
    batch_size = cfg["training"]["batch_size"]
    batch_type = cfg["training"].get("batch_type", "sentence")

    recognition_beam_sizes = testing_cfg.get("recognition_beam_sizes", [1])
    translation_beam_sizes = testing_cfg.get("translation_beam_sizes", [1])
    translation_beam_alphas = testing_cfg.get("translation_beam_alphas", [-1])

    max_recognition_beam_size = testing_cfg.get("max_recognition_beam_size")
    if max_recognition_beam_size is not None:
        recognition_beam_sizes = list(range(1, max_recognition_beam_size + 1))

    # Data to test
    data_to_test = {"dev": dev_data, "test": test_data}
    if "data_to_test" in testing_cfg:
        data_to_test = {
            k: v for k,v in data_to_test.items() if k in testing_cfg["data_to_test"]
        }

    for rec_beam_size in recognition_beam_sizes:
        for tr_beam_size in translation_beam_sizes:
            for tr_alpha in translation_beam_alphas:
                if not do_recognition and rec_beam_size > 1:
                    continue
                if not do_translation and (tr_beam_size > 1 or tr_alpha > -1):
                    continue

                logger.info(
                    "START TESTING - rec_bs: %d, tr_bs: %d, tr_alpha: %d",
                    rec_beam_size,
                    tr_beam_size,
                    tr_alpha
                )

                for data_set_name, data_set in data_to_test.items():
                    if data_set is None:
                        continue
                        
                    logger.info("Testing on %s set...", data_set_name)
                    
                    (
                        scores,
                        recognition_loss,
                        translation_loss,
                        ppl,
                        sequences,
                        gls_hyp,
                        gls_ref,
                        txt_hyp,
                        txt_ref,
                    ) = validate_on_data(
                    model=model,
                        data=data_set,
                    batch_size=batch_size,
                        batch_type=batch_type,
                    use_cuda=use_cuda,
                        level=cfg["data"]["level"],
                        sgn_dim=model.sgn_embed.input_size,
                        txt_pad_index=model.txt_pad_index,
                        gls_vocab=gls_vocab,
                        txt_vocab=txt_vocab,
                    do_recognition=do_recognition,
                        recognition_loss_function=recognition_loss_function,
                        recognition_loss_weight=cfg["training"]["recognition_loss_weight"],
                        recognition_beam_size=rec_beam_size,
                    do_translation=do_translation,
                    translation_loss_function=translation_loss_function,
                        translation_loss_weight=cfg["training"]["translation_loss_weight"],
                        translation_max_output_length=cfg["training"].get("translation_max_output_length", None),
                        translation_beam_size=tr_beam_size,
                        translation_beam_alpha=tr_alpha,
                        dataset_version=cfg["data"].get("version", "phoenix_2014_trans"),
                    )
                    
                    logger.info(
                        "%s Rec. Loss: %6.2f\tTr. Loss: %6.2f\tPPL: %6.2f",
                        data_set_name.upper(),
                        recognition_loss,
                        translation_loss,
                        ppl,
                    )
                    logger.info("Scores on %s set: %s", data_set_name, scores)

                    if output_path is not None:
                        if do_recognition:
                            rec_output_path = "{}.{}.BW_{:03d}.gls".format(
                                output_path, data_set_name, rec_beam_size
                            )
                            _write_to_file(rec_output_path, sequences, gls_hyp)

                        if do_translation:
                            tr_output_path = "{}.{}.BW_{:02d}.A_{:.1f}.txt".format(
                                output_path, data_set_name, tr_beam_size, tr_alpha,
                            )
                            _write_to_file(tr_output_path, sequences, txt_hyp)

                        results_file = "{}.{}.BW_{:03d}_{:02d}_{:.1f}_results.pkl".format(
                            output_path, data_set_name, rec_beam_size, tr_beam_size, tr_alpha
                        )
                        with open(results_file, "wb") as out:
                            pickle.dump(scores, out)
                        logger.info("Saved results to %s.", results_file)

                logger.info("END TESTING")
