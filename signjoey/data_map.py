# coding: utf-8
"""
Data module
"""
import os
import sys
import random

import numpy as np
import pandas as pd
from typing import List, Dict, Union

import torch
from torch.utils.data import Dataset
from datasets import load_dataset, DatasetDict

from signjoey.vocabulary import (
    load_vocab,
    SignVocab,
    TextVocab,
)
from signjoey.phoenix_utils.phoenix_cleanup import clean_phoenix_2014


def __load_and_filter_sents(data_cfg) -> pd.DataFrame:
    df = pd.read_csv(data_cfg["csv_root"] + f"/{data_cfg['split']}.csv", sep='\\t')
    df = df[df.SENTENCE_NAME.isin(data_cfg['include_only'])]
    df.SENTENCE = df.SENTENCE.apply(
        lambda x: clean_phoenix_2014(x, reclean=True)[1]
    )
    df = df.drop_duplicates(subset=['SENTENCE_NAME'])
    return df


def __filter_and_align_gls(example, df):
    example['SENTENCE'] = df[df.SENTENCE_NAME == example['name']].SENTENCE.values[0]
    return example


def __tokenize_and_build_vocab(
    data_cfg, datasets, gls_key, txt_key
):
    # Get combined gloss and text sentences
    gls_sentences = [d[gls_key] for d in datasets["train"]]
    txt_sentences = [d[txt_key] for d in datasets["train"]]

    # Build vocabularies
    gls_vocab = SignVocab(
        tokens=gls_sentences,
        **data_cfg["gls_vocab"]
    )
    txt_vocab = TextVocab(
        tokens=txt_sentences,
        **data_cfg["txt_vocab"]
    )
    return gls_vocab, txt_vocab


def load_data_mapped(
    data_cfg: dict,
) -> (Dataset, Dataset, Dataset, SignVocab, TextVocab):
    """
    Load data from files, create vocabulary, and prepare datasets.
    This version assumes a mapping file is available.

    :param data_cfg: data configuration
    :return:
        - train_data: training dataset
        - dev_data: development dataset
        - test_data: test dataset
        - gls_vocab: sign vocabulary
        - txt_vocab: text vocabulary
    """
    # Load the base dataset from Hugging Face
    all_splits: DatasetDict = load_dataset(
        data_cfg["hf_dataset"], trust_remote_code=True
    )
    
    # Load and process local CSV annotations
    train_df = __load_and_filter_sents(
        {"split": "train", "include_only": all_splits["train"]["name"], **data_cfg}
    )
    dev_df = __load_and_filter_sents(
        {"split": "dev", "include_only": all_splits["dev"]["name"], **data_cfg}
    )
    test_df = __load_and_filter_sents(
        {"split": "test", "include_only": all_splits["test"]["name"], **data_cfg}
    )
    
    # Filter and map datasets
    all_splits["train"] = all_splits["train"].filter(
        lambda example: example['name'] in train_df.SENTENCE_NAME.values
    ).map(lambda example: __filter_and_align_gls(example, train_df))
    all_splits["dev"] = all_splits["dev"].filter(
        lambda example: example['name'] in dev_df.SENTENCE_NAME.values
    ).map(lambda example: __filter_and_align_gls(example, dev_df))
    all_splits["test"] = all_splits["test"].filter(
        lambda example: example['name'] in test_df.SENTENCE_NAME.values
    ).map(lambda example: __filter_and_align_gls(example, test_df))


    gls_key = data_cfg.get("gls_key", "gloss")
    txt_key = data_cfg.get("txt_key", "text")

    gls_vocab, txt_vocab = __tokenize_and_build_vocab(
        data_cfg, all_splits, gls_key, txt_key
    )

    # Tokenize the datasets
    all_splits = all_splits.map(
        lambda ex: {
            "gls_ids": gls_vocab.tokens_to_ids(ex[gls_key].split()),
            "txt_ids": txt_vocab.tokens_to_ids(ex[txt_key].split()),
        },
        remove_columns=[gls_key, txt_key],
    )

    return (
        all_splits["train"],
        all_splits["dev"],
        all_splits["test"],
        gls_vocab,
        txt_vocab,
    ) 