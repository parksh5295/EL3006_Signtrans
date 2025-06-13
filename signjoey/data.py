# coding: utf-8
"""
Data module
"""
import os
import sys
import random
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torch.nn.utils.rnn import pad_sequence
from signjoey.dataset import SignTranslationDataset
from signjoey.vocabulary import (
    build_vocab,
    Vocabulary,
    UNK_TOKEN,
    EOS_TOKEN,
    BOS_TOKEN,
    PAD_TOKEN,
)
from signjoey.batch import Batch
import pickle
import numpy as np
from torch.utils.data.sampler import Sampler
from typing import List


def load_data(data_cfg: dict) -> (SignTranslationDataset, SignTranslationDataset, SignTranslationDataset, Vocabulary, Vocabulary):
    """
    Load train, dev and optionally test data as specified in configuration.
    Vocabularies are created from the training set with a limit of `voc_limit`
    tokens and a minimum token frequency of `voc_min_freq`
    (specified in the configuration dictionary).

    The training data is filtered to include sentences up to `max_sent_length`
    on source and target side.

    If you set ``random_train_subset``, a random selection of this size is used
    from the training set instead of the full training set.

    If you set ``random_dev_subset``, a random selection of this size is used
    from the dev development instead of the full development set.

    :param data_cfg: configuration dictionary for data
        ("data" part of configuration file)
    :return:
        - train_data: training dataset
        - dev_data: development dataset
        - test_data: test dataset if given, otherwise None
        - gls_vocab: gloss vocabulary extracted from training data
        - txt_vocab: spoken text vocabulary extracted from training data
    """

    data_path = data_cfg.get("data_path", "./data")

    if isinstance(data_cfg["train"], list):
        train_paths = [os.path.join(data_path, x) for x in data_cfg["train"]]
        dev_paths = [os.path.join(data_path, x) for x in data_cfg["dev"]]
        test_paths = [os.path.join(data_path, x) for x in data_cfg["test"]]
        # pad_feature_size = sum(data_cfg["feature_size"])

    else:
        train_paths = os.path.join(data_path, data_cfg["train"])
        dev_paths = os.path.join(data_path, data_cfg["dev"])
        test_paths = os.path.join(data_path, data_cfg["test"])
        # pad_feature_size = data_cfg["feature_size"]

    level = data_cfg["level"]
    txt_lowercase = data_cfg["txt_lowercase"]
    # max_sent_length = data_cfg["max_sent_length"]

    def tokenize_text(text):
        if level == "char":
            return list(text)
        else:
            return text.split()

    train_data = SignTranslationDataset(path=train_paths)
    
    # Handle random subsets
    random_train_subset = data_cfg.get("random_train_subset", -1)
    if random_train_subset > -1:
        subset_indices = random.sample(range(len(train_data)), random_train_subset)
        train_data = Subset(train_data, subset_indices)

    '''
    sequence_field = data.RawField()
    signer_field = data.RawField()

    sgn_field = data.Field(
        use_vocab=False,
        init_token=None,
        dtype=torch.float32,
        preprocessing=tokenize_features,
        tokenize=lambda features: features,  # TODO (Cihan): is this necessary?
        batch_first=True,
        include_lengths=True,
        postprocessing=stack_features,
        pad_token=torch.zeros((pad_feature_size,)),
    )

    gls_field = data.Field(
        pad_token=PAD_TOKEN,
        tokenize=tokenize_text,
        batch_first=True,
        lower=False,
        include_lengths=True,
    )

    txt_field = data.Field(
        init_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        tokenize=tokenize_text,
        unk_token=UNK_TOKEN,
        batch_first=True,
        lower=txt_lowercase,
        include_lengths=True,
    )

    train_data = SignTranslationDataset(
        path=train_paths,
        fields=(sequence_field, signer_field, sgn_field, gls_field, txt_field),
        filter_pred=lambda x: len(vars(x)["sgn"]) <= max_sent_length
        and len(vars(x)["txt"]) <= max_sent_length,
    )
    '''

    # Build vocabulary from training data
    gls_max_size = data_cfg.get("gls_voc_limit", sys.maxsize)
    gls_min_freq = data_cfg.get("gls_voc_min_freq", 1)
    txt_max_size = data_cfg.get("txt_voc_limit", sys.maxsize)
    txt_min_freq = data_cfg.get("txt_voc_min_freq", 1)

    gls_vocab_file = data_cfg.get("gls_vocab", None)
    txt_vocab_file = data_cfg.get("txt_vocab", None)
    
    # Use the Subset's underlying dataset for vocabulary building if it's a subset
    vocab_data_source = train_data.dataset if isinstance(train_data, Subset) else train_data
    gls_corpus = [tokenize_text(sample['gloss']) for sample in vocab_data_source.samples]
    txt_corpus = [tokenize_text(sample['text'].lower() if txt_lowercase else sample['text']) for sample in vocab_data_source.samples]

    gls_vocab = build_vocab(
        field="gls",
        min_freq=gls_min_freq,
        max_size=gls_max_size,
        # dataset=train_data,
        vocab_file=gls_vocab_file,
        sentences=gls_corpus
    )
    txt_vocab = build_vocab(
        field="txt",
        min_freq=txt_min_freq,
        max_size=txt_max_size,
        # dataset=train_data,
        vocab_file=txt_vocab_file,
        sentences=txt_corpus
    )
    '''
    random_train_subset = data_cfg.get("random_train_subset", -1)
    if random_train_subset > -1:
        # select this many training examples randomly and discard the rest
        keep_ratio = random_train_subset / len(train_data)
        keep, _ = train_data.split(
            split_ratio=[keep_ratio, 1 - keep_ratio], random_state=random.getstate()
        )
        train_data = keep

    dev_data = SignTranslationDataset(
        path=dev_paths,
        fields=(sequence_field, signer_field, sgn_field, gls_field, txt_field),
    )
    '''

    dev_data = SignTranslationDataset(path=dev_paths)
    random_dev_subset = data_cfg.get("random_dev_subset", -1)
    if random_dev_subset > -1:
        subset_indices = random.sample(range(len(dev_data)), random_dev_subset)
        dev_data = Subset(dev_data, subset_indices)
        
    test_data = SignTranslationDataset(path=test_paths)

    # gls_field.vocab = gls_vocab
    # txt_field.vocab = txt_vocab
    return train_data, dev_data, test_data, gls_vocab, txt_vocab

'''
# TODO (Cihan): I don't like this use of globals.
#  Need to find a more elegant solution for this it at some point.
# pylint: disable=global-at-module-level
global max_sgn_in_batch, max_gls_in_batch, max_txt_in_batch
'''

class TokenBatchSampler(Sampler):
    """
    A batch sampler that batches examples by token count.
    """
    def __init__(self, dataset, batch_size, type="gls", shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.type = type
        self.shuffle = shuffle
        
        if self.type == "gls":
            self.lengths = [len(s["gloss"].split()) for s in self.dataset.data]
        else: # txt
            if self.dataset.level == "word":
                self.lengths = [len(s["text"].split()) for s in self.dataset.data]
            else:
                self.lengths = [len(s["text"]) for s in self.dataset.data]

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        
        if self.shuffle:
            # Shuffle batters, but sort within a batch to minimize padding
            np.random.shuffle(indices)

        # Sort by length
        indices.sort(key=lambda i: self.lengths[i])
        
        batches = []
        current_batch = []
        current_token_count = 0
        
        for i in indices:
            if not current_batch:
                 current_batch = [i]
                 current_token_count = self.lengths[i]
            elif current_token_count + self.lengths[i] > self.batch_size:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [i]
                current_token_count = self.lengths[i]
            else:
                current_batch.append(i)
                current_token_count += self.lengths[i]
        
        if current_batch:
            batches.append(current_batch)
        
        if self.shuffle:
            np.random.shuffle(batches)
            
        for batch in batches:
            yield batch
    
    def __len__(self):
        # This is an estimation
        return len(list(self.__iter__()))


class PadCollate:
    def __init__(self, gls_vocab, txt_vocab, sgn_dim, level, txt_pad_index=1):
        self.gls_vocab = gls_vocab
        self.txt_vocab = txt_vocab
        self.sgn_dim = sgn_dim
        self.txt_pad_index = txt_pad_index
        self.level = level

    def __call__(self, batch):
        names = [item['name'] for item in batch]
        signers = [item['signer'] for item in batch]
        
        s_list = [item['sign'] for item in batch]
        s_lengths = torch.tensor([s.shape[0] for s in s_list])
        s_padded = pad_sequence(s_list, batch_first=True, padding_value=0)

        def tokenize(text, vocab):
            # Gloss is always word-level, text level is configurable
            tokens = text.split() if vocab.is_word_level else list(text)
            return [vocab.stoi[BOS_TOKEN]] + [vocab.stoi.get(t, vocab.stoi[UNK_TOKEN]) for t in tokens] + [vocab.stoi[EOS_TOKEN]]

        gls_list = [torch.tensor(tokenize(item['gloss'], self.gls_vocab)) for item in batch]
        gls_lengths = torch.tensor([len(gls) for gls in gls_list])
        gls_padded = pad_sequence(gls_list, batch_first=True, padding_value=self.gls_vocab.stoi[PAD_TOKEN])
        
        txt_list = [torch.tensor(tokenize(item['text'], self.txt_vocab)) for item in batch]
        txt_lengths = torch.tensor([len(txt) for txt in txt_list])
        txt_padded = pad_sequence(txt_list, batch_first=True, padding_value=self.txt_vocab.stoi[PAD_TOKEN])
        
        txt_input = txt_padded[:, :-1]
        txt_target = txt_padded[:, 1:]
        final_txt_lengths = txt_lengths - 1

        return Batch(
            sgn=s_padded,
            sgn_lengths=s_lengths,
            gls=gls_padded,
            gls_lengths=gls_lengths,
            txt=txt_target,
            txt_input=txt_input,
            txt_lengths=final_txt_lengths,
            sequence=names,
            signer=signers,
            use_cuda=torch.cuda.is_available(),
            txt_pad_index=self.txt_pad_index,
            sgn_dim=self.sgn_dim,
        )


def make_data_iter(
    dataset: Dataset,
    batch_size: int,
    gls_vocab: Vocabulary,
    txt_vocab: Vocabulary,
    sgn_dim: int,
    level: str,
    batch_type: str = "sentence",
    # train: bool = False,
    shuffle: bool = False,
) -> DataLoader:
    """
    Returns a data loader for a given dataset.
    :param dataset: torch dataset
    :param batch_size: batch size
    :param gls_vocab: gloss vocabulary
    :param txt_vocab: text vocabulary
    :param sgn_dim: sign feature dimension
    :param level: segmentation level
    :param batch_type: "sentence" or "token"
    :param shuffle: whether to shuffle the data
    :return: torch data loader
    """

    collate_fn = PadCollate(
        gls_vocab=gls_vocab,
        txt_vocab=txt_vocab,
        sgn_dim=sgn_dim,
        level=level,
        txt_pad_index=txt_vocab.stoi[PAD_TOKEN],
    )

    if batch_type == "token":
        batch_sampler = TokenBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            type="gls", # The old system batched by gloss tokens
            shuffle=shuffle
        )
        return DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=4,
        )
    
    # "sentence" batching
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=shuffle,
        num_workers=4,
    )

# Small helper property for vocabulary to make collate fn cleaner
Vocabulary.is_word_level = property(lambda self: self.specials == [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN])
