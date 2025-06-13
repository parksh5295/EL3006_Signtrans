# coding: utf-8
"""
Data module
"""
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict
from signjoey.vocabulary import (
    Vocabulary,
    GlossVocabulary,
    TextVocabulary,
    UNK_TOKEN,
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
)
from signjoey.batch import Batch
import pandas as pd

class SignTranslationDataset(Dataset):
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
        
    Dataset for sign language translation, modified to handle multiple feature sets.
    """

    def __init__(
        self,
        path: str,
        sgn_dirs: Dict[str, str], # Expects a dict like {'pose': 'path/to/pose', ...}
        sequence_key: str,
        gls_key: str,
        txt_key: str,
        level: str = "word",
        max_len: int = -1,
        phase: str = "train",
    ):
        self.level = level
        self.phase = phase
        self.data = self._load_data(path, max_len)

        self.sequence_key = sequence_key
        self.gls_key = gls_key
        self.txt_key = txt_key
        
        # Store directories for different feature types
        self.sgn_dirs = sgn_dirs
        self.feature_keys = list(self.sgn_dirs.keys())

        # Frame subsampling (optional, can be set from training script)
        self.frame_subsampling_ratio = None

    def _load_data(self, path: str, max_len: int = -1):
        # with open(path, "rb") as f:
        #     data = pickle.load(f)
        """Load data from TSV file."""
        # Read TSV file
        df = pd.read_csv(path, sep="\t")
        
        # Convert DataFrame to list of dictionaries
        data = df.to_dict("records")
        
        if max_len > 0:
            return data[:max_len]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        
        # Load features from their respective directories
        features = {}
        for key in self.feature_keys:
            sgn_path = f"{self.sgn_dirs[key]}/{item[self.sequence_key]}.npy"
            try:
                feature_data = np.load(sgn_path)
            except FileNotFoundError:
                # Handle cases where a feature file might be missing if necessary
                # For now, we'll raise an error.
                raise FileNotFoundError(f"Feature file not found: {sgn_path}")

            if self.frame_subsampling_ratio and self.frame_subsampling_ratio > 1:
                feature_data = feature_data[:: self.frame_subsampling_ratio]
                
            features[key] = torch.from_numpy(feature_data).float()

        return {
            "sequence": item[self.sequence_key],
            "gls": item[self.gls_key],
            "txt": item[self.txt_key],
            "features": features,
        }


class TokenBatchSampler(Sampler):
    """
    A batch sampler that batches examples by token count.
    Note: This might not be perfectly optimal for multi-stream inputs
    as it only considers the length of the gloss or text.
    """
    def __init__(self, dataset, batch_size, type="gls", shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.type = type
        self.shuffle = shuffle
        
        # Determine lengths based on gloss or text
        if self.type == "gls":
            self.lengths = [len(s[dataset.gls_key].split()) for s in self.dataset.data]
        else:  # txt
            if self.dataset.level == "word":
                self.lengths = [len(s[dataset.txt_key].split()) for s in self.dataset.data]
            else:
                self.lengths = [len(s[dataset.txt_key]) for s in self.dataset.data]

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        
        if self.shuffle:
            np.random.shuffle(indices)

        # Sort by length to minimize padding
        indices.sort(key=lambda i: self.lengths[i])
        
        batches = []
        current_batch = []
        current_token_count = 0
        
        for i in indices:
            token_len = self.lengths[i]
            if not current_batch:
                current_batch = [i]
                current_token_count = token_len
            elif (len(current_batch) + 1) * token_len > self.batch_size:
                batches.append(current_batch)
                current_batch = [i]
                current_token_count = token_len
            else:
                current_batch.append(i)
                current_token_count += token_len
        
        if current_batch:
            batches.append(current_batch)
        
        if self.shuffle:
            np.random.shuffle(batches)
            
        for batch in batches:
            yield batch
    
    def __len__(self):
        # Provides an estimate of the number of batches
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class PadCollate:
    """
    PadCollate for multiple feature streams.
    Takes a list of samples and collates them into a batch,
    padding each feature stream independently.
    """
    def __init__(
        self,
        gls_vocab: GlossVocabulary,
        txt_vocab: TextVocabulary,
        level: str,
        txt_pad_index: int,
    ):
        self.gls_vocab = gls_vocab
        self.txt_vocab = txt_vocab
        # self.sgn_dim = sgn_dim
        self.level = level
        self.txt_pad_index = txt_pad_index

    def __call__(self, batch: List[Dict]) -> Batch:
        # Get lists of all data components
        sequences = [b["sequence"] for b in batch]
        gls_list = [b["gls"] for b in batch]
        txt_list = [b["txt"] for b in batch]
        
        # Group features by key from the list of sample dictionaries
        if not batch:
            return Batch(is_train=False)

        feature_keys = sorted(batch[0]["features"].keys())
        features_by_key = {key: [b["features"][key] for b in batch] for key in feature_keys}

        # Pad each feature stream separately and collect their original lengths
        padded_features = []
        feature_lengths = [] # To store lengths of each feature stream
        for key in feature_keys:
            stream = features_by_key[key]
            lengths = torch.tensor([s.shape[0] for s in stream])
            feature_lengths.append(lengths)
            padded_stream = pad_sequence(stream, batch_first=True, padding_value=0.0)
            padded_features.append(padded_stream)

        # Pad gloss and text
        gls, gls_lengths = self.gls_vocab.sentences_to_ids(gls_list, bpe=False)
        txt, txt_lengths = self.txt_vocab.sentences_to_ids(txt_list, bpe=(self.level == "bpe"))

        # Prepare txt_input by prepending BOS
        bos_token_id = self.txt_vocab.stoi[BOS_TOKEN]
        txt_input = torch.full((txt.shape[0], txt.shape[1] + 1), bos_token_id, dtype=torch.long)
        txt_input[:, 1:] = txt

        '''
        txt_input = txt_padded[:, :-1]
        txt_target = txt_padded[:, 1:]
        final_txt_lengths = txt_lengths - 1
        '''
        
        # Create a new Batch object
        return Batch(
            is_train=True,
            sgn=None,  # `sgn` is deprecated, use `features`
            features=padded_features,
            feature_lengths=feature_lengths, # Pass lengths to the batch
            sgn_mask=None,
            sgn_lengths=None, # `sgn_lengths` is deprecated
            gls=gls,
            gls_lengths=gls_lengths,
            txt=txt,
            txt_input=txt_input,
            txt_lengths=txt_lengths,
            txt_pad_index=self.txt_pad_index,
            sequence=sequences,
        )


def make_data_iter(
    dataset: Dataset,
    batch_size: int,
    gls_vocab: GlossVocabulary,
    txt_vocab: TextVocabulary,
    level: str,
    batch_type: str = "sentence",
    # train: bool = False,
    shuffle: bool = False,
    num_workers: int = 4,
) -> DataLoader:
    """
    Create a data loader for a dataset.
    """
    if batch_type == "token":
        batch_sampler = TokenBatchSampler(
            dataset, batch_size=batch_size, type="gls", shuffle=shuffle
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=PadCollate(
                gls_vocab=gls_vocab,
                txt_vocab=txt_vocab,
                level=level,
                txt_pad_index=txt_vocab.stoi[PAD_TOKEN],
            ),
            num_workers=num_workers,
        )
    
    # "sentence" batch_type
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        collate_fn=PadCollate(
            gls_vocab=gls_vocab,
            txt_vocab=txt_vocab,
            level=level,
            txt_pad_index=txt_vocab.stoi[PAD_TOKEN],
        ),
        num_workers=num_workers,
    )


def load_data(data_cfg: dict) -> (Dataset, Dataset, Dataset, GlossVocabulary, TextVocabulary):
    """
    Load data from files and create datasets for training, development, and testing.
    """
    # Load vocabularies
    gls_vocab_path = data_cfg.get("gls_vocab", "data/gloss.vocab")
    txt_vocab_path = data_cfg.get("txt_vocab", "data/text.vocab")
    gls_vocab = GlossVocabulary(file=gls_vocab_path)
    txt_vocab = TextVocabulary(file=txt_vocab_path)

    # Get feature directories
    sgn_dirs = data_cfg.get("sgn_dirs")
    if not sgn_dirs:
        raise ValueError("`sgn_dirs` must be specified in the data configuration.")
        
    # Get other configuration keys
    sequence_key = data_cfg.get("sequence_key", "name")
    gls_key = data_cfg.get("gls_key", "gloss")
    txt_key = data_cfg.get("txt_key", "text")
    level = data_cfg.get("level", "word")

    # Create datasets
    train_data = SignTranslationDataset(
        path=data_cfg["train"],
        sgn_dirs=sgn_dirs,
        sequence_key=sequence_key,
        gls_key=gls_key,
        txt_key=txt_key,
        level=level,
        phase="train",
    )
    
    dev_data = SignTranslationDataset(
        path=data_cfg["dev"],
        sgn_dirs=sgn_dirs,
        sequence_key=sequence_key,
        gls_key=gls_key,
        txt_key=txt_key,
        level=level,
        phase="dev",
    )
    
    test_data = SignTranslationDataset(
        path=data_cfg["test"],
        sgn_dirs=sgn_dirs,
        sequence_key=sequence_key,
        gls_key=gls_key,
        txt_key=txt_key,
        level=level,
        phase="test",
    )

    return train_data, dev_data, test_data, gls_vocab, txt_vocab

# Small helper property for vocabulary to make collate fn cleaner
Vocabulary.is_word_level = property(lambda self: self.specials == [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN])
