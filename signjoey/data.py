# coding: utf-8
"""
Data module
"""
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Tuple
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
from datasets import load_dataset, DatasetDict
import os

class SignTranslationDataset(Dataset):
    """
    Load data from a Hugging Face dataset.
    Assumes the dataset has columns for sequence_key, gls_key, txt_key,
    and feature columns as specified in feature_keys.
    """

    def __init__(
        self,
        hf_split: Dataset,
        feature_keys: List[str],
        sequence_key: str,
        gls_key: str,
        txt_key: str,
        # level: str = "word",
        # max_len: int = -1,
        phase: str = "train",
    ):
        self.hf_split = hf_split
        self.feature_keys = feature_keys
        self.sequence_key = sequence_key
        self.gls_key = gls_key
        self.txt_key = txt_key
        self.phase = phase
        self.frame_subsampling_ratio = None

    def __len__(self):
        return len(self.hf_split)

    def __getitem__(self, idx: int) -> Dict:
        item = self.hf_split[idx]
        
        # Load features from their respective directories
        features = {}
        for key in self.feature_keys:
            # Assuming feature columns are named e.g., 'pose', 'hands', 'face'
            # and contain list/numpy array data.
            feature_data = np.array(item[key], dtype=np.float32)

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
            # self.lengths = [len(s[dataset.gls_key].split()) for s in self.dataset.data]
            self.lengths = [len(s[dataset.gls_key].split()) for s in self.dataset.hf_split]
        else:  # txt
            if self.dataset.level == "word":
                self.lengths = [len(s[dataset.txt_key].split()) for s in self.dataset.hf_split]
            else:
                self.lengths = [len(s[dataset.txt_key]) for s in self.dataset.hf_split]

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
    use_ddp: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[DataLoader, DistributedSampler]:
    """
    Returns a data loader for a given dataset.
    :return:
        - data_loader: torch.utils.data.DataLoader object
        - sampler: torch.utils.data.distributed.DistributedSampler object
    """
    if batch_type == "token":
        # This is not yet supported with DDP
        # ...
        pass
    else:  # sentence-based batching
        sampler = None
        if use_ddp:
            sampler = DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
            )
        
        # When using a sampler, shuffle must be False for the DataLoader
        dataloader_shuffle = shuffle and sampler is None

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=dataloader_shuffle,
            sampler=sampler,
            drop_last=False,
            collate_fn=PadCollate(
                gls_vocab=gls_vocab,
                txt_vocab=txt_vocab,
                level=level,
                txt_pad_index=txt_vocab.stoi[PAD_TOKEN],
            ),
            num_workers=num_workers,
        )
    return data_loader, sampler


def load_data(data_cfg: dict) -> (Dataset, Dataset, Dataset, GlossVocabulary, TextVocabulary):
    """
    Load keypoint data from Hugging Face datasets and merge it with
    text annotations from local CSV files.
    """
    # 1. Load the keypoint dataset from Hugging Face Hub
    hf_dataset_id = data_cfg["hf_dataset"]
    # all_splits: DatasetDict = load_dataset(hf_dataset_id)
    print(f"Loading keypoint data from Hugging Face dataset: {hf_dataset_id}")
    keypoints_ds: DatasetDict = load_dataset(hf_dataset_id)

    # 2. Load the text annotations from local CSV files
    csv_root = os.path.expanduser(data_cfg["csv_root"])
    print(f"Loading text annotations from local CSV files in: {csv_root}")
    
    def load_annotations(split_name):
        csv_path = os.path.join(csv_root, f"how2sign_realigned_{split_name}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Annotation file not found: {csv_path}")
        # Use SENTENCE_NAME as index for easy merging
        return pd.read_csv(csv_path, sep='\\t', engine='python').set_index("SENTENCE_NAME")

    train_ann = load_annotations("train")
    val_ann = load_annotations("val")
    test_ann = load_annotations("test")

    # 3. Merge text annotations into the keypoint datasets
    print("Merging text data into keypoint dataset...")
    
    def merge_datasets(keypoint_split, annotation_df):
        """
        Merge keypoint dataset with annotation dataframe.
        
        Args:
            keypoint_split: Hugging Face dataset split
            annotation_df: Pandas DataFrame with annotations
            
        Returns:
            Merged dataset
        """
        # Debug prints
        print("\n=== Debug: Dataset Structure ===")
        print("Hugging Face Dataset columns:", keypoint_split.column_names)
        print("First few keys in HF dataset:", [keypoint_split[i]['__key__'] for i in range(min(3, len(keypoint_split)))])
        print("\nAnnotation DataFrame columns:", annotation_df.columns.tolist())
        print("First few SENTENCE_NAMEs in annotation:", annotation_df['SENTENCE_NAME'].head(3).tolist())
        print("================================\n")
        
        # Original merge logic
        sentences = [annotation_df.loc[s_name].get("SENTENCE", "") for s_name in keypoint_split['__key__']]
        return keypoint_split.add_column("sentence", sentences)

    keypoints_ds["train"] = merge_datasets(keypoints_ds["train"], train_ann)
    keypoints_ds["validation"] = merge_datasets(keypoints_ds["validation"], val_ann)
    if "test" in keypoints_ds:
        keypoints_ds["test"] = merge_datasets(keypoints_ds["test"], test_ann)

    # Get config values
    feature_keys = data_cfg["feature_keys"]
    sequence_key = data_cfg["sequence_key"]
    gls_key = data_cfg["gls_key"]
    txt_key = data_cfg["txt_key"]

    # Create dataset objects for each split
    train_data = SignTranslationDataset(
        hf_split=keypoints_ds["train"],
        feature_keys=feature_keys,
        sequence_key=sequence_key,
        gls_key=gls_key,
        txt_key=txt_key,
        phase="train",
    )
    
    dev_data = SignTranslationDataset(
        hf_split=keypoints_ds["validation"],
        feature_keys=feature_keys,
        sequence_key=sequence_key,
        gls_key=gls_key,
        txt_key=txt_key,
        phase="dev",
    )
    
    test_data = None
    if "test" in keypoints_ds:
        test_data = SignTranslationDataset(
            hf_split=keypoints_ds["test"],
            feature_keys=feature_keys,
            sequence_key=sequence_key,
            gls_key=gls_key,
            txt_key=txt_key,
            phase="test",
        )

    # Build vocabularies from the training set
    # Note: build_vocab needs to be adapted if it can't handle HF datasets
    gls_vocab = build_vocab(data_cfg, dataset=train_data, vocab_type="gls")
    txt_vocab = build_vocab(data_cfg, dataset=train_data, vocab_type="txt")

    return train_data, dev_data, test_data, gls_vocab, txt_vocab

# Small helper property for vocabulary to make collate fn cleaner
Vocabulary.is_word_level = property(lambda self: self.specials == [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN])
