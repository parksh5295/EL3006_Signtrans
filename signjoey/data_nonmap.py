# coding: utf-8
"""
Data module for direct text-to-pose processing without complex mapping files.
This is the new data pipeline.
"""
import os
import pandas as pd
from typing import List, Dict, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset, DatasetDict, concatenate_datasets
import numpy as np

from signjoey.vocabulary import GlossVocabulary, TextVocabulary, BOS_TOKEN, PAD_TOKEN
from signjoey.batch import Batch

def load_data_nonmap(data_cfg: dict) -> (Dataset, Dataset, Dataset, GlossVocabulary, TextVocabulary):
    """
    Loads and merges keypoint and annotation data for direct text-to-pose models.
    """
    print("--- Using NEW data loader (data_nonmap.py) ---")

    # 1. Load remote keypoint dataset from Hugging Face
    print("Loading keypoint data from Hugging Face...")
    keypoint_ds = load_dataset(data_cfg["hf_keypoint_dataset"], trust_remote_code=True)

    # 2. Load local CSV annotations
    print("Loading local CSV annotations...")
    csv_root = data_cfg["csv_root"]
    train_csv = pd.read_csv(os.path.join(csv_root, "how2sign_realigned_train.csv"), sep='\\t', engine='python')
    val_csv = pd.read_csv(os.path.join(csv_root, "how2sign_realigned_val.csv"), sep='\\t', engine='python')
    test_csv = pd.read_csv(os.path.join(csv_root, "how2sign_realigned_test.csv"), sep='\\t', engine='python')

    # Function to prepare and merge a split
    def prepare_split(hf_split, csv_df):
        # Normalize keypoint filename to match CSV's SENTENCE_NAME
        hf_split = hf_split.map(lambda x: {"SENTENCE_NAME": x['file_name'].replace('.npy', '')})
        
        # Convert pandas DataFrame to Hugging Face Dataset
        csv_ds = Dataset.from_pandas(csv_df)
        
        # Merge (join) the two datasets on the 'SENTENCE_NAME' column
        # This adds the SENTENCE and time info to the keypoint data
        merged_ds = hf_split.map(
            lambda x: csv_ds.filter(lambda y: y['SENTENCE_NAME'] == x['SENTENCE_NAME'])[0],
            remove_columns=hf_split.column_names # Keep only columns from the CSV side after lookup
        )
        # Manually concatenate the original features back
        merged_ds = concatenate_datasets([merged_ds, hf_split], axis=1)

        return merged_ds

    print("Preparing and merging splits...")
    train_data = prepare_split(keypoint_ds['train'], train_csv)
    dev_data = prepare_split(keypoint_ds['validation'], val_csv)
    test_data = prepare_split(keypoint_ds['test'], test_csv)

    # 3. Build vocabularies
    print("Building vocabularies...")
    txt_key = data_cfg["txt_key"]
    gls_key = data_cfg["gls_key"]

    txt_vocab = TextVocabulary(tokens=train_data[txt_key], **data_cfg["txt_vocab"])
    gls_vocab = GlossVocabulary(tokens=train_data[gls_key], **data_cfg["gls_vocab"])
    
    # 4. Create final PyTorch datasets
    train_dataset = SignTranslationDataset_NonMap(train_data, data_cfg, txt_vocab, gls_vocab)
    dev_dataset = SignTranslationDataset_NonMap(dev_data, data_cfg, txt_vocab, gls_vocab)
    test_dataset = SignTranslationDataset_NonMap(test_data, data_cfg, txt_vocab, gls_vocab)

    return train_dataset, dev_dataset, test_dataset, gls_vocab, txt_vocab


class SignTranslationDataset_NonMap(Dataset):
    """
    Dataset for direct text-to-pose translation.
    """
    def __init__(self, data: Dataset, data_cfg: dict, txt_vocab: TextVocabulary, gls_vocab: GlossVocabulary):
        self.data = data
        self.data_cfg = data_cfg
        self.txt_vocab = txt_vocab
        self.gls_vocab = gls_vocab
        
        self.sequence_key = data_cfg["sequence_key"]
        self.gls_key = data_cfg["gls_key"]
        self.txt_key = data_cfg["txt_key"]
        self.feature_keys = data_cfg["feature_keys"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        entry = self.data[idx]
        
        # Combine all features into a single tensor
        features = np.concatenate([entry[key] for key in self.feature_keys], axis=-1)
        
        return {
            "sequence": entry[self.sequence_key],
            "gls": entry[self.gls_key],
            "txt": entry[self.txt_key],
            "features": torch.from_numpy(features).float(),
        }


class PadCollate_NonMap:
    """
    Custom collate function to pad sequences for the non-map pipeline.
    """
    def __init__(self, gls_vocab: GlossVocabulary, txt_vocab: TextVocabulary):
        self.gls_vocab = gls_vocab
        self.txt_vocab = txt_vocab

    def __call__(self, batch: List[Dict]) -> Batch:
        sequences = [b["sequence"] for b in batch]
        gls_list = [b["gls"] for b in batch]
        txt_list = [b["txt"] for b in batch]
        features = [b["features"] for b in batch]

        padded_features = pad_sequence(features, batch_first=True, padding_value=0.0)
        feature_lengths = torch.tensor([f.shape[0] for f in features])

        gls_ids, gls_lengths = self.gls_vocab.sentences_to_ids(gls_list)
        txt_ids, txt_lengths = self.txt_vocab.sentences_to_ids(txt_list)
        
        txt_input = torch.full((txt_ids.shape[0], txt_ids.shape[1] + 1), self.txt_vocab.stoi[BOS_TOKEN], dtype=torch.long)
        txt_input[:, 1:] = txt_ids

        return Batch(
            is_train=True,
            features=[padded_features],
            feature_lengths=[feature_lengths],
            sgn=None, sgn_mask=None, sgn_lengths=None, # Not used in this pipeline
            gls=gls_ids,
            gls_lengths=gls_lengths,
            txt=txt_ids,
            txt_input=txt_input,
            txt_lengths=txt_lengths,
            txt_pad_index=self.txt_vocab.stoi[PAD_TOKEN],
            sequence=sequences,
        )


def make_data_iter_nonmap(
    dataset: Dataset,
    batch_size: int,
    gls_vocab: GlossVocabulary,
    txt_vocab: TextVocabulary,
    shuffle: bool = False,
    use_ddp: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """
    Creates a data loader for a given dataset for the non-map pipeline.
    """
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle) if use_ddp else None
    
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        drop_last=False,
        collate_fn=PadCollate_NonMap(gls_vocab=gls_vocab, txt_vocab=txt_vocab),
        num_workers=4,
    ) 