# coding: utf-8
"""
Data module for PHOENIX-2014T data.
This is a simplified data pipeline for pre-processed torch pickle files.
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from typing import List, Dict, Tuple

from signjoey.vocabulary import (
    GlossVocabulary,
    TextVocabulary,
    PAD_TOKEN,
    BOS_TOKEN,
)
from signjoey.batch import Batch

# ==============================================================================
# Main data loading function for PHOENIX-2014T
# ==============================================================================

def load_data_phoenix(
    data_cfg: dict,
) -> (Dataset, Dataset, Dataset, GlossVocabulary, TextVocabulary):
    """
    Load PHOENIX-2014T data from pre-processed *.pami0 files,
    create vocabulary, and prepare datasets.
    """
    data_path = data_cfg["data_path"]
    
    # Load data from pickle files
    train_data = torch.load(os.path.join(data_path, data_cfg["train"]))
    dev_data = torch.load(os.path.join(data_path, data_cfg["dev"]))
    test_data = torch.load(os.path.join(data_path, data_cfg["test"]))

    # Build vocabularies
    gls_key = data_cfg.get("gls_key", "gloss")
    txt_key = data_cfg.get("txt_key", "text")
    
    gls_sentences = [d[gls_key] for d in train_data]
    txt_sentences = [d[txt_key] for d in train_data]
    
    gls_vocab = GlossVocabulary(tokens=gls_sentences, **data_cfg.get("gls_vocab", {}))
    txt_vocab = TextVocabulary(tokens=txt_sentences, **data_cfg.get("txt_vocab", {}))

    # Create PyTorch datasets
    train_dataset = SignTranslationDataset_Phoenix(train_data, data_cfg, "train")
    dev_dataset = SignTranslationDataset_Phoenix(dev_data, data_cfg, "dev")
    test_dataset = SignTranslationDataset_Phoenix(test_data, data_cfg, "test")

    return train_dataset, dev_dataset, test_dataset, gls_vocab, txt_vocab


# ==============================================================================
# Dataset, Sampler, and Collate classes
# ==============================================================================

class SignTranslationDataset_Phoenix(Dataset):
    """
    Dataset for PHOENIX-2014T, loading data from a list of samples.
    """
    def __init__(
        self,
        data: list,
        cfg: dict,
        phase: str,
    ):
        self.data = data
        self.cfg = cfg
        self.phase = phase
        self.gls_key = cfg.get("gls_key", "gloss")
        self.txt_key = cfg.get("txt_key", "text")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        return {
            "sequence": item["name"],
            "gls": item[self.gls_key],
            "txt": item[self.txt_key],
            # Features are expected to be torch tensors already
            "features": item["sign"], 
        }

class PadCollate_Phoenix:
    """
    Custom collate function to pad sequences for the phoenix pipeline.
    """
    def __init__(self, gls_vocab: GlossVocabulary, txt_vocab: TextVocabulary):
        self.gls_vocab = gls_vocab
        self.txt_vocab = txt_vocab

    def __call__(self, batch: List[Dict]) -> Batch:
        sequences = [b["sequence"] for b in batch]
        gls_list = [b["gls"] for b in batch]
        txt_list = [b["txt"] for b in batch]
        features = [b["features"] for b in batch]

        # Use torch.nn.utils.rnn.pad_sequence for padding features
        padded_features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0.0)
        feature_lengths = torch.tensor([f.shape[0] for f in features])

        gls_ids, gls_lengths = self.gls_vocab.sentences_to_ids(gls_list)
        txt_ids, txt_lengths = self.txt_vocab.sentences_to_ids(txt_list)
        
        # Prepare txt_input for decoder (starts with BOS)
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


def make_data_iter_phoenix(
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
    Creates a data loader for a given dataset for the phoenix pipeline.
    """
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle) if use_ddp else None
    
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        drop_last=False,
        collate_fn=PadCollate_Phoenix(gls_vocab=gls_vocab, txt_vocab=txt_vocab),
        num_workers=4,
    ) 