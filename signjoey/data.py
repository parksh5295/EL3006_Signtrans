# coding: utf-8
"""
Data module
"""
# import pickle
import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Tuple
from datasets import load_dataset, DatasetDict
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

# ==============================================================================
# Helper functions for data loading and merging
# ==============================================================================

def get_lookup_dict(annotation_df: pd.DataFrame) -> Dict[str, str]:
    """Creates a dictionary to look up sentences by a normalized video name."""
    lookup = {}
    for _, row in annotation_df.iterrows():
        video_name = row.get('VIDEO_NAME')
        if video_name and isinstance(video_name, str):
            # Normalize the key from the CSV, e.g., '--7E2sU6zP4-5-rgb_front' -> '--7E2sU6zP4-5'
            normalized_key = video_name.split('-rgb_front')[0]
            lookup[normalized_key] = row['SENTENCE']
    return lookup

def merge_datasets(keypoint_split: Dataset, annotation_df: pd.DataFrame) -> Dataset:
    """
    Merge keypoint dataset with annotation dataframe using a robust
    normalized key matching strategy.
    """
    # sentence_lookup = get_lookup_dict(annotation_df)
    
    def get_normalized_key_from_hf(hf_key: str) -> str:
        # Extracts and normalizes the key from Hugging Face __key__
        # e.g., '.../--7E2sU6zP4_5-rgb_front/...' -> '--7E2sU6zP4-5'
        dir_name = os.path.dirname(hf_key).split('/')[-1]
        if len(dir_name) > 11 and dir_name[11] == '_':
            part1 = dir_name[:11]
            part2 = dir_name[12:].split('-rgb_front')[0]
            return f"{part1}-{part2}"
        # Fallback for keys that don't match the expected pattern
        return dir_name.split('-rgb_front')[0]

    sentence_lookup = get_lookup_dict(annotation_df)
    
    # --- NEW DEBUG ---
    print("\n--- Debug: Key Matching ---")
    hf_keys_sample = [
        get_normalized_key_from_hf(item['__key__']) 
        for item in keypoint_split.select(range(min(5, len(keypoint_split))))
    ]
    csv_keys_sample = list(sentence_lookup.keys())[:5]
    print(f"Sample Normalized HF Keys: {hf_keys_sample}")
    print(f"Sample Annotation Keys:    {csv_keys_sample}")
    print("---------------------------\n")
    # --- END DEBUG ---

    sentences = []
    unmatched_count = 0
    for item in keypoint_split:
        hf_key = item['__key__']
        normalized_key = get_normalized_key_from_hf(hf_key)
        
        sentence = sentence_lookup.get(normalized_key)
        if sentence is None:
            unmatched_count += 1
            sentence = ""  # Use empty string for unmatched entries
        
        sentences.append(sentence)
        
    if unmatched_count > 0:
        print(
            f"Warning: {unmatched_count} out of {len(keypoint_split)} entries "
            f"could not be matched with annotations and were assigned an empty sentence."
        )
        
    return keypoint_split.add_column("SENTENCE", sentences)

def load_annotations(csv_root: str, split_name: str) -> pd.DataFrame:
    """Loads a CSV annotation file into a pandas DataFrame."""
    csv_path = os.path.join(csv_root, f"how2sign_realigned_{split_name}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Annotation file not found: {csv_path}")
    return pd.read_csv(csv_path, sep='\\t', engine='python')

# ==============================================================================
# Main data loading function
# ==============================================================================

def load_data(data_cfg: dict) -> (Dataset, Dataset, Dataset, GlossVocabulary, TextVocabulary):
    """
    Load keypoint data from Hugging Face datasets and merge it with
    text annotations from local CSV files.
    """
    # 1. Load the keypoint dataset from Hugging Face Hub
    hf_dataset_id = data_cfg["hf_dataset"]
    print(f"Loading keypoint data from Hugging Face dataset: {hf_dataset_id}")
    keypoints_ds: DatasetDict = load_dataset(hf_dataset_id)

    # 2. Load the text annotations from local CSV files
    csv_root = os.path.expanduser(data_cfg["csv_root"])
    print(f"Loading text annotations from local CSV files in: {csv_root}")
    
    train_ann = load_annotations(csv_root, "train")
    val_ann = load_annotations(csv_root, "val")
    test_ann = load_annotations(csv_root, "test")

    # 3. Merge text annotations into the keypoint datasets
    print("Merging text data into keypoint dataset...")
    keypoints_ds["train"] = merge_datasets(keypoints_ds["train"], train_ann)
    keypoints_ds["validation"] = merge_datasets(keypoints_ds["validation"], val_ann)
    if "test" in keypoints_ds:
        keypoints_ds["test"] = merge_datasets(keypoints_ds["test"], test_ann)

    # 4. Create SignTranslationDataset objects for each split
    feature_keys = data_cfg["feature_keys"]
    sequence_key = data_cfg["sequence_key"]
    # The 'gls' and 'txt' keys now refer to the 'SENTENCE' column we just added
    gls_key = "SENTENCE"
    txt_key = "SENTENCE"

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

    # 5. Build vocabularies from the training set
    print("Building vocabularies...")
    gls_vocab = build_vocab(data_cfg, dataset=train_data, vocab_type="gls")
    txt_vocab = build_vocab(data_cfg, dataset=train_data, vocab_type="txt")

    return train_data, dev_data, test_data, gls_vocab, txt_vocab

# ==============================================================================
# build_vocab, Dataset, Sampler, and Collate classes (mostly unchanged)
# ==============================================================================

def build_vocab(cfg, dataset, vocab_type="gls"):
    """
    Builds a vocabulary from the dataset.
    Assumes `dataset` has `gls` and `txt` attributes.
    """
    # level = cfg["level"]
    level = cfg.get("level", "word") # Default to 'word' level
    max_size = cfg.get(f"{vocab_type}_max_size", -1)
    min_freq = cfg.get(f"{vocab_type}_min_freq", 1)
    
    vocab_file = cfg.get(f"{vocab_type}_vocab", None)
    
    if vocab_file and os.path.exists(vocab_file):
        # Load from file
        if vocab_type == "gls":
            return GlossVocabulary(file=vocab_file)
        else: # txt
            return TextVocabulary(file=vocab_file, level=level)
    else:
        # Build from scratch
        if vocab_type == "gls":
            sentences = [d["gls"] for d in dataset]
            # return GlossVocabulary(sentences, max_size=max_size, min_freq=min_freq)
            return GlossVocabulary(tokens=sentences)
        else: # txt
            sentences = [d["txt"] for d in dataset]
            return TextVocabulary(tokens=sentences, level=level)


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
    """A batch sampler that batches examples by token count."""
    def __init__(self, dataset, batch_size, type="gls", shuffle=False, level="word"):
        self.dataset = dataset
        self.batch_size = batch_size
        self.type = type
        self.shuffle = shuffle
        
        # Determine lengths based on gloss or text
        if self.type == "gls":
            # self.lengths = [len(s[dataset.gls_key].split()) for s in self.dataset.data]
            self.lengths = [len(s[dataset.gls_key].split()) for s in self.dataset.hf_split]
        else:  # txt
            if level == "word":
                self.lengths = [len(s[dataset.txt_key].split()) for s in self.dataset.hf_split]
            else:
                self.lengths = [len(s[dataset.txt_key]) for s in self.dataset.hf_split]

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle: np.random.shuffle(indices)
        indices.sort(key=lambda i: self.lengths[i])
        
        batches, current_batch = [], []
        for i in indices:
            if not current_batch or (len(current_batch) + 1) * self.lengths[i] > self.batch_size:
                if current_batch: batches.append(current_batch)
                current_batch = [i]
            else:
                current_batch.append(i)
        if current_batch: batches.append(current_batch)
        if self.shuffle: np.random.shuffle(batches)
        for batch in batches: yield batch
    
    def __len__(self):
        # Provides an estimate of the number of batches
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class PadCollate:
    """Pads collated samples."""
    def __init__(
        self,
        gls_vocab: GlossVocabulary,
        txt_vocab: TextVocabulary,
        txt_pad_index: int,
        level: str,
    ):
        self.gls_vocab = gls_vocab
        self.txt_vocab = txt_vocab
        # self.sgn_dim = sgn_dim
        self.level = level
        self.txt_pad_index = txt_pad_index

    def __call__(self, batch: List[Dict]) -> Batch:
        if not batch: return Batch(is_train=False)
        
        sequences = [b["sequence"] for b in batch]
        gls_list = [b["gls"] for b in batch]
        txt_list = [b["txt"] for b in batch]
        
        feature_keys = sorted(batch[0]["features"].keys())
        features_by_key = {key: [b["features"][key] for b in batch] for key in feature_keys}

        padded_features, feature_lengths = [], []
        for key in feature_keys:
            stream = features_by_key[key]
            lengths = torch.tensor([s.shape[0] for s in stream])
            feature_lengths.append(lengths)
            padded_features.append(pad_sequence(stream, batch_first=True, padding_value=0.0))

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
            sgn=None,
            features=padded_features,
            feature_lengths=feature_lengths,
            sgn_mask=None,
            sgn_lengths=None,
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
    batch_type: str = "sentence",
    # train: bool = False,
    shuffle: bool = False,
    num_workers: int = 4,
    use_ddp: bool = False,
    rank: int = 0,
    world_size: int = 1,
    level: str = "word",
) -> Tuple[DataLoader, Sampler]:
    """Returns a data loader for a given dataset."""
    sampler = None
    if batch_type == "token":
        sampler = TokenBatchSampler(dataset, batch_size, type="txt", shuffle=shuffle, level=level)
    elif use_ddp:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
    
    dataloader_shuffle = shuffle and sampler is None

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=dataloader_shuffle,
        sampler=sampler,
        drop_last=False,
        collate_fn=PadCollate(
            gls_vocab=gls_vocab,
            txt_vocab=txt_vocab,
            txt_pad_index=txt_vocab.stoi[PAD_TOKEN],
            level=level,
        ),
        num_workers=num_workers,
    ), sampler

# Small helper property for vocabulary to make collate fn cleaner
Vocabulary.is_word_level = property(lambda self: self.specials == [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN])
