# coding: utf-8
"""
Data module for How2Sign, correctly processing remote keypoint archives and local CSVs.
"""
import os
import pandas as pd
from typing import List, Dict, Tuple
import json
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from huggingface_hub import hf_hub_download, list_repo_files

from signjoey.vocabulary import GlossVocabulary, TextVocabulary, BOS_TOKEN, PAD_TOKEN
from signjoey.batch import Batch

# A cache for downloaded keypoint files to avoid re-downloading during a run.
# This is a simple in-memory cache. For larger datasets, a disk-based cache might be better.
KP_CACHE = {}

# Order of keypoints from the 'GloFE' project's GitHub issues
KEYPOINT_ORDER = [
    "pose_keypoints_2d",
    "hand_left_keypoints_2d",
    "hand_right_keypoints_2d",
    "face_keypoints_2d",
]

def load_and_parse_keypoint_file(repo_id: str, filepath: str) -> torch.Tensor:
    """
    Downloads a single keypoint JSON file from the HF Hub, parses it, and returns a tensor.
    Uses a cache to avoid re-downloading.
    """
    if filepath in KP_CACHE:
        return KP_CACHE[filepath]
    
    try:
        local_path = hf_hub_download(repo_id=repo_id, filename=filepath, repo_type="dataset")
        with open(local_path, 'r') as f:
            data = json.load(f)
        
        person_data = data.get("people", [{}])[0]
        
        frame_keypoints = []
        for key in KEYPOINT_ORDER:
            kp = person_data.get(key, [])
            frame_keypoints.extend(kp)
            
        tensor = torch.tensor(frame_keypoints, dtype=torch.float32).reshape(-1, 3)
        KP_CACHE[filepath] = tensor
        return tensor
        
    except Exception as e:
        # print(f"Warning: Could not load or parse {filepath}. Error: {e}")
        return None


def load_data_nonmap(data_cfg: dict) -> Tuple[Dataset, Dataset, Dataset, GlossVocabulary, TextVocabulary]:
    """
    Loads data by mapping local CSV annotations to file listings from a Hugging Face Hub repo.
    """
    print("--- Starting data loading (nonmap v3: hub file listing) ---")
    
    repo_id = data_cfg["hf_keypoint_dataset"]
    csv_root = data_cfg["csv_root"]

    print(f"Listing files in HF repo: {repo_id}...")
    try:
        # This can be slow, but is necessary.
        repo_files = set(list_repo_files(repo_id, repo_type="dataset"))
        print(f"Found {len(repo_files)} files in the repo.")
    except Exception as e:
        raise IOError(f"Could not list files in Hugging Face Hub repo '{repo_id}'. "
                      f"Please check repo name and your connection. Error: {e}")

    # Create a mapping from SENTENCE_NAME (video clip) to its frame files
    print("Creating a map from video clips to frame files...")
    clip_to_frames_map = {}
    for filepath in tqdm(repo_files, desc="Mapping repo files"):
        if filepath.endswith("_keypoints.json"):
            # e.g., "openpose_output/json/VIDEO_CLIP_NAME/FRAME_FILE.json"
            parts = filepath.split('/')
            if len(parts) >= 3:
                clip_name = parts[2]
                if clip_name not in clip_to_frames_map:
                    clip_to_frames_map[clip_name] = []
                clip_to_frames_map[clip_name].append(filepath)
    
    # Sort the frames within each clip chronologically
    for clip_name in clip_to_frames_map:
        clip_to_frames_map[clip_name].sort()

    def build_split(split_name: str):
        """Builds a dataset split by reading a CSV and matching with the file map."""
        csv_path = os.path.join(csv_root, f"how2sign_realigned_{split_name}.csv")
        df = pd.read_csv(csv_path, sep='\\t', engine='python')
        
        data_list = []
        print(f"Processing {split_name} split...")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            sentence_name = row["SENTENCE_NAME"]
            if sentence_name in clip_to_frames_map:
                data_list.append({
                    "sequence_name": sentence_name,
                    "text": row["SENTENCE"],
                    "gloss": row.get("GLOSS", ""), # Handle optional GLOSS column
                    "frame_files": clip_to_frames_map[sentence_name]
                })
        
        print(f"Matched {len(data_list)} of {len(df)} entries in {split_name}.csv")
        return data_list

    train_list = build_split("train")
    dev_list = build_split("val")
    test_list = build_split("test")

    print("Building vocabularies...")
    txt_vocab = TextVocabulary(tokens=[d['text'] for d in train_list], **data_cfg["txt_vocab"])
    gls_vocab = GlossVocabulary(tokens=[d['gloss'] for d in train_list], **data_cfg["gls_vocab"])
    
    train_dataset = SignTranslationDataset_NonMap(train_list, repo_id, data_cfg)
    dev_dataset = SignTranslationDataset_NonMap(dev_list, repo_id, data_cfg)
    test_dataset = SignTranslationDataset_NonMap(test_list, repo_id, data_cfg)

    return train_dataset, dev_dataset, test_dataset, gls_vocab, txt_vocab


class SignTranslationDataset_NonMap(Dataset):
    """
    A PyTorch Dataset that loads keypoint data frame-by-frame from the HF Hub
    based on a pre-computed file list.
    """
    def __init__(self, data_list: List[Dict], repo_id: str, data_cfg: dict):
        self.data_list = data_list
        self.repo_id = repo_id
        self.data_cfg = data_cfg
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data_list[idx]
        
        frame_tensors = []
        for frame_file in item["frame_files"]:
            tensor = load_and_parse_keypoint_file(self.repo_id, frame_file)
            if tensor is not None:
                frame_tensors.append(tensor)
        
        if not frame_tensors:
            # Return a dummy tensor if no frames could be loaded for this item
            # This should be handled by the collate function
            features = torch.zeros((1, 225), dtype=torch.float32) # 75 keypoints * 3 coords
        else:
            # Stack all frame tensors to create the full sequence tensor
            features = torch.stack(frame_tensors, dim=0)
            
        return {
            "sequence": item["sequence_name"],
            "gls": item["gloss"],
            "txt": item["text"],
            "features": features,
        }

class PadCollate_NonMap:
    def __init__(self, gls_vocab: GlossVocabulary, txt_vocab: TextVocabulary):
        self.gls_vocab = gls_vocab
        self.txt_vocab = txt_vocab

    def __call__(self, batch: List[Dict]) -> Batch:
        # Filter out items where features might be empty
        batch = [b for b in batch if b["features"].shape[0] > 1]
        if not batch:
            return None # Should not happen in practice if data is clean

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
            is_train=True, features=[padded_features], feature_lengths=[feature_lengths],
            sgn=None, sgn_mask=None, sgn_lengths=None, gls=gls_ids, gls_lengths=gls_lengths,
            txt=txt_ids, txt_input=txt_input, txt_lengths=txt_lengths,
            txt_pad_index=self.txt_vocab.stoi[PAD_TOKEN], sequence=sequences,
        )

# This function remains the same as before
from torch.utils.data.distributed import DistributedSampler
def make_data_iter_nonmap(
    dataset: Dataset, batch_size: int, gls_vocab: GlossVocabulary,
    txt_vocab: TextVocabulary, shuffle: bool = False, use_ddp: bool = False,
    rank: int = 0, world_size: int = 1,
) -> torch.utils.data.DataLoader:
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle) if use_ddp else None
    return torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle and sampler is None,
        sampler=sampler, drop_last=False,
        collate_fn=PadCollate_NonMap(gls_vocab=gls_vocab, txt_vocab=txt_vocab),
        num_workers=4,
    )