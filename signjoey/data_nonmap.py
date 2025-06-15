"""
Data module for time-alignment pre-training.
Loads ONLY keypoint data from How2Sign without any text annotations.
"""
import os
from typing import List, Dict, Tuple
import json
from tqdm import tqdm
import datasets
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from huggingface_hub import hf_hub_download

from signjoey.batch import Batch

# A cache for downloaded keypoint files to avoid re-downloading during a run.
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
            
        tensor = torch.tensor(frame_keypoints, dtype=torch.float32)
        KP_CACHE[filepath] = tensor
        return tensor
        
    except Exception:
        return None

def load_data_nonmap(data_cfg: dict) -> Tuple[Dataset, Dataset, Dataset, None, None]:
    """
    Loads data for time-alignment. It scans all splits ('train', 'val', 'test')
    to create a comprehensive map of all available video clips and their frames.
    It does NOT load any text annotations.
    """
    print("--- Starting data loading (nonmap for time-alignment) ---")
    
    repo_id = data_cfg["hf_keypoint_dataset"]
    
    print(f"Streaming dataset from HF Hub: {repo_id}...")
    
    # --- Create a comprehensive map from ALL splits ---
    clip_to_frames_map = defaultdict(list)
    
    for split in ["train", "validation", "test"]:
        print(f"Scanning '{split}' split for video clips...")
        try:
            hf_dataset = datasets.load_dataset(repo_id, streaming=True, split=split)
            
            subset_size = data_cfg.get("dataset_subset_size", -1)
            if subset_size > 0:
                print(f"Limiting scan to the first {subset_size} items for speed.")
                hf_dataset = hf_dataset.take(subset_size)

            for item in tqdm(hf_dataset, desc=f"Scanning {split}"):
                key = item.get("__key__")
                if key and key.endswith("_keypoints"):
                    parts = key.split('/')
                    if len(parts) >= 3:
                        clip_name = parts[2]
                        # Append the .json extension for the actual download filename
                        clip_to_frames_map[clip_name].append(key + ".json")
        except Exception as e:
            print(f"Warning: Could not load or stream split '{split}'. Error: {e}")

    if not clip_to_frames_map:
        raise ValueError("Could not find any keypoint files in the dataset stream.")

    print(f"Mapped {len(clip_to_frames_map)} unique video clips across all splits.")

    # Sort the frames within each clip chronologically
    for clip_name in clip_to_frames_map:
        clip_to_frames_map[clip_name].sort()

    # Since we don't use CSVs, we just use the keys from our map as the data
    all_clips = list(clip_to_frames_map.keys())
    
    # We can create dummy splits, or just use all data as 'train' for pre-training
    # For simplicity, let's just create one big dataset.
    data_list = [
        {"sequence_name": name, "frame_files": files}
        for name, files in clip_to_frames_map.items()
    ]

    print(f"Created a single dataset with {len(data_list)} clips for time-alignment.")
    
    # We create one dataset and return it for all splits (train, dev, test)
    # The training script can then decide which one to use.
    alignment_dataset = TimeAlignmentDataset(data_list, repo_id)

    # Return the dataset and None for vocabs
    return alignment_dataset, alignment_dataset, alignment_dataset, None, None


class TimeAlignmentDataset(Dataset):
    """
    A PyTorch Dataset for time-alignment pre-training. It loads keypoint data
    frame-by-frame from the HF Hub.
    """
    def __init__(self, data_list: List[Dict], repo_id: str):
        self.data_list = data_list
        self.repo_id = repo_id
        
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
            # Return a dummy tensor if no frames could be loaded.
            # Shape: (1, 411) - based on previous findings.
            features = torch.zeros((1, 411), dtype=torch.float32)
        else:
            features = torch.stack(frame_tensors, dim=0)
            
        return {
            "sequence": item["sequence_name"],
            "features": features,
        }

class PadCollate_TimeAlignment:
    """A collate_fn for time-alignment that only pads features."""
    def __call__(self, batch: List[Dict]) -> Dict:
        # Filter out items where features might be empty
        batch = [b for b in batch if b["features"].shape[0] > 1]
        if not batch:
            return None

        sequences = [b["sequence"] for b in batch]
        features = [b["features"] for b in batch]

        padded_features = pad_sequence(features, batch_first=True, padding_value=0.0)
        feature_lengths = torch.tensor([f.shape[0] for f in features])

        return {
            "sequences": sequences,
            "features": padded_features,
            "feature_lengths": feature_lengths
        }

def make_data_iter_nonmap(
    dataset: Dataset, batch_size: int, shuffle: bool = False, use_ddp: bool = False,
    rank: int = 0, world_size: int = 1, num_workers: int = 4
) -> torch.utils.data.DataLoader:
    
    sampler = None
    if use_ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
        )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=PadCollate_TimeAlignment(),
        pin_memory=True,
    )