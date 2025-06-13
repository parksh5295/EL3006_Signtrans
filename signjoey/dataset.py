# coding: utf-8
# Data module
import gzip
import pickle
from typing import List

import torch
from torch.utils.data import Dataset


def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object


class SignTranslationDataset(Dataset):
    """
    A custom dataset for sign language translation.
    This dataset loads data from gzipped pickle files.
    """

    '''
    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.sgn), len(ex.txt))

    def __init__(
        self,
        path: str,
        fields: Tuple[RawField, RawField, Field, Field, Field],
        **kwargs
    ):
    '''

    def __init__(self, path: str or List[str]):
        """
        Create a SignTranslationDataset.
        param path: Path to the data file(s). Can be a single path or a list of paths.
        """
        if isinstance(path, str):
            path = [path]

        samples = {}
        for annotation_file in path:
            tmp = load_dataset_file(annotation_file)
            for s in tmp:
                seq_id = s["name"]
                if seq_id in samples:
                    # If the sequence ID already exists, concatenate the sign features.
                    # This is used for multi-channel features (e.g., RGB + Keypoints).
                    assert samples[seq_id]["name"] == s["name"]
                    assert samples[seq_id]["signer"] == s["signer"]
                    assert samples[seq_id]["gloss"] == s["gloss"]
                    assert samples[seq_id]["text"] == s["text"]
                    samples[seq_id]["sign"] = torch.cat(
                        [samples[seq_id]["sign"], s["sign"]], axis=1
                    )
                else:
                    samples[seq_id] = {
                        "name": s["name"],
                        "signer": s["signer"],
                        "gloss": s["gloss"],
                        "text": s["text"],
                        "sign": s["sign"],
                    }

        self.samples = list(samples.values())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Add a small epsilon for numerical stability
        return {
            "name": sample["name"],
            "signer": sample["signer"],
            "sign": sample["sign"] + 1e-8,
            "gloss": sample["gloss"].strip(),
            "text": sample["text"].strip(),
        }
