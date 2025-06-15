# coding: utf-8
import math
import random
import torch
import numpy as np
from typing import List


class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(
        self,
        # sgn: torch.Tensor,
        # sgn_lengths: torch.Tensor,
        is_train: bool,
        gls: torch.Tensor,
        gls_lengths: torch.Tensor,
        txt: torch.Tensor,
        txt_input: torch.Tensor,
        txt_lengths: torch.Tensor,
        txt_pad_index: int,
        sequence: List[str],
        rank: int = 0,
        features: List[torch.Tensor] = None,
        feature_lengths: List[torch.Tensor] = None,
        sgn: torch.Tensor = None,
        sgn_lengths: torch.Tensor = None,
        signer: List[str] = None,
    ):
        """
        Create a new batch.
        :param is_train: Whether this batch is for training.
        :param features: List of sign language video feature tensors.
        :param feature_lengths: List of lengths for each feature stream.
        :param gls: Gloss sequences.
        :param gls_lengths: Lengths of gloss sequences.
        :param txt: Text sequences.
        :param txt_input: Shifted text sequences for teacher forcing.
        :param txt_lengths: Lengths of text sequences.
        :param txt_pad_index: Padding index for text.
        :param sequence: List of sequence names.
        :param rank: The GPU device rank for distributed training.
        :param signer: List of signer names (optional).
        """
        self.is_train = is_train
        self.features = features
        self.feature_lengths = feature_lengths
        self.gls = gls
        self.gls_lengths = gls_lengths
        self.txt = txt
        self.txt_input = txt_input
        self.txt_lengths = txt_lengths
        self.txt_mask = (self.txt_input != txt_pad_index).unsqueeze(1)
        self.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        
        self.sequence = sequence
        self.signer = signer
        
        # For backward compatibility and convenience, expose the first feature stream
        # as the 'sgn' tensor. This might be removed in future versions.
        if self.features:
            self.sgn = self.features[0]
            self.sgn_lengths = self.feature_lengths[0]
            self.sgn_mask = (torch.sum(self.sgn, dim=2) != 0).unsqueeze(1)
            self.num_seqs = self.sgn.size(0)
        else: # Handle cases where there are no features (e.g. text-only batches)
            self.sgn = sgn
            self.sgn_lengths = sgn_lengths
            if sgn is not None:
                self.sgn_mask = (torch.sum(self.sgn, dim=2) != 0).unsqueeze(1)
                self.num_seqs = self.sgn.size(0)
            else:
                self.sgn_mask = None
                self.num_seqs = len(sequence) if sequence else 0


        if self.is_train:
            self.num_gls_tokens = self.gls_lengths.sum().item()
            self.num_txt_tokens = (self.txt != txt_pad_index).data.sum().item()
        
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self._make_cuda()

    def _make_cuda(self):
        """Move the batch to the designated GPU."""
        if self.features:
            self.features = [f.to(self.device) for f in self.features]
            self.sgn_mask = self.sgn_mask.to(self.device)
        elif self.sgn is not None:
             self.sgn = self.sgn.to(self.device)
             self.sgn_mask = self.sgn_mask.to(self.device)


        if self.txt_input is not None:
            self.txt = self.txt.to(self.device)
            self.txt_mask = self.txt_mask.to(self.device)
            self.txt_input = self.txt_input.to(self.device)
        
        if self.gls is not None:
            self.gls = self.gls.to(self.device)

    def sort_by_feature_lengths(self):
        """
        Sort by sgn length (descending) and return index to revert sort.
        This primarily sorts by the length of the first feature stream.
        """
        if self.sgn_lengths is None:
            return None # Cannot sort if there are no sign features

        _, perm_index = self.sgn_lengths.sort(0, descending=True)
        rev_index = torch.tensor([0] * perm_index.size(0), dtype=torch.long)
        for new_pos, old_pos in enumerate(perm_index.cpu().numpy()):
            rev_index[old_pos] = new_pos

        # Sort all feature streams and their length tensors
        if self.features:
            self.features = [f[perm_index] for f in self.features]
            self.feature_lengths = [fl[perm_index] for fl in self.feature_lengths]
        
        # Also sort the convenience attributes
        self.sgn = self.sgn[perm_index]
        self.sgn_mask = self.sgn_mask[perm_index]
        self.sgn_lengths = self.sgn_lengths[perm_index]

        # self.signer = [self.signer[pi] for pi in perm_index]
        self.sequence = [self.sequence[pi] for pi in perm_index]
        if self.signer:
            self.signer = [self.signer[pi] for pi in perm_index]

        if self.gls is not None:
            self.gls = self.gls[perm_index]
            self.gls_lengths = self.gls_lengths[perm_index]

        if self.txt is not None:
            self.txt = self.txt[perm_index]
            self.txt_mask = self.txt_mask[perm_index]
            self.txt_input = self.txt_input[perm_index]
            self.txt_lengths = self.txt_lengths[perm_index]

        # No need to call _make_cuda again, as tensors are already on the device.
        # if self.use_cuda:
        #     self._make_cuda()

        return rev_index
