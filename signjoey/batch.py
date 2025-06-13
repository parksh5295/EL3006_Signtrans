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
        sgn: torch.Tensor,
        sgn_lengths: torch.Tensor,
        gls: torch.Tensor,
        gls_lengths: torch.Tensor,
        txt: torch.Tensor,
        txt_input: torch.Tensor,
        txt_lengths: torch.Tensor,
        sequence: List[str],
        signer: List[str],
        use_cuda: bool = False,
        txt_pad_index: int = 1,
        sgn_dim: int = -1, # sgn_dim is needed for mask creation
    ):
        """
        Create a new batch.
        :param sgn: Sign language video features
        :param sgn_lengths: Lengths of sign language videos
        :param gls: Gloss sequences
        :param gls_lengths: Lengths of gloss sequences
        :param txt: Text sequences
        :param txt_input: Shifted text sequences for teacher forcing
        :param txt_lengths: Lengths of text sequences
        :param sequence: List of sequence names
        :param signer: List of signer names
        :param use_cuda: Whether to use CUDA
        :param txt_pad_index: Padding index for text
        :param sgn_dim: Dimension of sign features
        """
        self.sgn = sgn
        self.sgn_lengths = sgn_lengths
        if sgn_dim != -1:
             self.sgn_mask = (self.sgn != torch.zeros(sgn_dim))[..., 0].unsqueeze(1)
        else: # Fallback for older checkpoints that might not have sgn_dim
             self.sgn_mask = (torch.sum(self.sgn, dim=2) != 0).unsqueeze(1)

        self.gls = gls
        self.gls_lengths = gls_lengths
        self.txt = txt
        self.txt_input = txt_input
        self.txt_lengths = txt_lengths
        self.txt_mask = (self.txt_input != txt_pad_index).unsqueeze(1)
        
        self.sequence = sequence
        self.signer = signer

        self.num_seqs = self.sgn.size(0)
        self.num_gls_tokens = self.gls_lengths.sum().item()
        self.num_txt_tokens = (self.txt != txt_pad_index).data.sum().item()
        self.use_cuda = use_cuda

        if self.use_cuda:
            self._make_cuda()

    def _make_cuda(self):
        # Move the batch to GPU
        self.sgn = self.sgn.cuda()
        self.sgn_mask = self.sgn_mask.cuda()

        if self.txt_input is not None:
            self.txt = self.txt.cuda()
            self.txt_mask = self.txt_mask.cuda()
            self.txt_input = self.txt_input.cuda()
        
        if self.gls is not None:
            self.gls = self.gls.cuda()


    def sort_by_sgn_lengths(self):
        """
        Sort by sgn length (descending) and return index to revert sort
        """
        _, perm_index = self.sgn_lengths.sort(0, descending=True)
        rev_index = [0] * perm_index.size(0)
        for new_pos, old_pos in enumerate(perm_index.cpu().numpy()):
            rev_index[old_pos] = new_pos

        self.sgn = self.sgn[perm_index]
        self.sgn_mask = self.sgn_mask[perm_index]
        self.sgn_lengths = self.sgn_lengths[perm_index]

        # self.signer = [self.signer[pi] for pi in perm_index]
        self.sequence = [self.sequence[pi] for pi in perm_index]
        self.signer = [self.signer[pi] for pi in perm_index]

        if self.gls is not None:
            self.gls = self.gls[perm_index]
            self.gls_lengths = self.gls_lengths[perm_index]

        if self.txt is not None:
            self.txt = self.txt[perm_index]
            self.txt_mask = self.txt_mask[perm_index]
            self.txt_input = self.txt_input[perm_index]
            self.txt_lengths = self.txt_lengths[perm_index]

        if self.use_cuda:
            self._make_cuda()

        return rev_index
