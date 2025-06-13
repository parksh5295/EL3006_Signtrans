# coding: utf-8
"""
Attention modules
"""

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class AttentionMechanism(nn.Module):
    """
    Base attention class
    """

    def forward(self, *inputs):
        raise NotImplementedError("Implement this.")


class BahdanauAttention(AttentionMechanism):
    """
    Implements Bahdanau (MLP) attention

    Section A.1.2 in https://arxiv.org/pdf/1409.0473.pdf.
    """

    def __init__(self, hidden_size=1, key_size=1, query_size=1):
        """
        Creates attention mechanism.

        :param hidden_size: size of the projection for query and key
        :param key_size: size of the attention input keys
        :param query_size: size of the query
        """

        super(BahdanauAttention, self).__init__()

        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)

        self.proj_keys = None  # to store projected keys
        self.proj_query = None  # projected query

    # pylint: disable=arguments-differ
    def forward(self, query: Tensor = None, mask: Tensor = None, values: Tensor = None):
        """
        Bahdanau MLP attention forward pass.

        :param query: the item (decoder state) to compare with the keys/memory,
            shape (batch_size, 1, decoder.hidden_size)
        :param mask: mask out keys position (0 in invalid positions, 1 else),
            shape (batch_size, 1, sgn_length)
        :param values: values (encoder states),
            shape (batch_size, sgn_length, encoder.hidden_size)
        :return: context vector of shape (batch_size, 1, value_size),
            attention probabilities of shape (batch_size, 1, sgn_length)
        """
        self._check_input_shapes_forward(query=query, mask=mask, values=values)

        assert mask is not None, "mask is required"
        assert self.proj_keys is not None, "projection keys have to get pre-computed"

        # We first project the query (the decoder state).
        # The projected keys (the encoder states) were already pre-computated.
        self.compute_proj_query(query)

        # Calculate scores.
        # proj_keys: batch x sgn_len x hidden_size
        # proj_query: batch x 1 x hidden_size
        scores = self.energy_layer(torch.tanh(self.proj_query + self.proj_keys))
        # scores: batch x sgn_len x 1

        scores = scores.squeeze(2).unsqueeze(1)
        # scores: batch x 1 x time

        # mask out invalid positions by filling the masked out parts with -inf
        scores = torch.where(mask, scores, scores.new_full([1], float("-inf")))

        # turn scores to probabilities
        alphas = F.softmax(scores, dim=-1)  # batch x 1 x time

        # the context vector is the weighted sum of the values
        context = alphas @ values  # batch x 1 x value_size

        return context, alphas

    def compute_proj_keys(self, keys: Tensor):
        """
        Compute the projection of the keys.
        Is efficient if pre-computed before receiving individual queries.

        :param keys:
        :return:
        """
        self.proj_keys = self.key_layer(keys)

    def compute_proj_query(self, query: Tensor):
        """
        Compute the projection of the query.

        :param query:
        :return:
        """
        self.proj_query = self.query_layer(query)

    def _check_input_shapes_forward(
        self, query: torch.Tensor, mask: torch.Tensor, values: torch.Tensor
    ):
        """
        Make sure that inputs to `self.forward` are of correct shape.
        Same input semantics as for `self.forward`.

        :param query:
        :param mask:
        :param values:
        :return:
        """
        assert query.shape[0] == values.shape[0] == mask.shape[0]
        assert query.shape[1] == 1 == mask.shape[1]
        assert query.shape[2] == self.query_layer.in_features
        assert values.shape[2] == self.key_layer.in_features
        assert mask.shape[2] == values.shape[1]

    def __repr__(self):
        return "BahdanauAttention"


class LuongAttention(AttentionMechanism):
    """
    Implements Luong (bilinear / multiplicative) attention.

    Eq. 8 ("general") in http://aclweb.org/anthology/D15-1166.
    """

    def __init__(self, hidden_size: int = 1, key_size: int = 1):
        """
        Creates attention mechanism.

        :param hidden_size: size of the key projection layer, has to be equal
            to decoder hidden size
        :param key_size: size of the attention input keys
        """

        super(LuongAttention, self).__init__()
        self.key_layer = nn.Linear(
            in_features=key_size, out_features=hidden_size, bias=False
        )
        self.proj_keys = None  # projected keys

    # pylint: disable=arguments-differ
    def forward(
        self,
        query: torch.Tensor = None,
        mask: torch.Tensor = None,
        values: torch.Tensor = None,
    ):
        """
        Luong (multiplicative / bilinear) attention forward pass.
        Computes context vectors and attention scores for a given query and
        all masked values and returns them.

        :param query: the item (decoder state) to compare with the keys/memory,
            shape (batch_size, 1, decoder.hidden_size)
        :param mask: mask out keys position (0 in invalid positions, 1 else),
            shape (batch_size, 1, sgn_length)
        :param values: values (encoder states),
            shape (batch_size, sgn_length, encoder.hidden_size)
        :return: context vector of shape (batch_size, 1, value_size),
            attention probabilities of shape (batch_size, 1, sgn_length)
        """
        self._check_input_shapes_forward(query=query, mask=mask, values=values)

        assert self.proj_keys is not None, "projection keys have to get pre-computed"
        assert mask is not None, "mask is required"

        # scores: batch_size x 1 x sgn_length
        scores = query @ self.proj_keys.transpose(1, 2)

        # mask out invalid positions by filling the masked out parts with -inf
        scores = torch.where(mask, scores, scores.new_full([1], float("-inf")))

        # turn scores to probabilities
        alphas = F.softmax(scores, dim=-1)  # batch x 1 x sgn_len

        # the context vector is the weighted sum of the values
        context = alphas @ values  # batch x 1 x values_size

        return context, alphas

    def compute_proj_keys(self, keys: Tensor):
        """
        Compute the projection of the keys and assign them to `self.proj_keys`.
        This pre-computation is efficiently done for all keys
        before receiving individual queries.

        :param keys: shape (batch_size, sgn_length, encoder.hidden_size)
        """
        # proj_keys: batch x sgn_len x hidden_size
        self.proj_keys = self.key_layer(keys)

    def _check_input_shapes_forward(
        self, query: torch.Tensor, mask: torch.Tensor, values: torch.Tensor
    ):
        """
        Make sure that inputs to `self.forward` are of correct shape.
        Same input semantics as for `self.forward`.

        :param query:
        :param mask:
        :param values:
        :return:
        """
        assert query.shape[0] == values.shape[0] == mask.shape[0]
        assert query.shape[1] == 1 == mask.shape[1]
        assert query.shape[2] == self.key_layer.out_features
        assert values.shape[2] == self.key_layer.in_features
        assert mask.shape[2] == values.shape[1]

    def __repr__(self):
        return "LuongAttention"


class TimeAlignmentModule(nn.Module):
    """
    A module to align different feature streams (e.g., pose, hands, mouth) in time.
    It uses a primary stream (assumed to be the first in the list) as a query
    and aligns other streams to it using cross-attention.
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        """
        :param hidden_size: The dimension of the features.
        :param num_heads: The number of attention heads.
        :param dropout: Dropout probability.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_streams = -1 # Will be determined from input

        # We'll create attention layers dynamically on the first forward pass
        # once we know the number of streams.
        self.attention_layers = None
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        # The final projection layer will combine the primary stream with the 
        # aligned auxiliary streams. The input dimension will depend on the
        # number of streams.
        self.final_projection = None
        self.dropout = nn.Dropout(p=dropout)

    def _initialize_modules(self, num_streams: int):
        """
        Initialize the modules once the number of streams is known.
        """
        self.num_streams = num_streams
        # Create attention layers for each auxiliary stream
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.hidden_size,
                num_heads=self.num_heads,
                dropout=self.dropout,
                batch_first=True,
            ) for _ in range(num_streams - 1)
        ])
        # The input to the final projection will be the concatenation of all streams
        self.final_projection = nn.Linear(
            self.hidden_size * num_streams, self.hidden_size
        )


    def forward(self, features: List[Tensor]) -> Tensor:
        """
        Aligns and fuses the feature streams.
        :param features: A list of tensors, where the first is the primary stream (e.g., pose)
                         and the rest are auxiliary streams (e.g., hands, mouth).
                         Each tensor is of shape (B, T, D).
        :return: A single fused tensor of shape (B, T, D).
        """
        if not features:
            return None
        
        if self.attention_layers is None:
            self._initialize_modules(len(features))

        primary_stream = features[0]
        auxiliary_streams = features[1:]

        aligned_features = [primary_stream]
        for i, aux_stream in enumerate(auxiliary_streams):
            # The primary stream acts as the query, attending to the auxiliary stream
            # Note: MultiheadAttention expects (T, B, D) if batch_first=False
            # but we use batch_first=True, so (B, T, D) is correct.
            aligned_aux, _ = self.attention_layers[i](
                query=primary_stream,
                key=aux_stream,
                value=aux_stream,
            )
            aligned_features.append(aligned_aux)

        # Concatenate the primary stream with the aligned auxiliary streams
        combined = torch.cat(aligned_features, dim=-1)

        # Project the combined features back to the original hidden size
        fused = self.final_projection(combined)
        fused = self.layer_norm(fused + primary_stream) # Add residual connection
        fused = self.dropout(fused)

        return fused
