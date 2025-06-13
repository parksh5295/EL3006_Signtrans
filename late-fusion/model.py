# coding: utf-8
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

from itertools import groupby
from signjoey.initialization import initialize_model
from signjoey.embeddings import Embeddings, SpatialEmbeddings
from signjoey.encoders import Encoder, RecurrentEncoder, TransformerEncoder
from signjoey.decoders import Decoder, RecurrentDecoder, TransformerDecoder
from signjoey.search import beam_search_ensemble, greedy
from signjoey.vocabulary import (
    TextVocabulary,
    GlossVocabulary,
    PAD_TOKEN,
    EOS_TOKEN,
    BOS_TOKEN,
)
from signjoey.batch import Batch
from signjoey.helpers import freeze_params
from torch import Tensor
from typing import Union, List

# Import the new TimeAlignmentModule
from time_alignment import TimeAlignmentModule


class SignModelEnsemble(nn.Module):
    """
    Ensemble Model class with Time Alignment
    """

    def __init__(
        self,
        # Add the time alignment module
        time_aligner: TimeAlignmentModule,
        encoders: nn.ModuleList,
        gloss_output_layers: nn.ModuleList,
        decoders: nn.ModuleList,
        txt_embed: Embeddings,
        gls_vocab: GlossVocabulary,
        txt_vocab: TextVocabulary,
        do_recognition: bool = True,
        do_translation: bool = True,
    ):
        """
        Create a new encoder-decoder model with time alignment.
        """
        super().__init__()

        self.time_aligner = time_aligner
        self.encoders = encoders
        self.decoders = decoders
        self.txt_embed = txt_embed
        self.gls_vocab = gls_vocab
        self.txt_vocab = txt_vocab
        self.txt_bos_index = self.txt_vocab.stoi[BOS_TOKEN]
        self.txt_pad_index = self.txt_vocab.stoi[PAD_TOKEN]
        self.txt_eos_index = self.txt_vocab.stoi[EOS_TOKEN]
        # self.gloss_output_layers = nn.ModuleList(gloss_output_layers)
        self.gloss_output_layers = gloss_output_layers
        self.do_recognition = do_recognition
        self.do_translation = do_translation

    # pylint: disable=arguments-differ
    def forward(
        self,
        features: List[Tensor],
        txt_input: Tensor,
        sgn_mask: Tensor,
        txt_mask: Tensor = None,
    ) -> (Tensor, Tensor):
        """
        First aligns the features, then encodes them, 
        and then produces the target one word at a time.

        :param features: A list of feature tensors [pose, hand, mouth],
                         each with shape (B, T_x, D_x) where T_x can be different.
        :param txt_input: target input
        :param sgn_mask: source mask (for decoder attention)
        :param txt_mask: target mask
        :return: decoder outputs, gloss probabilities
        """
        encoder_outputs, encoder_hiddens = self.encode(
            # sgn=sgn, sgn_mask=sgn_mask, sgn_length=sgn_lengths, sgn_dims=sgn_dims
            features=features, sgn_mask=sgn_mask
        )

        gloss_probabilities_list = []
        if self.do_recognition:
            # Gloss Recognition Part
            # N x T x C
            for i in range(len(encoder_outputs)):
                gloss_scores = self.gloss_output_layers[i](encoder_outputs[i])
                gloss_probabilities = gloss_scores.log_softmax(2).permute(1, 0, 2)
                gloss_probabilities_list.append(gloss_probabilities)
        else:
            gloss_probabilities = None

        if self.do_translation:
            unroll_steps = txt_input.size(1)
            decoder_outputs = self.decode(
                encoder_output=encoder_outputs,
                encoder_hidden=encoder_hiddens,
                sgn_mask=sgn_mask, # Note: This mask should correspond to the aligned length
                txt_input=txt_input,
                unroll_steps=unroll_steps,
                txt_mask=txt_mask,
                # sgn_dims=sgn_dims
            )
        else:
            decoder_outputs = None

        if gloss_probabilities_list:
            # Ensemble gloss probabilities (example: weighted average)
            gloss_probabilities = torch.stack(gloss_probabilities_list)
            gloss_probabilities = gloss_probabilities[0] * 0.7 + gloss_probabilities[1] * 0.15 + gloss_probabilities[2] * 0.15
        else:
            gloss_probabilities = None
            
        return decoder_outputs, gloss_probabilities

    def encode(self, features: List[Tensor], sgn_mask: Tensor) -> (List[Tensor], List[Tensor]):
        """
        Applies time alignment and then encodes the source features.

        :param features: A list of feature tensors [pose, hand, mouth].
        :param sgn_mask: Original source mask.
        :return: A list of encoder outputs and a list of hidden states.
        """
        # 1. Apply Time Alignment
        # Input: list of (B, T_x, D_x)
        # Output: tuple of (B, target_len, hidden_dim)
        pose_feat, hand_feat, mouth_feat = self.time_aligner(
            features[0], features[1], features[2]
        )
        aligned_features = [pose_feat, hand_feat, mouth_feat]

        # 2. Create a new mask for the aligned features
        batch_size, target_len, _ = aligned_features[0].shape
        device = aligned_features[0].device
        # All sequences now have the same length `target_len`
        new_sgn_lengths = torch.full((batch_size,), target_len, device=device)
        new_sgn_mask = (
            torch.arange(target_len, device=device)[None, :] < new_sgn_lengths[:, None]
        ).unsqueeze(1)

        # 3. Encode each aligned feature stream with its dedicated encoder
        encoder_outputs = []
        encoder_hiddens = []
        for i in range(len(self.encoders)):
            out = self.encoders[i](
                embed_src=aligned_features[i],
                src_length=new_sgn_lengths,
                mask=new_sgn_mask,
            )
            encoder_outputs.append(out[0])
            encoder_hiddens.append(out[1])

        return encoder_outputs, encoder_hiddens

    def decode(
        self,
        encoder_output: List[Tensor],
        encoder_hidden: List[Tensor],
        sgn_mask: Tensor,
        txt_input: Tensor,
        unroll_steps: int,
        decoder_hidden: Tensor = None,
        txt_mask: Tensor = None,
    ) -> List:
        """
        Decode, given encoded source features. This part remains mostly the same,
        as it operates on the list of encoder outputs.
        """
        outputs = []
        for i in range(len(encoder_output)):
            hidden = decoder_hidden[i] if decoder_hidden is not None else None
            outputs.append(
                self.decoders[i](
                    encoder_output=encoder_output[i],
                    encoder_hidden=encoder_hidden[i],
                    src_mask=sgn_mask, # This mask should be the new_sgn_mask
                    trg_embed=self.txt_embed(x=txt_input, mask=txt_mask),
                    trg_mask=txt_mask,
                    unroll_steps=unroll_steps,
                    hidden=hidden,
                )
            )
        return outputs

    def get_loss_for_batch(
        self,
        batch: Batch,
        recognition_loss_function: nn.Module,
        translation_loss_function: nn.Module,
        recognition_loss_weight: float,
        translation_loss_weight: float,
    ) -> (Tensor, Tensor):
        """
        Compute loss for a batch.
        NOTE: This function and the data loader need to be adapted.
        The `batch` object must provide separate feature tensors.
        For now, we assume a placeholder `batch.features` exists as a list.
        """
        # Do a forward pass
        # The dataloader needs to be modified to provide a list of features.
        # Placeholder for the required input format:
        # features = [batch.pose_feat, batch.hand_feat, batch.mouth_feat]
        # For now, this will raise an error, signaling the required change.
        features = batch.features 

        decoder_outputs_list, gloss_probabilities = self.forward(
            features=features,
            sgn_mask=batch.sgn_mask,
            txt_input=batch.txt_input,
            txt_mask=batch.txt_mask,
        )

        '''
        if self.do_recognition:
            assert gloss_probabilities is not None
            # Calculate Recognition Loss
            recognition_loss = (
                recognition_loss_function(
                    gloss_probabilities,
                    batch.gls,
                    batch.sgn_lengths.long(),
                    batch.gls_lengths.long(),
                )
                * recognition_loss_weight
            )
        else:
            recognition_loss = None

        if self.do_translation:
           # assert decoder_outputs is not None
            translation_loss_list = []
            for decoder_outputs in decoder_outputs_list:
                word_outputs, _, _, _ = decoder_outputs
                # Calculate Translation Loss
                txt_log_probs = F.log_softmax(word_outputs, dim=-1)
                translation_loss = (
                    translation_loss_function(txt_log_probs, batch.txt)
                    * translation_loss_weight
                )
                translation_loss_list.append(translation_loss)
            translation_loss = torch.mean(torch.stack(translation_loss_list), 0)
        else:
            translation_loss = None

        return recognition_loss, translation_loss
        '''
        # ... (rest of the loss calculation remains the same)
        # ...

    def run_batch(
        self,
        batch: Batch,
        # sgn_dims,
        recognition_beam_size: int = 1,
        translation_beam_size: int = 1,
        translation_beam_alpha: float = -1,
        translation_max_output_length: int = 100,
    ) -> (np.array, np.array, np.array):
        """
        Get outputs and attentions scores for a given batch

        :param batch: batch to generate hypotheses for
        :param recognition_beam_size: size of the beam for CTC beam search
            if 1 use greedy
        :param translation_beam_size: size of the beam for translation beam search
            if 1 use greedy
        :param translation_beam_alpha: alpha value for beam search
        :param translation_max_output_length: maximum length of translation hypotheses
        :return: stacked_output: hypotheses for batch,
            stacked_attention_scores: attention scores for batch
        """

        # NOTE encoder_output, encoder_hidden = self.encode(
        # NOTE    sgn=batch.sgn, sgn_mask=batch.sgn_mask, sgn_length=batch.sgn_lengths
        # NOTE)

        encoder_outputs, encoder_hiddens = self.encode(
                sgn=batch.sgn, sgn_mask=batch.sgn_mask, sgn_length=batch.sgn_lengths, sgn_dims=sgn_dims
        )

        if self.do_recognition:
            # Gloss Recognition Part
            # N x T x C
            gloss_probabilities_list = []
            for i in range(len(encoder_outputs)):
                gloss_scores = self.gloss_output_layers[i](encoder_outputs[i])
                # N x T x C
                gloss_probabilities = gloss_scores.log_softmax(2)
                # Turn it into T x N x C
                gloss_probabilities = gloss_probabilities.permute(1, 0, 2)
                gloss_probabilities = gloss_probabilities.cpu().detach()
                gloss_probabilities_list.append(gloss_probabilities)
           # gloss_probabilities = torch.mean(torch.stack(gloss_probabilities_list), 0)
            gloss_probabilities = torch.stack(gloss_probabilities_list)
            gloss_probabilities = gloss_probabilities[0] * 0.7 + gloss_probabilities[1] * 0.15 + gloss_probabilities[2] * 0.15
            
            gloss_probabilities = gloss_probabilities.numpy()
            tf_gloss_probabilities = np.concatenate(
                (gloss_probabilities[:, :, 1:], gloss_probabilities[:, :, 0, None]),
                axis=-1,
            )

            assert recognition_beam_size > 0
            ctc_decode, _ = tf.nn.ctc_beam_search_decoder(
                inputs=tf_gloss_probabilities,
                sequence_length=batch.sgn_lengths.cpu().detach().numpy(),
                beam_width=recognition_beam_size,
                top_paths=1,
            )
            ctc_decode = ctc_decode[0]
            # Create a decoded gloss list for each sample
            tmp_gloss_sequences = [[] for i in range(gloss_scores.shape[0])]
            for (value_idx, dense_idx) in enumerate(ctc_decode.indices):
                tmp_gloss_sequences[dense_idx[0]].append(
                    ctc_decode.values[value_idx].numpy() + 1
                )
            decoded_gloss_sequences = []
            for seq_idx in range(0, len(tmp_gloss_sequences)):
                decoded_gloss_sequences.append(
                    [x[0] for x in groupby(tmp_gloss_sequences[seq_idx])]
                )
        else:
            decoded_gloss_sequences = None

        if self.do_translation:
            # greedy decoding
            if translation_beam_size < 2:
                stacked_txt_output, stacked_attention_scores = greedy(
                    encoder_hidden=encoder_hiddens, #NOTE CHANGED
                    encoder_output=encoder_outputs, #NOTE CHANGED
                    src_mask=batch.sgn_mask,
                    embed=self.txt_embed,
                    bos_index=self.txt_bos_index,
                    eos_index=self.txt_eos_index,
                    decoder=self.decoders, # NOTE CHANGED
                    max_output_length=translation_max_output_length,
                    sgn_dims=sgn_dims,    # NOTE CHANGED
                )
                # batch, time, max_sgn_length
            else:  # beam size
                stacked_txt_output, stacked_attention_scores = beam_search_ensemble(
                    size=translation_beam_size,
                    encoder_hiddens=encoder_hiddens, # NOTE CHANGED
                    encoder_outputs=encoder_outputs, # NOTE CHANGED
                    src_mask=batch.sgn_mask,
                    embed=self.txt_embed,
                    max_output_length=translation_max_output_length,
                    alpha=translation_beam_alpha,
                    eos_index=self.txt_eos_index,
                    pad_index=self.txt_pad_index,
                    bos_index=self.txt_bos_index,
                    decoders=self.decoders, # NOTE CHANGED
                    sgn_dims=sgn_dims,     # NOTE CHANGED
                )
        else:
            stacked_txt_output = stacked_attention_scores = None

        return decoded_gloss_sequences, stacked_txt_output, stacked_attention_scores

    def __repr__(self) -> str:
        """
        String representation: a description of encoder, decoder and embeddings

        :return: string representation
        """
        return "."
        


class SignModel(nn.Module):
    """
    Base Model class
    """

    def __init__(
        self,
        encoder: Encoder,
        gloss_output_layer: nn.Module,
        decoder: Decoder,
        sgn_embed: SpatialEmbeddings,
        txt_embed: Embeddings,
        gls_vocab: GlossVocabulary,
        txt_vocab: TextVocabulary,
        do_recognition: bool = True,
        do_translation: bool = True,
    ):
        """
        Create a new encoder-decoder model

        :param encoder: encoder
        :param decoder: decoder
        :param sgn_embed: spatial feature frame embeddings
        :param txt_embed: spoken language word embedding
        :param gls_vocab: gls vocabulary
        :param txt_vocab: spoken language vocabulary
        :param do_recognition: flag to build the model with recognition output.
        :param do_translation: flag to build the model with translation decoder.
        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.sgn_embed = sgn_embed
        self.txt_embed = txt_embed

        self.gls_vocab = gls_vocab
        self.txt_vocab = txt_vocab

        self.txt_bos_index = self.txt_vocab.stoi[BOS_TOKEN]
        self.txt_pad_index = self.txt_vocab.stoi[PAD_TOKEN]
        self.txt_eos_index = self.txt_vocab.stoi[EOS_TOKEN]

        self.gloss_output_layer = gloss_output_layer
        self.do_recognition = do_recognition
        self.do_translation = do_translation

    # pylint: disable=arguments-differ
    def forward(
        self,
        sgn: Tensor,
        sgn_mask: Tensor,
        sgn_lengths: Tensor,
        txt_input: Tensor,
        txt_mask: Tensor = None,
    ) -> (Tensor, Tensor, Tensor, Tensor):
        """
        First encodes the source sentence.
        Then produces the target one word at a time.

        :param sgn: source input
        :param sgn_mask: source mask
        :param sgn_lengths: length of source inputs
        :param txt_input: target input
        :param txt_mask: target mask
        :return: decoder outputs
        """
        encoder_output, encoder_hidden = self.encode(
            sgn=sgn, sgn_mask=sgn_mask, sgn_length=sgn_lengths
        )

        if self.do_recognition:
            # Gloss Recognition Part
            # N x T x C
            gloss_scores = self.gloss_output_layer(encoder_output)
            # N x T x C
            gloss_probabilities = gloss_scores.log_softmax(2)
            # Turn it into T x N x C
            gloss_probabilities = gloss_probabilities.permute(1, 0, 2)
        else:
            gloss_probabilities = None

        if self.do_translation:
            unroll_steps = txt_input.size(1)
            decoder_outputs = self.decode(
                encoder_output=encoder_output,
                encoder_hidden=encoder_hidden,
                sgn_mask=sgn_mask,
                txt_input=txt_input,
                unroll_steps=unroll_steps,
                txt_mask=txt_mask,
            )
        else:
            decoder_outputs = None

        return decoder_outputs, gloss_probabilities

    def encode(
        self, sgn: Tensor, sgn_mask: Tensor, sgn_length: Tensor
    ) -> (Tensor, Tensor):
        """
        Encodes the source sentence.

        :param sgn:
        :param sgn_mask:
        :param sgn_length:
        :return: encoder outputs (output, hidden_concat)
        """
        return self.encoder(
            embed_src=self.sgn_embed(x=sgn, mask=sgn_mask),
            src_length=sgn_length,
            mask=sgn_mask,
        )

    def decode(
        self,
        encoder_output: Tensor,
        encoder_hidden: Tensor,
        sgn_mask: Tensor,
        txt_input: Tensor,
        unroll_steps: int,
        decoder_hidden: Tensor = None,
        txt_mask: Tensor = None,
    ) -> (Tensor, Tensor, Tensor, Tensor):
        """
        Decode, given an encoded source sentence.

        :param encoder_output: encoder states for attention computation
        :param encoder_hidden: last encoder state for decoder initialization
        :param sgn_mask: sign sequence mask, 1 at valid tokens
        :param txt_input: spoken language sentence inputs
        :param unroll_steps: number of steps to unroll the decoder for
        :param decoder_hidden: decoder hidden state (optional)
        :param txt_mask: mask for spoken language words
        :return: decoder outputs (outputs, hidden, att_probs, att_vectors)
        """
        return self.decoder(
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            src_mask=sgn_mask,
            trg_embed=self.txt_embed(x=txt_input, mask=txt_mask),
            trg_mask=txt_mask,
            unroll_steps=unroll_steps,
            hidden=decoder_hidden,
        )

    def get_loss_for_batch(
        self,
        batch: Batch,
        recognition_loss_function: nn.Module,
        translation_loss_function: nn.Module,
        recognition_loss_weight: float,
        translation_loss_weight: float,
    ) -> (Tensor, Tensor):
        """
        Compute non-normalized loss and number of tokens for a batch

        :param batch: batch to compute loss for
        :param recognition_loss_function: Sign Language Recognition Loss Function (CTC)
        :param translation_loss_function: Sign Language Translation Loss Function (XEntropy)
        :param recognition_loss_weight: Weight for recognition loss
        :param translation_loss_weight: Weight for translation loss
        :return: recognition_loss: sum of losses over sequences in the batch
        :return: translation_loss: sum of losses over non-pad elements in the batch
        """
        # pylint: disable=unused-variable

        # Do a forward pass
        decoder_outputs, gloss_probabilities = self.forward(
            sgn=batch.sgn,
            sgn_mask=batch.sgn_mask,
            sgn_lengths=batch.sgn_lengths,
            txt_input=batch.txt_input,
            txt_mask=batch.txt_mask,
        )

        if self.do_recognition:
            assert gloss_probabilities is not None
            # Calculate Recognition Loss
            recognition_loss = (
                recognition_loss_function(
                    gloss_probabilities,
                    batch.gls,
                    batch.sgn_lengths.long(),
                    batch.gls_lengths.long(),
                )
                * recognition_loss_weight
            )
        else:
            recognition_loss = None

        if self.do_translation:
            assert decoder_outputs is not None
            word_outputs, _, _, _ = decoder_outputs
            # Calculate Translation Loss
            txt_log_probs = F.log_softmax(word_outputs, dim=-1)
            translation_loss = (
                translation_loss_function(txt_log_probs, batch.txt)
                * translation_loss_weight
            )
        else:
            translation_loss = None

        return recognition_loss, translation_loss

    def run_batch(
        self,
        batch: Batch,
        recognition_beam_size: int = 1,
        translation_beam_size: int = 1,
        translation_beam_alpha: float = -1,
        translation_max_output_length: int = 100,
    ) -> (np.array, np.array, np.array):
        """
        Get outputs and attentions scores for a given batch

        :param batch: batch to generate hypotheses for
        :param recognition_beam_size: size of the beam for CTC beam search
            if 1 use greedy
        :param translation_beam_size: size of the beam for translation beam search
            if 1 use greedy
        :param translation_beam_alpha: alpha value for beam search
        :param translation_max_output_length: maximum length of translation hypotheses
        :return: stacked_output: hypotheses for batch,
            stacked_attention_scores: attention scores for batch
        """

        encoder_output, encoder_hidden = self.encode(
            sgn=batch.sgn, sgn_mask=batch.sgn_mask, sgn_length=batch.sgn_lengths
        )

        if self.do_recognition:
            # Gloss Recognition Part
            # N x T x C
            gloss_scores = self.gloss_output_layer(encoder_output)
            # N x T x C
            gloss_probabilities = gloss_scores.log_softmax(2)
            # Turn it into T x N x C
            gloss_probabilities = gloss_probabilities.permute(1, 0, 2)
            gloss_probabilities = gloss_probabilities.cpu().detach().numpy()
            tf_gloss_probabilities = np.concatenate(
                (gloss_probabilities[:, :, 1:], gloss_probabilities[:, :, 0, None]),
                axis=-1,
            )

            assert recognition_beam_size > 0
            ctc_decode, _ = tf.nn.ctc_beam_search_decoder(
                inputs=tf_gloss_probabilities,
                sequence_length=batch.sgn_lengths.cpu().detach().numpy(),
                beam_width=recognition_beam_size,
                top_paths=1,
            )
            ctc_decode = ctc_decode[0]
            # Create a decoded gloss list for each sample
            tmp_gloss_sequences = [[] for i in range(gloss_scores.shape[0])]
            for (value_idx, dense_idx) in enumerate(ctc_decode.indices):
                tmp_gloss_sequences[dense_idx[0]].append(
                    ctc_decode.values[value_idx].numpy() + 1
                )
            decoded_gloss_sequences = []
            for seq_idx in range(0, len(tmp_gloss_sequences)):
                decoded_gloss_sequences.append(
                    [x[0] for x in groupby(tmp_gloss_sequences[seq_idx])]
                )
        else:
            decoded_gloss_sequences = None

        if self.do_translation:
            # greedy decoding
            if translation_beam_size < 2:
                stacked_txt_output, stacked_attention_scores = greedy(
                    encoder_hidden=encoder_hidden,
                    encoder_output=encoder_output,
                    src_mask=batch.sgn_mask,
                    embed=self.txt_embed,
                    bos_index=self.txt_bos_index,
                    eos_index=self.txt_eos_index,
                    decoder=self.decoder,
                    max_output_length=translation_max_output_length,
                )
                # batch, time, max_sgn_length
            else:  # beam size
                stacked_txt_output, stacked_attention_scores = beam_search(
                    size=translation_beam_size,
                    encoder_hidden=encoder_hidden,
                    encoder_output=encoder_output,
                    src_mask=batch.sgn_mask,
                    embed=self.txt_embed,
                    max_output_length=translation_max_output_length,
                    alpha=translation_beam_alpha,
                    eos_index=self.txt_eos_index,
                    pad_index=self.txt_pad_index,
                    bos_index=self.txt_bos_index,
                    decoder=self.decoder,
                )
        else:
            stacked_txt_output = stacked_attention_scores = None

        return decoded_gloss_sequences, stacked_txt_output, stacked_attention_scores

    def __repr__(self) -> str:
        """
        String representation: a description of encoder, decoder and embeddings

        :return: string representation
        """
        return (
            "%s(\n"
            "\tencoder=%s,\n"
            "\tdecoder=%s,\n"
            "\tsgn_embed=%s,\n"
            "\ttxt_embed=%s)"
            % (
                self.__class__.__name__,
                self.encoder,
                self.decoder,
                self.sgn_embed,
                self.txt_embed,
            )
        )

def build_ensemble_model(
    cfg: dict,
    gls_vocab: GlossVocabulary,
    txt_vocab: TextVocabulary,
    do_recognition: bool = True,
    do_translation: bool = True,
) -> SignModelEnsemble:
    """
    Builds the Time-Aligned Late Fusion model.
    """
    # 1. Build TimeAlignmentModule
    align_cfg = cfg["model"]["alignment"]
    time_aligner = TimeAlignmentModule(
        pose_dim=align_cfg["pose_dim"],
        hand_dim=align_cfg["hand_dim"],
        mouth_dim=align_cfg["mouth_dim"],
        hidden_dim=align_cfg["hidden_dim"],
        target_len=align_cfg["target_len"],
    )

    # 2. Build individual encoders
    # The input dimension for these encoders is the output of the alignment module
    encoder_input_dim = align_cfg["hidden_dim"]
    
    encoders = []
    # gloss_output_layers = []
    # decoders = []
    # sgn_embeds = []
    for _ in range(3): # For pose, hand, mouth
        # NOTE: Encoder config needs to be adapted to not have SpatialEmbeddings
        # but to take the aligned features directly.
        # Assuming TransformerEncoder is used.
        encoder_cfg = cfg["model"]["encoder"]
        encoders.append(
            TransformerEncoder(
                **encoder_cfg,
                input_size=encoder_input_dim, # Important: input size matches aligner output
                emb_size=encoder_input_dim
            )
        )

    '''
    txt_embed = models[0].txt_embed
    gls_vocab = models[0].gls_vocab
    txt_vocab = models[0].txt_vocab
    do_recognition = models[0].do_recognition
    do_translation = models[0].do_translation

    for model in models:
        encoders.append(model.encoder)
        gloss_output_layers.append(model.gloss_output_layer)
        decoders.append(model.decoder)
        sgn_embeds.append(model.sgn_embed)
    '''

    # 3. Build other components (Decoders, output layers, etc.)
    # This part is similar to the original implementation
    txt_embed = Embeddings(
        **cfg["model"]["txt_embeddings"],
        vocab_size=len(txt_vocab),
        padding_idx=txt_vocab.stoi[PAD_TOKEN],
    )
    
    decoders = []
    gloss_output_layers = []
    for _ in range(len(encoders)):
        decoder_cfg = cfg["model"]["decoder"]
        decoders.append(
            TransformerDecoder(
                **decoder_cfg,
                encoder=encoders[_], # Each decoder is tied to an encoder
                vocab_size=len(txt_vocab),
                pad_index=txt_vocab.stoi[PAD_TOKEN],
                bos_index=txt_vocab.stoi[BOS_TOKEN],
                eos_index=txt_vocab.stoi[EOS_TOKEN],
            )
        )
        gloss_output_layers.append(
            nn.Linear(encoders[_].output_size, len(gls_vocab))
        )

    # 4. Build the final ensemble model
    model = SignModelEnsemble(
        time_aligner=time_aligner,
        encoders=nn.ModuleList(encoders),
        gloss_output_layers=nn.ModuleList(gloss_output_layers),
        decoders=nn.ModuleList(decoders),
        txt_embed=txt_embed,
        gls_vocab=gls_vocab,
        txt_vocab=txt_vocab,
        do_recognition=do_recognition,
        do_translation=do_translation,
    )

    # Initialize model parameters
    if cfg["training"].get("load_model", None) is None:
        initialize_model(model, cfg)

    return model



def build_model(
    cfg: dict,
    sgn_dim: int,
    gls_vocab: GlossVocabulary,
    txt_vocab: TextVocabulary,
    do_recognition: bool = True,
    do_translation: bool = True,
) -> SignModel:
    """
    Build and initialize the model according to the configuration.

    :param cfg: dictionary configuration containing model specifications
    :param sgn_dim: feature dimension of the sign frame representation, i.e. 2560 for EfficientNet-7.
    :param gls_vocab: sign gloss vocabulary
    :param txt_vocab: spoken language word vocabulary
    :return: built and initialized model
    :param do_recognition: flag to build the model with recognition output.
    :param do_translation: flag to build the model with translation decoder.
    """

    txt_padding_idx = txt_vocab.stoi[PAD_TOKEN]
    sgn_embed: SpatialEmbeddings = SpatialEmbeddings(
        **cfg["encoder"]["embeddings"],
        num_heads=cfg["encoder"]["num_heads"],
        input_size=sgn_dim,
    )

    # build encoder
    enc_dropout = cfg["encoder"].get("dropout", 0.0)
    enc_emb_dropout = cfg["encoder"]["embeddings"].get("dropout", enc_dropout)
    if cfg["encoder"].get("type", "recurrent") == "transformer":
        assert (
            cfg["encoder"]["embeddings"]["embedding_dim"]
            == cfg["encoder"]["hidden_size"]
        ), "for transformer, emb_size must be hidden_size"

        encoder = TransformerEncoder(
            **cfg["encoder"],
            emb_size=sgn_embed.embedding_dim,
            emb_dropout=enc_emb_dropout,
        )
    else:
        encoder = RecurrentEncoder(
            **cfg["encoder"],
            emb_size=sgn_embed.embedding_dim,
            emb_dropout=enc_emb_dropout,
        )

    if do_recognition:
        gloss_output_layer = nn.Linear(encoder.output_size, len(gls_vocab))
        if cfg["encoder"].get("freeze", False):
            freeze_params(gloss_output_layer)
    else:
        gloss_output_layer = None

    # build decoder and word embeddings
    if do_translation:
        txt_embed: Union[Embeddings, None] = Embeddings(
            **cfg["decoder"]["embeddings"],
            num_heads=cfg["decoder"]["num_heads"],
            vocab_size=len(txt_vocab),
            padding_idx=txt_padding_idx,
        )
        dec_dropout = cfg["decoder"].get("dropout", 0.0)
        dec_emb_dropout = cfg["decoder"]["embeddings"].get("dropout", dec_dropout)
        if cfg["decoder"].get("type", "recurrent") == "transformer":
            decoder = TransformerDecoder(
                **cfg["decoder"],
                encoder=encoder,
                vocab_size=len(txt_vocab),
                emb_size=txt_embed.embedding_dim,
                emb_dropout=dec_emb_dropout,
            )
        else:
            decoder = RecurrentDecoder(
                **cfg["decoder"],
                encoder=encoder,
                vocab_size=len(txt_vocab),
                emb_size=txt_embed.embedding_dim,
                emb_dropout=dec_emb_dropout,
            )
    else:
        txt_embed = None
        decoder = None
    model: SignModel = SignModel(
        encoder=encoder,
        gloss_output_layer=gloss_output_layer,
        decoder=decoder,
        sgn_embed=sgn_embed,
        txt_embed=txt_embed,
        gls_vocab=gls_vocab,
        txt_vocab=txt_vocab,
        do_recognition=do_recognition,
        do_translation=do_translation,
    )


    if do_translation:
        # tie softmax layer with txt embeddings
        if cfg.get("tied_softmax", False):
            # noinspection PyUnresolvedReferences
            if txt_embed.lut.weight.shape == model.decoder.output_layer.weight.shape:
                # (also) share txt embeddings and softmax layer:
                # noinspection PyUnresolvedReferences
                model.decoder.output_layer.weight = txt_embed.lut.weight
            else:
                raise ValueError(
                    "For tied_softmax, the decoder embedding_dim and decoder "
                    "hidden_size must be the same."
                    "The decoder must be a Transformer."
                )

    # custom initialization of model parameters
    initialize_model(model, cfg, txt_padding_idx)

    return model
