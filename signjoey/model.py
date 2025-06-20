# coding: utf-8
# import tensorflow as tf

# tf.config.set_visible_devices([], "GPU")

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from itertools import groupby
from signjoey.initialization import initialize_model
from signjoey.embeddings import Embeddings, SpatialEmbeddings
from signjoey.encoders import Encoder, RecurrentEncoder, TransformerEncoder
from signjoey.decoders import Decoder, RecurrentDecoder, TransformerDecoder
from signjoey.search import beam_search, greedy
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
        features: List[Tensor] = None,
        feature_lengths: List[Tensor] = None,
    ) -> (Tensor, Tensor, Tensor, Tensor):
        """
        First encodes the source sentence.
        Then produces the target one word at a time.

        :param sgn: source input (either a single tensor or None if features are used)
        :param sgn_mask: source mask
        :param sgn_lengths: length of source inputs
        :param txt_input: target input
        :param txt_mask: target mask
        :param features: list of source feature tensors
        :param feature_lengths: list of source feature lengths
        :return: decoder outputs
        """
        if features is not None:
            # Use the new multi-stream feature processing
            encoder_output, encoder_hidden = self.encode(
                features=features,
                feature_lengths=feature_lengths,
                sgn_mask=sgn_mask, # sgn_mask might still be needed by the decoder
            )
        else:
            # Use the original single-stream processing
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
        self, 
        sgn: Tensor = None, 
        sgn_mask: Tensor = None, 
        sgn_length: Tensor = None,
        features: List[Tensor] = None,
        feature_lengths: List[Tensor] = None,
    ) -> (Tensor, Tensor):
        """
        Encodes the source sentence.
        Can handle either a single tensor or a list of feature tensors.

        :param sgn:
        :param sgn_mask:
        :param sgn_length:
        :param features:
        :param feature_lengths:
        :return: encoder outputs (output, hidden_concat)
        """
        if features is not None:
            # Late-fusion path
            return self.encoder(
                features=features,
                feature_lengths=feature_lengths,
                sgn_embed=self.sgn_embed, # Pass embedding module to the encoder
                mask=sgn_mask,
            )
        else:
            # Standard path
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
            features=batch.features,
            feature_lengths=batch.feature_lengths,
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
            if 1, greedy decoding is applied
        :param translation_beam_size: size of the beam for translation beam search
            if 1, greedy decoding is applied
        :param translation_beam_alpha: alpha value for length penalty
        :param translation_max_output_length: maximum length for translation hypotheses
        :return:
            - gloss_hypotheses: an array of post-processed gloss hypotheses
            - text_hypotheses: an array of post-processed translation hypotheses
            - attention_scores: an array of attention scores
        """
        encoder_output, encoder_hidden = self.encode(
            features=batch.features,
            feature_lengths=batch.feature_lengths,
            sgn_mask=batch.sgn_mask,
        )

        # Gloss Recognition
        if self.do_recognition:
            # TF-based beam search requires a batch of size 1
            # assert batch.sgn.size(0) == 1, "Currently, only batch_size=1 for CTC is supported."

            gloss_scores = self.gloss_output_layer(encoder_output)
            gloss_probabilities = gloss_scores.log_softmax(2)

            # Move blank token from index 0 to last index for TensorFlow
            gloss_probabilities_np = gloss_probabilities.cpu().detach().numpy()
            tf_gloss_probabilities = np.concatenate(
                (gloss_probabilities_np[:, :, 1:], gloss_probabilities_np[:, :, 0, None]),
                axis=-1,
            )

            # Lazy import tensorflow to avoid CUDA context conflicts during DDP init
            import tensorflow as tf
            tf.config.set_visible_devices([], "GPU")

            assert recognition_beam_size > 0
            # TF expects T x N x C
            tf_gloss_probabilities = tf_gloss_probabilities.transpose(1, 0, 2)
            
            ctc_decode, _ = tf.nn.ctc_beam_search_decoder(
                inputs=tf_gloss_probabilities,
                sequence_length=batch.sgn_lengths.cpu().detach().numpy(),
                beam_width=recognition_beam_size,
                top_paths=1,
            )

            ctc_decode = ctc_decode[0]
            # Create a decoded gloss list for each sample
            tmp_gloss_hyp = [[] for _ in range(batch.num_seqs)]
            for (i, v) in zip(ctc_decode.indices, ctc_decode.values):
                tmp_gloss_hyp[i[0]].append(v)
            gloss_hypotheses = [
                self.gls_vocab.array_to_sentence(tmp_gloss_hyp[i])
                for i in range(batch.num_seqs)
            ]
        else:
            gloss_hypotheses = None

        # Spoken Language Translation
        if self.do_translation:
            if translation_beam_size == 1:
                # Greedy decoding
                decoded_words, attention_scores = greedy(
                    src_mask=batch.sgn_mask,
                    embed=self.txt_embed,
                    bos_index=self.txt_bos_index,
                    eos_index=self.txt_eos_index,
                    decoder=self.decoder,
                    encoder_output=encoder_output,
                    encoder_hidden=encoder_hidden,
                    max_output_length=translation_max_output_length,
                )
            else:
                # Beam search
                decoded_words, attention_scores = beam_search(
                    size=translation_beam_size,
                    src_mask=batch.sgn_mask,
                    embed=self.txt_embed,
                    bos_index=self.txt_bos_index,
                    eos_index=self.txt_eos_index,
                    pad_index=self.txt_pad_index,
                    decoder=self.decoder,
                    encoder_output=encoder_output,
                    encoder_hidden=encoder_hidden,
                    alpha=translation_beam_alpha,
                    max_output_length=translation_max_output_length,
                )
            text_hypotheses = self.txt_vocab.arrays_to_sentences(decoded_words)
        else:
            text_hypotheses = None
            attention_scores = None

        return gloss_hypotheses, text_hypotheses, attention_scores

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

    def count_params(self) -> int:
        """
        Count the number of trainable parameters in the model.
        :return:
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(
    cfg: dict,
    gls_vocab: GlossVocabulary,
    txt_vocab: TextVocabulary,
    do_recognition: bool = True,
    do_translation: bool = True,
) -> SignModel:
    """
    Build and set up model object.

    :param cfg: full configuration dictionary
    :param gls_vocab: gloss vocabulary
    :param txt_vocab: text vocabulary
    :param do_recognition: whether to build the recognition part of the model
    :param do_translation: whether to build the translation part of the model
    :return: built and initialized model
    """
    model_cfg = cfg["model"]
    
    # Assert values are within acceptable ranges
    assert model_cfg["embeddings"]["norm_type"] in ["batch", "layer", "none"]
    assert model_cfg["encoder"]["type"] in ["transformer", "recurrent"]
    assert model_cfg["decoder"]["type"] in ["transformer", "recurrent"]

    sgn_embed_cfg = model_cfg["embeddings"]
    encoder_cfg = model_cfg["encoder"]
    decoder_cfg = model_cfg["decoder"]

    # Late Fusion configuration
    fusion_cfg = encoder_cfg.get("late_fusion", None)
    use_fusion = fusion_cfg.get("enabled", False) if fusion_cfg else False

    sgn_embed = None
    if not use_fusion:
        # For single-stream, sgn_dim is required.
        sgn_dim = model_cfg.get("sgn_dim")
        if not sgn_dim:
            raise ValueError("`sgn_dim` must be specified in model config for non-fusion models.")
        
        sgn_embed = SpatialEmbeddings(
            **sgn_embed_cfg,
            num_heads=encoder_cfg["num_heads"],
        input_size=sgn_dim,
    )

    # Build Encoder
    enc_dropout = encoder_cfg.get("dropout", 0.0)
    enc_emb_dropout = encoder_cfg.get("emb_dropout", 0.0)
    if encoder_cfg["type"] == "transformer":
        if not use_fusion:
            assert (
                encoder_cfg["hidden_size"]
                == sgn_embed.embedding_dim
                == decoder_cfg["hidden_size"]
            ), "for transformer, sgn embedding, encoder and decoder dimensions must be the same"
        
        # Remove emb_dropout and dropout from encoder_cfg if they exist
        encoder_cfg = encoder_cfg.copy()
        encoder_cfg.pop("emb_dropout", None)
        encoder_cfg.pop("dropout", None)
        encoder = TransformerEncoder(
            **encoder_cfg,
            emb_dropout=enc_emb_dropout,
            dropout=enc_dropout,
            fusion_cfg=fusion_cfg,
        )
    else: # recurrent (fusion not implemented for recurrent)
        if use_fusion:
            raise NotImplementedError("Late fusion is not implemented for RecurrentEncoder.")
        encoder = RecurrentEncoder(
            **encoder_cfg,
            emb_size=sgn_embed.embedding_dim,
            emb_dropout=enc_emb_dropout,
            dropout=enc_dropout,
        )

    # Build Decoder
    dec_dropout = decoder_cfg.get("dropout", 0.1)
    dec_emb_dropout = decoder_cfg.get("emb_dropout", 0.1)
    if decoder_cfg["type"] == "transformer":
        # Remove emb_dropout and dropout from decoder_cfg if they exist
        decoder_cfg = decoder_cfg.copy()
        decoder_cfg.pop("emb_dropout", None)
        decoder_cfg.pop("dropout", None)
        
        decoder = TransformerDecoder(
            **decoder_cfg,
            encoder=encoder,
            vocab_size=len(txt_vocab),
            emb_dropout=dec_emb_dropout,
            dropout=dec_dropout,
        )
    else:  # recurrent
        decoder = RecurrentDecoder(
            **decoder_cfg,
            encoder=encoder,
            vocab_size=len(txt_vocab),
            emb_dropout=dec_emb_dropout,
            dropout=dec_dropout,
        )

    # Spoken Language Embeddings
    txt_embed: Embeddings = Embeddings(
        **model_cfg["embeddings"],
                vocab_size=len(txt_vocab),
        padding_idx=txt_vocab.stoi[PAD_TOKEN],
    )

    # Recognition Output Layer
    gloss_output_layer = nn.Linear(encoder.output_size, len(gls_vocab))
    if do_recognition:
        if "recognition_loss_weight" not in cfg["training"]:
            cfg["training"]["recognition_loss_weight"] = 1.0
        if cfg["training"]["recognition_loss_weight"] < 1.0:
            # Use a smaller representation for the gloss output layer
            gloss_output_layer = nn.Sequential(
                nn.Linear(encoder.output_size, encoder.output_size // 2),
                nn.ReLU(),
                nn.Linear(encoder.output_size // 2, len(gls_vocab)),
            )

    model = SignModel(
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

    if "load_weights" in model_cfg:
        model = load_model_from_file(model, model_cfg["load_weights"])
    
    # Freeze params
    if "freeze" in model_cfg:
        for name, child in model.named_children():
            if name in model_cfg["freeze"]:
                freeze_params(child)

    # Initialize weights
    initialize_model(model, model_cfg, txt_vocab.stoi[PAD_TOKEN])

    return model
