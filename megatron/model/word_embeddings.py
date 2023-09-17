# Copyright (c) 2021, EleutherAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import math
from torch.nn.parameter import Parameter

from megatron import mpu
from megatron.model.positional_embeddings import SinusoidalPositionalEmbedding
from megatron.model.init_functions import get_init_methods
# from megatron.model.encoders.image_encoders import ImageEncoder
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests

import einops
from einops import rearrange

from megatron.utils import MODALITY_DICT

class Embedding(torch.nn.Module):
    """Language model embeddings.
    Arguments:
        hidden_size: hidden size
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        init_method: weight initialization method
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(
        self,
        neox_args,
        hidden_size,
        vocab_size,
        max_sequence_length,
        embedding_dropout_prob,
        init_method,
        num_tokentypes=0,
        use_pos_emb=True,
    ):
        super(Embedding, self).__init__()

        self.hidden_size = hidden_size
        self.init_method = init_method
        self.num_tokentypes = num_tokentypes
        self.use_mup = neox_args.use_mup
        self.mup_embedding_mult = neox_args.mup_embedding_mult
        self.mup_rp_embedding_mult = neox_args.mup_rp_embedding_mult

        # Word embeddings (parallel).
        self.word_embeddings = mpu.VocabParallelEmbedding(
            neox_args=neox_args,
            num_embeddings=vocab_size,
            embedding_dim=self.hidden_size,
            init_method=self.init_method,
        )
        self._word_embeddings_key = "word_embeddings"

        if neox_args.use_bnb_optimizer:
            try:
                import bitsandbytes as bnb

                self.embedding_module = bnb.nn.StableEmbedding
            except ModuleNotFoundError:
                print(
                    "Please install bitsandbytes following https://github.com/facebookresearch/bitsandbytes."
                )
                raise Exception
        else:
            self.embedding_module = torch.nn.Embedding

        # Position embedding (serial).
        self.use_pos_emb = use_pos_emb
        if self.use_pos_emb:
            self.embedding_type = neox_args.pos_emb
            if self.embedding_type == "learned":
                self.position_embeddings = self.embedding_module(
                    max_sequence_length, self.hidden_size
                )
                self._position_embeddings_key = "position_embeddings"
                # Initialize the position embeddings.
                self.init_method(self.position_embeddings.weight)
            elif self.embedding_type == "sinusoidal":
                self.position_embeddings = SinusoidalPositionalEmbedding(
                    self.hidden_size
                )

        # Token type embedding.
        # Add this as an optional field that can be added through
        # method call so we can load a pretrain model without
        # token types and add them as needed.
        self._tokentype_embeddings_key = "tokentype_embeddings"
        if self.num_tokentypes > 0:
            self.tokentype_embeddings = self.embedding_module(
                self.num_tokentypes, self.hidden_size
            )
            # Initialize the token-type embeddings.
            self.init_method(self.tokentype_embeddings.weight)
        else:
            self.tokentype_embeddings = None

        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)
        self.opt_pos_emb_offset = neox_args.opt_pos_emb_offset

        # For ticking position ids forward
        self.layer_past = None

    def add_tokentype_embeddings(self, num_tokentypes):
        """Add token-type embedding. This function is provided so we can add
        token-type embeddings in case the pretrained model does not have it.
        This allows us to load the model normally and then add this embedding.
        """
        if self.tokentype_embeddings is not None:
            raise Exception("tokentype embeddings is already initialized")
        if torch.distributed.get_rank() == 0:
            print(
                "adding embedding for {} tokentypes".format(num_tokentypes), flush=True
            )
        self.num_tokentypes = num_tokentypes
        self.tokentype_embeddings = self.embedding_module(
            num_tokentypes, self.hidden_size
        )
        # Initialize the token-type embeddings.
        self.init_method(self.tokentype_embeddings.weight)

    def forward(self, input_ids, position_ids, tokentype_ids=None):
        # Embeddings.
        words_embeddings = self.word_embeddings(input_ids)
        if self.use_pos_emb and self.embedding_type in ["learned", "sinusoidal"]:
            if self.opt_pos_emb_offset:
                if self.layer_past is not None:
                    position_ids = position_ids + self.layer_past + 1
                self.layer_past = position_ids[:, -1]
                # OPT always adds 2 for some reason, according to the HF implementation
                position_ids = position_ids + self.opt_pos_emb_offset
            position_embeddings = self.position_embeddings(position_ids)
            position_embeddings.mul_(self.mup_rp_embedding_mult)
            embeddings = words_embeddings + position_embeddings
        else:
            embeddings = words_embeddings
        if tokentype_ids is not None:
            assert self.tokentype_embeddings is not None
            embeddings = embeddings + self.tokentype_embeddings(tokentype_ids)
        else:
            assert self.tokentype_embeddings is None

        # Dropout.
        embeddings = self.embedding_dropout(embeddings)

        if self.use_mup:
            with torch.no_grad():
                embeddings.mul_(self.mup_embedding_mult)

        return embeddings

# Simple module that takes as input a batch of images [B, T, H, W, C] and returns a batch of embeddings [B, T, E]
class ImageEncoder(torch.nn.Module):
    # Simple image encoder that flattens the image and does a forward pass through a linear layer

    def __init__(self, neox_args):
        super().__init__()
        self.neox_args = neox_args
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.model = AutoModel.from_pretrained('facebook/dinov2-base')
        self.encoder_layer = torch.nn.Linear(768, neox_args.hidden_size)
        url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
        image = Image.open(requests.get(url, stream=True).raw)
        self.image = image


    def forward(self, images):
        # images is [B, T, F, C, H, W]
        num_images = images.shape[1]
        original_batch_size = images.shape[0]
        images = rearrange(images, "b t f c h w -> (b t) f c h w")

        batch_size = images.shape[0]
        max_seq_length = images.shape[1]
        # images = images.reshape(batch_size, max_seq_length, -1)
        # print("Shape of images", images.shape)

        # images = images.reshape((images.shape[0] * images.shape[1], images.shape[2], images.shape[3], images.shape[4]))
        images_2 = self.processor(self.image, return_tensors="pt", padding=True)
        # print(images_2)
        image = images_2["pixel_values"]
        #replicate the image to match the batch size
        image = image.repeat(images.shape[0] * images.shape[1], 1, 1, 1)
        images_2["pixel_values"] = image
        # Convert to image to c10::Half
        images_2["pixel_values"] = images_2["pixel_values"].half().to(self.model.device)
        output = self.model(**images_2).last_hidden_state
        # images = images.reshape()
        output = output[:, 0, :]
        embeddings = self.encoder_layer(output)
        embeddings = embeddings.reshape(batch_size, max_seq_length, -1)  
        embeddings = rearrange(embeddings, "(b t) n f -> b t n f", b=original_batch_size, t=num_images)
        return embeddings

class EmbeddingPipe(Embedding):
    """Extends Embedding to forward attention_mask through the pipeline."""

    def __init__(self, neox_args, *args, **kwargs):
        super().__init__(neox_args, *args, **kwargs)
        self.image_encoder = ImageEncoder(neox_args)
        self.seq_length = neox_args.seq_length

    @property
    def word_embeddings_weight(self):
        """Easy accessory for the pipeline engine to tie embeddings across stages."""
        return self.word_embeddings.weight

    def forward(self, args):
        assert (
            len(args) == 5
        ), f"Expected 3 arguments (input_ids, vision_input, multimodal_position_ids, position_ids, attention_mask), but got {len(args)}."

        input_ids = args[0]
        vision_input = args[1]
        multimodal_position_ids = args[2] # [B, T]
        assert self.seq_length == torch.max(multimodal_position_ids)+1
        position_ids = args[3]
        attention_mask = args[4]
        
        word_embeddings = super().forward(input_ids, position_ids) # [B, T, E]
        # Vision Input is [B, T, F, C, H, W]
        image_embeddings = self.image_encoder(vision_input) # [B, T, N, E] where N=1 for now 
        image_embeddings = rearrange(image_embeddings, "b t n e -> b (t n) e") # [B, T*N, E]
        
        # Concatenate the embeddings
        all_embeddings = torch.cat([word_embeddings, image_embeddings], dim=1) # [B, T + T*N, E]

        # Rearrange the embeddigs based on multimodal position ids

        # The following code is much faster but equivalent to : 
        # torch.gather(all_embeddings, 1, multimodal_position_ids.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        batch_size = all_embeddings.shape[0]
        time_steps = all_embeddings.shape[1]
        batch_indices = torch.arange(batch_size).view(-1, 1).expand(-1, time_steps)
        all_embeddings = all_embeddings[batch_indices, multimodal_position_ids]
        # print(multimodal_position_ids[batch_indices, multimodal_position_ids].shape, multimodal_position_ids[batch_indices, multimodal_position_ids])
        if multimodal_position_ids.shape[1] > self.seq_length:
            assert all(multimodal_position_ids[batch_indices, multimodal_position_ids][:, self.seq_length:]) == -1, "Multimodal position ids should be -1 after the sequence length"
        return all_embeddings, attention_mask


class SoftEmbedding(torch.nn.Module):
    def __init__(
        self,
        neox_args,
        wte,
        n_tokens: int = 10,
        init_range: float = 0.5,
        init_string: str = "",
    ):
        super(SoftEmbedding, self).__init__()
        self.n_tokens = n_tokens
        self.neox_args = neox_args
        self.init_range = init_range
        self.init_string = init_string
        self.soft_embedding_weight = torch.nn.parameter.Parameter(
            self.initialize_embedding(wte)
        )

    def initialize_embedding(self):
        if self.init_string:
            embeds = torch.LongTensor(
                self.neox_args.tokenizer.tokenize(self.init_string)
            ).to(self.embedding_module.weight.device)
            embeds = self.embedding_module(embeds)
            if embeds.shape[0] >= self.n_tokens:
                embeds = embeds[: self.n_tokens, :]  # slice
            else:
                embeds = embeds.repeat(math.ceil(self.n_tokens / embeds.shape[0]), 1)[
                    : self.n_tokens, :
                ]  # pad up to n_tokens
            return embeds
        return torch.Tensor(n_tokens, neox_args.hidden_size).uniform_(
            -self.random_range, self.random_range
        )

    def forward(self, args: tuple):
        in_inference = len(args) == 3  # embeddings, layer_past, attention_mask
        in_train = len(args) == 2  # embeddings, attention_mask
        if in_train:
            embedding, attention_mask = args
        else:
            embedding, layer_past, attention_mask = args
        soft_embedding = self.soft_embedding_weight.repeat(
            embedding.shape[0], 1, 1
        )  # repeat batch_size times
        if in_train:
            # append soft embedding at the beginning in training
            embedding = torch.cat((soft_embedding, embedding), dim=1)
            embedding = embedding[:, : self.neox_args.seq_length, ...]
            return embedding, attention_mask
        else:
            if not (exists(layer_past) and layer_past.numel() > 0):
                # if in inference, on the first forward pass, we want to do the same as in training (append soft embedding)
                embedding = torch.cat((soft_embedding, embedding), dim=1)
                embedding = embedding[:, : self.neox_args.seq_length, ...]
            # otherwise, we're in incremental mode, and just want to forward the single embedding (since the soft prompt has already been cached)
            return embedding, layer_past, attention_mask
