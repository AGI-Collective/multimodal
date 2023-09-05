import torch
import torch.nn as nn
from torchtyping import TensorType, patch_typeguard
from einops import rearrange

from typing import Callable, Union
import timm
import open_clip
from functools import partial
import logging
import math

import einops
import torch
from einops import rearrange
from timm.models.layers.drop import DropPath
from torch import nn
from torch.nn import LayerNorm, Linear, MultiheadAttention

from dinov2.models import DinoVisionTransformer
from dinov2.models.vision_transformer import vit_large, vit_small, vit_base
import dinov2.layers as layers

# Temporal Attention
class TemporalAttention(nn.Module):

    """perform temporal self-attention"""

    def __init__(self, input_dim=768, droppath_rate=0.1):
        """

        Kwargs:
            input_dim (int): The input feature dimension.


        """
        super().__init__()

        self._input_dim = input_dim
        self.temporal_attn = MultiheadAttention(input_dim, num_heads=input_dim // 64)
        self.norm = LayerNorm(input_dim, eps=1e-12)
        self.linear = Linear(input_dim, input_dim)
        self.droppath = DropPath(droppath_rate)
        self.scale = nn.parameter.Parameter(torch.zeros([]))

    def forward(self, x: torch.Tensor):
        """forward

        Args:
            x (torch.Tensor): input features. Shape: [bs, nframes, l, c]. l = 1 + h*w

        Returns: features after adapter. The same shape as input.

        """
        if x.shape[1] == 1:  # for single frame, return itself.
            return x

        shortcut = x
        x = einops.rearrange(x, "b t l c -> t (b l) c")
        x = self.norm(x)
        x = self.temporal_attn(x, x, x)[0]
        x = einops.rearrange(x, "t (b l) c -> b t l c", b=shortcut.shape[0])
        return shortcut + self.scale * self.droppath(x)

class TemporalAdapter(nn.Module):
    def __init__(self, block):
        self.temporal_attention = TemporalAttention()
        self.block = block
    
    def forward(self, x):
        x = self.temporal_attention(x)
        return self.block(x)

class DinoWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def add_temporal_attention(self):
        for child_name, child in self.model.named_children():
            if isinstance(child, layers.Block):
                new = TemporalAdapter(child)
                setattr(self.model, child_name, new)
            else:
                self.add_temporal_attention(child)
    
    def forward(self, x):
        return self.model(x)

ENCODER_OUT_DIMS = {
    "clip": 512,
    "openclip-H": 1024,
}

ENCODER_SEQ_LENS = {
    "openclip-H": 257
}

def get_vision_encoder(
    name: str, 
    device: Union[torch.device, str] = None, 
    pretrained: bool = False,
    cache_path: str = None
) -> torch.nn.Module:
    """
    Loads vision encoder module
    """
    if name == "dinov2":
        dino_model = vit_small(#Could also be vit_base, vit_large, vit_giant
            patch_size=14,
            img_size=526,
            init_values=1.0,
            block_chunks=0)
        if pretrained = True:
            #Load pretrained model from where you saved it before.
            dino_model.load_state_dict(torch.load("./model.pt"))
        encoder = DinoWrapper(dino_model)
    else:
        raise ValueError(f"vision encoder {name} not recognized")
    return encoder

# Vision encoder 
class visionEncoder(nn.Module):

    """
    Takes in a batch of visions and returns a batch of embeddings of the
    same dimensions as the LM's word embeddings.

    :param config: Neox args
    :param out_dim: output dimension of the embedding
    :param device: device to run the model on
    """

    def __init__(
        self,
        config,
        out_dim: int = 2048,
        device=None,
    ):
        super().__init__()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.config = config
        self.encoder_type = config.encoder_name

        # get vision encoder backbone
        self.enc = get_vision_encoder(
            config.encoder_name,
            # device=self.device,
            pretrained=config.pretrained_img_encoder,
            cache_path = config.load_clip
        )

        # Add temporal attention into the vision encoder
        self.enc.add_temporal_attention()

        self.encoder_out_dim = ENCODER_OUT_DIMS[
            self.encoder_type
        ]  # out dim for vision encoder

        self.out_dim = out_dim  # out dim for lm

        self.proj = nn.Linear(self.encoder_out_dim, self.out_dim)
        self.dropout = nn.Dropout(config.image_embed_dropout_prob)
        self.use_layernorm = config.use_image_embed_layernorm
        if self.use_layernorm:
            self.ln = nn.LayerNorm(self.out_dim)

    '''
    B, N, T, H, W, C
    B*N*T, H, W, C
    '''
    def forward(
        self, x: TensorType["b", "c", "h", "w"]
    ) -> TensorType["b", "seq", "out_dim"]:

        # pass through image encoder
        logits = self.enc(x)

        # remove trailing dimensions of size 1 + pass through linear
        if logits.ndim == 4:
            logits = rearrange(logits, "b d 1 1 -> b d")
        elif logits.ndim == 3:
            assert self.encoder_type in ENCODER_SEQ_LENS
        else:
            assert logits.ndim == 2

        logits = self.proj(logits)

        # reshape to desired output shape
        if (
            self.encoder_type not in ENCODER_SEQ_LENS
        ):  # don't need to reshape those with fixed seq lens / no pooling
            logits = rearrange(
                logits, "b (s d) -> b s d", d=self.out_dim, s=self.out_seq_len
            )

        # pass through dropout and layer norm
        logits = self.dropout(logits)

        if self.use_layernorm:
            logits = self.ln(logits)

        # Added for shape mismatch.
        if logits.ndim == 2:
            logits = logits.unsqueeze(1)

        return logits
