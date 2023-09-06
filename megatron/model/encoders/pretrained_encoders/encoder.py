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

from loralib.layers import Linear as LoraLinear
from loralib.layers import MergedLinear as MergedLoraLinear
from loralib.layers import Conv2d as LoraConv2d

import torch
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import einsum, nn

from perceiver import PerceiverResampler

from .audio_encoders import get_audio_encoder
from .vision_encoders import get_vision_encoder


def add_lora(self, model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.Linear) and child.requires_grad == False:
            weight = child.weight
            bias = child.bias
            if child_name == 'qkv':
                new = MergedLoraLinear(child.in_features, child.out_features, 
                            r = 128, dtype = torch.float32, enable_lora = [True,True,True])
            else:
                new = LoraLinear(child.in_features, child.out_features, r = 128, dtype = torch.float32)
            
            new.weight = weight
            new.bias = bias
            setattr(model, child_name, new)
        elif isinstance(child, nn.Conv2d):
            weight = child.weight
            bias = child.bias
            new = LoraConv2d(child.in_channels, child.out_channels, child.kernel_size[0], r = 128, dtype = torch.float32)                     
            new.weight = weight
            new.bias = bias
            new.stride = child.stride
            new.padding = child.padding
            new.dilation = child.dilation
            setattr(model, child_name, new)
        else:
            self.add_lora(child)

def recursive_freeze_unfreeze(self, model, param_types, freeze=True):
    for child_name, child in model.named_children():
        child_class_name = child.__class__.__name__
        if str(child_class_name) in param_types:
            child.requires_grad = not freeze
        else:
            self.recursive_freeze_unfreeze(child, param_types, freeze)

ENCODER_OUT_DIMS = {
    "clip": 512,
    "openclip-H": 1024,
    "dinov2_base": 768, 
    "dinov2_large": 1024, 
    "dinov2_small": 384,
}

ENCODER_SEQ_LENS = {
    "openclip-H": 257,
    "dinov2_base": 257, 
    "dinov2_large": 257, 
    "dinov2_small": 257,
}

# Vision encoder 
class Encoder(nn.Module):

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
        self.modality = config.modality 
        if self.modality == "vision":
            self.encoder = get_vision_encoder(config.encoder_name, load_path=config.load_vision_encoder_path)
        elif self.modality == "audio":
            self.encoder = get_audio_encoder(config.encoder_name, load_path=config.load_audio_encoder_path)
        else:
            raise ValueError(f"modality {self.modality} not recognized")

        self.encoder_out_dim = ENCODER_OUT_DIMS[
            self.encoder_type
        ]  # out dim for vision encoder
        self.encoder_seq_len = ENCODER_SEQ_LENS[
            self.encoder_type
        ]
        self.out_dim = out_dim  # out dim for lm

        self.proj = nn.Linear(self.encoder_out_dim, self.out_dim)
        self.dropout = nn.Dropout(config.embed_dropout_prob)
        self.use_layernorm = config.use_embed_layernorm
        if self.use_layernorm:
            self.ln = nn.LayerNorm(self.out_dim)
        
        self.perceiver = PerceiverResampler(dim=config.encoder_seq_length)
    
    def forward(
        self, x: TensorType["b", "c", "h", "w"] or TensorType["b", "seq", "c", "h", "w"]
    ) -> TensorType["b", "seq", "out_dim"]:

        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """
        embeddings = self.encoder(x) # B, T, N_E, E
        B, T, N_E, E = embeddings.shape
        assert N_E == self.encoder_seq_len
        
        embeddings = rearrange(embeddings, "b t n_e e -> (b t) n_e e")
        embeddings = self.perceiver(embeddings) # (B*T, N_E_new, E)

        logits = self.proj(logits) # (B*T, N_E_new, E_L)
        logits = self.dropout(logits) # (B*T, N_E_new, E_L)

        if self.use_layernorm:
            logits = self.ln(logits) # (B*T, N_E_new, E_L)

        logits = rearrange(logits, "(b t) n_e e_l -> b t n_e e_l", b=B, t=T) # (B, T, N_E_new, E_L)
    
        return logits