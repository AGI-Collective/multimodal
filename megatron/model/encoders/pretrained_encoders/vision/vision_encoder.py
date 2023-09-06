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
from loralib.layers import Linear as LoraLinear
from loralib.layers import MergedLinear as MergedLoraLinear
from loralib.layers import Conv2d as LoraConv2d

import torch
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import einsum, nn

from perceiver import PerceiverResampler
from encoder import add_lora
from encoder import recursive_freeze_unfreeze

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
    def __init__(self, encoder, config):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.prepare_encoder()
        
    def add_temporal_attention(self):
        for child_name, child in self.encoder.named_children():
            if isinstance(child, layers.Block):
                new = TemporalAdapter(child)
                setattr(self.encoder, child_name, new)
            else:
                self.add_temporal_attention(child)
    
    def freeze_model(self):
        num_layers_to_unfreeze = self.config.num_layers_to_unfreeze
        
        # Freeze everything
        self.encoder.requires_grad_(False)

        # Unfreeze last num_layers_to_unfreeze layers
        for child_name, child in list(self.encoder.named_modules())[-num_layers_to_unfreeze:]:
            child.requires_grad_(True)

        # Unfreeze LayerNorm
        recursive_freeze_unfreeze(self.encoder, ['LayerNorm'], freeze=False)

        # What about cls token? TODO
    
    def prepare_encoder(self):
        if self.config.freeze_encoder:
            self.freeze_model()
        self.add_temporal_attention()
        if self.config.add_lora:
            add_lora(self.encoder)

    
    def forward(self, x):
        '''
        x: (b, t, f, c, h, w)
        '''
        b, t, f, c, h, w = x.shape # b=batch size, t=number of images/videos in a sample, f=number of frames in each image/video, c=number of channels, h=height, w=width
        x = rearrange(x, "b t f c h w -> (b t) f c h w")
        embeddings = self.encoder(x) # B*T, N_E, E
        embeddings = rearrange(embeddings, "(b t) v d -> b t v d", b=b, t=t, v=f)
        return embeddings

def get_vision_encoder(
    name: str, 
    device: Union[torch.device, str] = None, 
    pretrained: bool = False,
    load_path: str = None
) -> torch.nn.Module:
    """
    Loads vision encoder module
    """
    if name == "dinov2_small":
        dino_model = vit_small(#Could also be vit_base, vit_large, vit_giant
            patch_size=14,
            img_size=526,
            init_values=1.0,
            block_chunks=0)
        if pretrained == True:
            #Load pretrained model from where you saved it before.
            dino_model.load_state_dict(torch.load(load_path))
        encoder = DinoWrapper(dino_model)
    else:
        raise ValueError(f"vision encoder {name} not recognized")
    return encoder