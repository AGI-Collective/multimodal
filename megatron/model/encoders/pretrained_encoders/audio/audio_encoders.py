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
from encoder import add_lora
from encoder import recursive_freeze_unfreeze

from transformers import AutoProcessor, ClapAudioModel

class CLAPWrapper(nn.Module):
    def __init__(self, encoder, config):
        super().__init__()
        self.encoder = encoder
        self.processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused")
        self.config = config
        self.prepate_encoder()
    
    def freeze_model(self): # TODO
        num_layers_to_unfreeze = self.config.num_layers_to_unfreeze

        # Freeze everything
        self.encoder.requires_grad_(False)

        # Unfreeze last num_layers_to_unfreeze layers
        for child_name, child in list(self.encoder.named_modules())[-num_layers_to_unfreeze:]:
            child.requires_grad_(True)

        recursive_freeze_unfreeze(self.encoder, ['Embedding', 'LayerNorm', 'BatchNorm2d', 'Parameter'], freeze=False)

        # TODO If they are using residual with torch multihead attention, lora might not work

        # TODO There is ViT inside of CLAP code???? 

        # TODO SpareEmbedding??
    
    def prepate_encoder(self):
        if self.config.freeze_encoder:
            self.freeze_model()
        if self.config.add_lora:
            add_lora(self.encoder)
        
    def forward(self, x):
        '''
        x: (B, T, C, H, W)
        '''

        b, t, c, h, w = x.shape
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.processor(x, return_tensors="pt")
        embeddings = self.encoder(x)
        embeddings = rearrange(embeddings, "(b t) n e -> b t n e", b=b, t=t)
        return embeddings

def get_audio_encoder(
    name: str, 
    device: Union[torch.device, str] = None, 
    pretrained: bool = False,
    cache_path: str = None
) -> torch.nn.Module:
    """
    Loads audio encoder module
    """
    if name == "clap":
        encoder = ClapAudioModel.from_pretrained("laion/clap-htsat-fused")
    else:
        raise ValueError(f"audio encoder {name} not recognized")
    return encoder
