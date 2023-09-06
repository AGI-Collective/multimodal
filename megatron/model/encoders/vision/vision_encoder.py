import torch
import torch.nn as nn
from urllib.parse import urlparse
import einops
from einops import rearrange
from timm.layers.drop import DropPath

from ..utils import add_lora
from ..utils import recursive_freeze_unfreeze
from .dinov2.models import vision_transformer as vits
from .dinov2 import layers

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
        self.temporal_attn = nn.MultiheadAttention(input_dim, num_heads=input_dim // 64)
        self.norm = nn.LayerNorm(input_dim, eps=1e-12) #[TODO] 
        self.linear = nn.Linear(input_dim, input_dim)
        self.droppath = DropPath(droppath_rate)
        self.scale = nn.parameter.Parameter(torch.zeros([]))

    def forward(self, x: torch.Tensor):
        """
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


def load_pretrained_weights(model, pretrained_weights, checkpoint_key):
    # [TODO] add logger here
    if urlparse(pretrained_weights).scheme:  # If it looks like an URL
        state_dict = torch.hub.load_state_dict_from_url(pretrained_weights, map_location="cpu")
    else:
        state_dict = torch.load(pretrained_weights, map_location="cpu")
    if checkpoint_key is not None and checkpoint_key in state_dict:
        state_dict = state_dict[checkpoint_key]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    return model

def get_vision_encoder(
    args,
    name,
    pretrained: bool = False,
) -> torch.nn.Module:
    """
    Loads vision encoder module, supporting dinov2.
    """
    if "vit" in name:
        vit_kwargs = dict(
            img_size=args.global_crops_size,
            patch_size=args.patch_size,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            # [TODO] rethink about the following args when training
            # init_values=args.layerscale,
            # qkv_bias=args.qkv_bias,
            # proj_bias=args.proj_bias,
            # ffn_bias=args.ffn_bias,
        )
        model = vits.__dict__[name](**vit_kwargs)
        if pretrained:
            model = load_pretrained_weights(model, args.pretrained_weights, "teacher")
        # encoder = DinoWrapper(model)
        encoder = model
    else:
        raise ValueError(f"vision encoder {name} not recognized")
    return encoder