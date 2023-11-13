import torch
import torch.nn as nn
from urllib.parse import urlparse
import einops
from einops import rearrange

from ..utils import add_lora
from ..utils import recursive_freeze_unfreeze
from .dinov2.models import vision_transformer as vits
from .dinov2 import layers

from open_clip import create_model_and_transforms

from .transforms import make_classification_eval_transform, OPEN_CLIP_MEAN, OPEN_CLIP_STD
from abc import ABC, abstractmethod

class VisionWrapper(nn.Module, ABC):
    def __init__(self, encoder, config):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.unfreeze_params = ['LayerNorm']
        self.prepare_encoder()
        self.transform = make_classification_eval_transform()
        
    def freeze_model(self):
        num_layers_to_unfreeze = self.config.num_layers_to_unfreeze
        # Freeze everything
        self.encoder.requires_grad_(False)
        if num_layers_to_unfreeze > 0:
            # Unfreeze last num_layers_to_unfreeze layers
            for child_name, child in list(self.encoder.named_modules())[-num_layers_to_unfreeze:]:
                child.requires_grad_(True)
        # Unfreeze LayerNorm
        recursive_freeze_unfreeze(self.encoder, param_types=self.unfreeze_params, freeze=False)
        # What about cls token? TODO
    
    def prepare_encoder(self):
        if self.config.freeze_encoder:
            self.freeze_model()
        if self.config.add_lora:
            add_lora(self.encoder)
    
    @abstractmethod
    def get_embeddings(self, *args, **kwargs):
        pass

    def forward(self, x):
        '''
        x: (b, t, c, h, w)
        x.shape:
            b=batch size, 
            t=number of frames in each image/video, 
            c=number of channels, h=height, w=width
        '''
        b, t, c, h, w = x.shape 
        combined_batch = rearrange(x, "b t c h w -> (b t) c h w")
        preprocessed_vision = self.transform(combined_batch).bfloat16().contiguous() #.half()
        x = rearrange(preprocessed_vision, "(b t) c h w -> b t c h w", b=b, t=t)
        if "vision" in self.config.arch:
            embeddings = self.get_embeddings(x) # B, N_E, E
        else:
            x = rearrange(x, "b t c h w -> (b t) c h w")
            embeddings = self.get_embeddings(x) # B*T, N_E, E
            embeddings = rearrange(embeddings, "(b t) n_e e -> b (t n_e) e", b=b, t=t)
        return embeddings

class DinoWrapper(VisionWrapper):

    def __init__(self, encoder, config):
        super().__init__(encoder, config)
    
    def get_embeddings(self, x):
        return self.encoder(x)

class ClipWrapper(VisionWrapper):
    
    def __init__(self, encoder, config):
        super().__init__(encoder, config)
        self.transform = make_classification_eval_transform(resize_size=224, mean=OPEN_CLIP_MEAN, std=OPEN_CLIP_STD)

    def get_embeddings(self, x):
        cls_token, all_embeddings = self.encoder(x)
        return all_embeddings

def load_pretrained_dino_weights(model, pretrained_weights, checkpoint_key):
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
    if "dino" in name:
        vit_kwargs = dict(
            img_size=args.img_size,
            patch_size=args.patch_size,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            # [TODO] rethink about the following args when training
            # init_values=args.layerscale,
            # qkv_bias=args.qkv_bias,
            # proj_bias=args.proj_bias,
            # ffn_bias=args.ffn_bias,
        )
        model = vits.__dict__[args.arch](**vit_kwargs)
        if pretrained:
            model = load_pretrained_dino_weights(model, args.pretrained_weights, "teacher")
        encoder = DinoWrapper(model,args)
    elif "openclip" in name:
        model, _, _ = create_model_and_transforms(args.arch, pretrained=args.pretrained_data, cache_dir=args.cache_dir)
        #Todo unsure if preprocess is right - it was two things before...
        model = model.visual
        model.output_tokens = True
        encoder = ClipWrapper(model, args)    
    else:
        raise ValueError(f"vision encoder {name} not recognized")
    return encoder