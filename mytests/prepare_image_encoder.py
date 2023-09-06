import os
import sys
sys.path.append('.')
from megatron.model.encoders.image_encoders import ImageEncoder
from megatron.neox_arguments import NeoXArgs


def download_dino(size = "base", location):
    import torch
    BACKBONE_SIZE = size # in ("small", "base", "large" or "giant")
    backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
    backbone_arch = backbone_archs[BACKBONE_SIZE]
    backbone_name = f"dinov2_{backbone_arch}"
    
    backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    torch.save(backbone_model.state_dict(), str(location) + "./dino"+str(size)+".pt")
