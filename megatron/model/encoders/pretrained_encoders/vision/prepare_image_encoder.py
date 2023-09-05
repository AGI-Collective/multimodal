import os
import sys
sys.path.append('.')
from megatron.model.encoders.image_encoders import ImageEncoder
from megatron.neox_arguments import NeoXArgs


neox_args = NeoXArgs.from_ymls(['./configs/summit-70m-openclipH.yml', './configs/summit_setup.yml'])
image_prefix = ImageEncoder(
    config = neox_args,
    out_dim=neox_args.hidden_size,
)
print(f"Downloaded pretrain weight of {neox_args.encoder_name} to {neox_args.load} !")



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
