import os
import sys
sys.path.append('.')
from megatron.neox_arguments import NeoXArgs
import torch
import argparse

def download_dino(size, location):
    
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

# Main function
if __name__ == "__main__":
    # Argument is size and location
    parser = argparse.ArgumentParser(description='Download DINO')
    parser.add_argument('--size', type=str, default="base", help='size of DINO')
    parser.add_argument('--location', type=str, default="./", help='location to save DINO')
    args = parser.parse_args()
    download_dino(args.size, args.location)