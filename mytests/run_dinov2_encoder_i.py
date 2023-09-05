import os
import sys
sys.path.append('.')
import torch
from megatron.model.encoders.pretrained_encoders.vision.dinov2.models import get_pretrained_model_from_cfg

# neox_args = NeoXArgs.from_ymls(['./configs/summit-70m-openclipH.yml', './configs/summit_setup.yml'])
# dinov2 setting
class Args:
    def __init__(self):
        self.arch = "vit_large"
        self.patch_size = 14
        self.drop_path_rate = 0.4
        self.ffn_layer = "swiglufused"
        self.block_chunks = 4
        self.global_crops_size = 518
        self.pretrained_weights=r'/home/lfsm/pretrained_weights/dinov2_vit_l14/dinov2_vitl14_pretrain.pth'

# Create an instance of the Args class
args = Args()
model=get_pretrained_model_from_cfg(args)
images = torch.ones(5,3,224,224)
print(model)