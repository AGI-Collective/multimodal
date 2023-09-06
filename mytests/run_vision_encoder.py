import os
import sys
sys.path.append('.')
import torch
from megatron.model.encoders.vision.vision_encoder import get_vision_encoder

# vision setting
class Args:
    def __init__(self):
        self.patch_size = 14
        self.drop_path_rate = 0.4
        self.ffn_layer = "swiglufused"
        self.block_chunks = 4
        self.global_crops_size = 518
        self.pretrained_weights=r'/home/lfsm/pretrained_weights/dinov2_vit_l14/dinov2_vitl14_pretrain.pth'
        self.num_layers_to_unfreeze=1
        self.freeze_encoder=False
        self.add_lora=False

# Create an instance of the Args class
args = Args()
model=get_vision_encoder(args,"vit_large",True)
images = torch.ones(5,3,224,224)
# images = torch.ones(5,3,224,224)
output = model(images)
print(model)
print(output.shape)