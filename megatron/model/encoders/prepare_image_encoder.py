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