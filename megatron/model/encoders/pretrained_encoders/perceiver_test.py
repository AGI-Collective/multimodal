import open_clip
import torch
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import einsum, nn



from perceiver import PerceiverResampler


vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai")

vision_encoder = vision_encoder.visual
vis_dim=open_clip.get_model_config("ViT-L-14")["vision_cfg"]["width"]
perceiver = PerceiverResampler(dim=vis_dim)

            
            
batch = 5
num_images = 27
channels = 3
height = 512
width = 512
input_data = torch.randn((batch, num_images, 1, channels, height, width))
# vision_x (torch.Tensor): Vision input
#     shape (B, T_img, F, C, H, W) with F=1



def encode_vision_x(vision_x: torch.Tensor):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """

        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]
        assert F == 1, "Only single frame supported"

        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        with torch.no_grad():
            vision_x = vision_encoder(vision_x)[1]
        
        
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
        vision_x = perceiver(vision_x)

        # for layer in lang_encoder._get_decoder_layers():
        #     layer.condition_vis_x(vision_x)

encode_vision_x(input_data)