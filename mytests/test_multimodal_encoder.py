import torch
# from .vision_encoder import get_vision_encoder
# from dinov2.models import vision_transformer as vits
from megatron.model.encoders.vision.vision_encoder import get_vision_encoder
from megatron.model.multimodal_encoder import MultiModalEncoder

class Args:
    def __init__(self):
        self.arch = "vit_base"
        self.modality = "vision"
        self.patch_size = 14
        self.drop_path_rate = 0.4
        self.ffn_layer = "swiglufused"
        self.block_chunks = 4
        self.img_size = 518
        self.pretrained_weights=r'/p/scratch/ccstdl/gupta6/dinov2/dinobase.pt'
        self.freeze_encoder = False
        self.add_lora = False
        self.pretrained = True
        self.encoder_type = "dinov2_base"
        self.embed_dropout_prob = 0.1
        self.use_embed_layernorm = True
        self.perceiver_seq_length = 64
        

def test_dino_image_frozen_transformer():
    args = Args()
    args.freeze_encoder = True
    args.num_layers_to_unfreeze = 2
    batch_size = 4
    num_channels = 3
    num_frames = 1
    timesteps = 2
    image = torch.randn(batch_size, timesteps, num_frames, num_channels, 224, 224)

    # Initialize the model
    model = MultiModalEncoder(args, 2048)
    
    # Pass the video through the model
    output = model(image)
    print(output.shape)


def main():
    test_dino_image_frozen_transformer()

if __name__ == "__main__":
    main()