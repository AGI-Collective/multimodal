import torch
# from .vision_encoder import get_vision_encoder
# from dinov2.models import vision_transformer as vits
from megatron.model.encoders.vision.vision_encoder import get_vision_encoder

class Args:
    def __init__(self):
        self.arch = "vit_base"
        self.patch_size = 14
        self.drop_path_rate = 0.4
        self.ffn_layer = "swiglufused"
        self.block_chunks = 4
        self.img_size = 518
        self.pretrained_weights=r'/p/scratch/ccstdl/gupta6/dinov2/dinobase.pt'
        self.freeze_encoder = False
        self.add_lora = False

def test_dino_image_transformer():
    args = Args()
    batch_size = 3
    num_channels = 3
    num_frames = 1
    image = torch.randn(batch_size,num_frames, num_channels, 224, 224)

    # Initialize the model
    model = get_vision_encoder(args,args.arch,True)
    
    # Pass the video through the model
    output = model(image)
    print(output.shape)

def test_dino_image_frozen_transformer():
    args = Args()
    args.freeze_encoder = True
    args.num_layers_to_unfreeze = 2
    batch_size = 3
    num_channels = 3
    num_frames = 1
    image = torch.randn(batch_size,num_frames, num_channels, 224, 224)

    # Initialize the model
    model = get_vision_encoder(args,args.arch,True)
    
    # Pass the video through the model
    output = model(image)
    print(output.shape)

def test_dino_image_frozen_lora_transformer():
    args = Args()
    args.freeze_encoder = True
    args.add_lora = True
    args.num_layers_to_unfreeze = 2
    batch_size = 3
    num_channels = 3
    num_frames = 1
    image = torch.randn(batch_size,num_frames, num_channels, 224, 224)

    # Initialize the model
    model = get_vision_encoder(args,args.arch,True)
    
    # Pass the video through the model
    output = model(image)
    print(output.shape)


def test_dino_video_transformer_basic():
    args = Args()

    # Create a batch of 3 videos, each with 16 frames and 3 channels
    batch_size = 3
    num_frames = 16
    num_channels = 3
    video = torch.randn(batch_size, num_frames, num_channels, 224, 224)

    model = get_vision_encoder(args,"vit_vision_base",True)
    
    # Pass the video through the model
    output = model(video)
    print(output.shape)

def main():
    test_dino_image_transformer()
    test_dino_image_frozen_transformer()
    test_dino_image_frozen_lora_transformer()
    # test_dino_video_transformer_basic()
    
if __name__ == "__main__":
    main()
