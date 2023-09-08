import torch
# from .vision_encoder import get_vision_encoder
from dinov2.models import vision_transformer as vits

def test_dino_video_transformer_basic():
    # Create a batch of 3 videos, each with 16 frames and 3 channels
    batch_size = 3
    num_frames = 16
    num_channels = 3
    video = torch.randn(batch_size, num_frames, num_channels, 224, 224)

    # Initialize the model
    model = vits.vit_small(#Could also be vit_base, vit_large, vit_giant
            patch_size=14,
            img_size=526,
            init_values=1.0,
            block_chunks=0, video_mode=True)
    
    # Pass the video through the model
    output = model(video)
    print(output.shape)
    # Check that the output has the expected shape
    # assert output.shape == (batch_size, model.num_classes)

def test_dino_video_transformer_single_frame():
    # Create a batch of 3 videos, each with 16 frames and 3 channels
    batch_size = 3
    num_frames = 1
    num_channels = 3
    video = torch.randn(batch_size, num_frames, num_channels, 224, 224)

    # Initialize the model
    model = vits.vit_small(#Could also be vit_base, vit_large, vit_giant
            patch_size=14,
            img_size=526,
            init_values=1.0,
            block_chunks=0, video_mode=True)
    
    # Pass the video through the model
    output = model(video)
    print(output.shape)
    # Check that the output has the expected shape
    # assert output.shape == (batch_size, model.num_classes)

def __main__():
    print("I am here")
    test_dino_video_transformer_basic()
    print("Done testing")
    test_dino_video_transformer_single_frame()
