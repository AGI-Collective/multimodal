import torch
# from .vision_encoder import get_vision_encoder
# from dinov2.models import vision_transformer as vits
# from megatron.model.encoders.vision.vision_encoder import get_vision_encoder
import torch
from PIL import Image
import open_clip
    
class Args:
    def __init__(self):
        
        pass

def test_eva_clip_image_transformer_shapes():
    
    device = "cuda"
    
    model, preprocess = open_clip.create_model_from_pretrained('ViT-B-16','hf-hub:timm/vit_base_patch16_224.augreg_in21k')
    
    for transform in preprocess.transforms:
        print(transform)
    # preprocess.transforms = preprocess.transforms[3:]
    # print(preprocess.transforms)
    
    visual = model.visual
    visual.trunk.output_tokens = True
    
    visual = visual.to(device)
    del model
    
    image_path = "dog.jpg"
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    print(image.shape)

    pooled = visual(image)
    
    print(pooled.shape)
    # print(tokens.shape)
    
    

def main():
    

    test_eva_clip_image_transformer_shapes()
    # test_eva_clip_image_transformer()
    # test_dino_image_frozen_transformer()
    # test_dino_image_frozen_lora_transformer()
    # test_dino_video_transformer_basic()
    
if __name__ == "__main__":
    
    
    main()
