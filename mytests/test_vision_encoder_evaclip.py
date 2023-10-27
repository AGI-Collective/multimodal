import torch
# from .vision_encoder import get_vision_encoder
# from dinov2.models import vision_transformer as vits
# from megatron.model.encoders.vision.vision_encoder import get_vision_encoder
import torch
from PIL import Image
import open_clip
from conda.common._logic import TRUE
    
class Args:
    def __init__(self):
        
        pass

def notest_eva_clip_image_transformer():
    
    device = "cuda"
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:timm/eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k')
    tokenizer = open_clip.get_tokenizer('hf-hub:timm/eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k')
    
    print("Loaded model")
    
    vision = model.visual
    vision = vision.to(device)
    
    vision.output_tokens = True
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    
    pooled, tokens = vision(image)
    
    print(pooled.shape)
    print(tokens.shape)

def test_eva_clip_image_transformer():
    

    image_path = "dog.jpg"
    caption = ["a diagram", "a dog", "a cat"]
    
    device = "cuda"
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:timm/eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k')
    tokenizer = open_clip.get_tokenizer('hf-hub:timm/eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k')
    
    print("Loaded model")
    model = model.to(device)
    
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = tokenizer(["a diagram", "a dog", "a cat"]).to(device)
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)


    print("Label probs:", text_probs)  # prints: [[0.8275, 0.1372, 0.0352]]
    
def main():
    
    
    test_eva_clip_image_transformer()
    # test_dino_image_frozen_transformer()
    # test_dino_image_frozen_lora_transformer()
    # test_dino_video_transformer_basic()
    
if __name__ == "__main__":
    main()
