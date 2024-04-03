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
    #
    # model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:timm/eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k')
    import open_clip

    model, preprocess = open_clip.create_model_from_pretrained('hf-hub:timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k')
    tokenizer = open_clip.get_tokenizer('hf-hub:timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k')
    # print(type(preprocess))
    # print(preprocess.transforms)
    
    
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
    
    

def test_eva_clip_image_transformer():
    

    image_path = "dog.jpg"
    caption = ["a diagram", "a dog", "a cat"]
    
    device = "cuda"
    model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:timm/eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k')
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
    

    test_eva_clip_image_transformer_shapes()
    # test_eva_clip_image_transformer()
    # test_dino_image_frozen_transformer()
    # test_dino_image_frozen_lora_transformer()
    # test_dino_video_transformer_basic()
    
if __name__ == "__main__":
    
    
    main()
