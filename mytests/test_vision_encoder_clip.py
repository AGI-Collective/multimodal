import torch
from PIL import Image
from megatron.model.encoders.vision.vision_encoder import get_vision_encoder
from open_clip import create_model_and_transforms, get_tokenizer
import numpy as np 


from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torchvision

class Args:
    def __init__(self):
        self.name = "openclip"
        self.arch = "ViT-B-32"
        self.cache_dir = '/p/scratch/ccstdl/gupta6/openclip/vitb32.pt'
        self.pretrained_data = "laion2b_s34b_b79k"
        self.freeze_encoder = False
        self.add_lora = False
        self.pretrained = True
        self.perceiver_seq_length = 64
        self.embed_dropout_prob = 0.1
        self.use_embed_layernorm = True
        self.num_layers_to_unfreeze = 2


def clip_transform_match():

    image_path = "/p/project/ccstdl/gupta6/multimodal/mytests/dog.jpg"
    device = "cuda"
    model, preprocess_train, preprocess_val = create_model_and_transforms("ViT-B-32", pretrained = "laion2b_s34b_b79k", cache_dir = '/p/scratch/ccstdl/gupta6/openclip/vitb32.pt')
    tokenizer = get_tokenizer('ViT-B-32')

    vision = model.visual
    vision = vision.to(device, dtype = torch.bfloat16)

    vision.output_tokens = True
    image = Image.open(image_path)
    image = torchvision.transforms.functional.resize(image, [224, 224], interpolation=InterpolationMode.BICUBIC)
    image = preprocess_val(image).unsqueeze(0).to(device).bfloat16().contiguous()

    pooled, tokens = vision(image)
    args = Args()
    
    
    model = get_vision_encoder(args, args.name, True).to(device, dtype = torch.bfloat16)

    image = Image.open(image_path)
    image = torchvision.transforms.functional.resize(image, [224, 224], interpolation=InterpolationMode.BICUBIC)
    image = torch.tensor(np.array(image))
    

    image = image.permute(2, 0, 1)
    image = image.unsqueeze(0).unsqueeze(0).to(device)
    output = model(image)
    
    assert torch.allclose(tokens, output, atol=1e-3), "Tokens and output do not match"

def clip_image_text_test():


    image_path = "/p/project/ccstdl/gupta6/multimodal/mytests/dog.jpg"
    caption = ["a diagram", "a dog", "a cat"]

    device = "cuda"
    model, preprocess_train, preprocess_val = create_model_and_transforms("ViT-B-32", pretrained = "laion2b_s34b_b79k", cache_dir = '/p/scratch/ccstdl/gupta6/openclip/vitb32.pt')
    tokenizer = get_tokenizer('ViT-B-32')

    model = model.to(device)
    image = Image.open(image_path)
    image = torchvision.transforms.functional.resize(image, [224, 224], interpolation=InterpolationMode.BICUBIC)
    image = preprocess_val(image).unsqueeze(0).to(device)
    
    text = tokenizer(["a diagram", "a dog", "a cat"]).to(device)

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)


    print("Label probs:", text_probs)  # tensor([[4.0692e-05, 9.9989e-01, 6.5026e-05]], device='cuda:0')

def main():
    clip_image_text_test()
    clip_transform_match()

if __name__ == "__main__":
    main()