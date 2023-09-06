import torch
import torch.nn as nn
from loralib.layers import Linear as LoraLinear
from loralib.layers import MergedLinear as MergedLoraLinear
from loralib.layers import Conv2d as LoraConv2d

def add_lora(self, model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.Linear) and child.requires_grad == False:
            weight = child.weight
            bias = child.bias
            if child_name == 'qkv':
                new = MergedLoraLinear(child.in_features, child.out_features, 
                            r = 128, dtype = torch.float32, enable_lora = [True,True,True])
            else:
                new = LoraLinear(child.in_features, child.out_features, r = 128, dtype = torch.float32)
            new.weight = weight
            new.bias = bias
            setattr(model, child_name, new)
        elif isinstance(child, nn.Conv2d):
            weight = child.weight
            bias = child.bias
            new = LoraConv2d(child.in_channels, child.out_channels, child.kernel_size[0], r = 128, dtype = torch.float32)                     
            new.weight = weight
            new.bias = bias
            new.stride = child.stride
            new.padding = child.padding
            new.dilation = child.dilation
            setattr(model, child_name, new)
        else:
            self.add_lora(child)

def recursive_freeze_unfreeze(self, model, param_types, freeze=True):
    for child_name, child in model.named_children():
        child_class_name = child.__class__.__name__
        if str(child_class_name) in param_types:
            child.requires_grad = not freeze
        else:
            self.recursive_freeze_unfreeze(child, param_types, freeze)
