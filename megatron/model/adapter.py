import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import Literal
import math

class Adapter(nn.Module):
    def __init__(
        self,
        dim: int,
        downsample_factor: int = 4,
        activation: nn.Module = nn.ReLU,
        add_layernorm: bool = False,
        is_lora: bool = False,
        rank : int = 8,
    ):
        super().__init__()
        if add_layernorm:
            self.layer_norm = nn.LayerNorm(dim)
        else:
            self.layer_norm = False
        
        if not is_lora:
            self.input_lora = nn.Linear(dim, dim // downsample_factor)
            self.activation = activation()
            self.output_lora = nn.Linear(dim // downsample_factor, dim)
        else:
            self.input_lora = nn.Linear(dim, rank)
            self.activation = None
            self.output_lora = nn.Linear(rank, dim)
        
        
        self.init_weights()

    def init_weights(self, std=1e-3):
        
        torch.nn.init.normal_(self.input_lora.weight, std=std)
        torch.nn.init.normal_(self.input_lora.bias, std=std)
        self.input_lora.weight.data = torch.clamp(self.input_lora.weight.data, min=-2 * std, max=2 * std)
        self.input_lora.bias.data = torch.clamp(self.input_lora.bias.data, min=-2 * std, max=2 * std)
        
        #We are not setting output to zero - usually you would with adapters, but we are doing domain adaptation so it doesn't matter.
        torch.nn.init.normal_(self.output_lora.weight, std=std)
        torch.nn.init.normal_(self.output_lora.bias, std=std)
        self.output_lora.weight.data = torch.clamp(self.output_lora.weight.data, min=-2 * std, max=2 * std)
        self.output_lora.bias.data = torch.clamp(self.output_lora.bias.data, min=-2 * std, max=2 * std)
        
        if self.layer_norm != False:
            self.layer_norm.bias.data.zero_()
            self.layer_norm.weight.data.fill_(1.0)

    def adapt(self, x):
        
        if self.layer_norm != False:
            x = self.layer_norm(x)
        
        x = self.input_lora(x)
        
        if self.activation != None:
            x = self.activation(x)
            
        x = self.output_lora(x)
        return x
        
    def forward(self, x: TensorType["b", "s", "d"]) -> TensorType["b", "s", "d"]:
        return self.adapt(x) + x


class ParallelAdapter(Adapter):
    def __init__(
        self,
        module: nn.Module,
        dim: int,
        downsample_factor: int = 4,
        scaled: bool = False,
        add_layernorm: bool = False,
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__(
            dim, downsample_factor, add_layernorm=add_layernorm, activation=activation
        )
        self.module = module

        if scaled:
            # init scaling param
            self.adapter_scale = nn.Parameter(torch.ones(1))
        else:
            self.adapter_scale = 1

    def forward(self, x: TensorType["b", "s", "d"], **module_kwargs):
        y = self.module(x, **module_kwargs)
        z = self.adapt(x)
        return y + (z * self.adapter_scale)


class ParallelAdapterWrapper(ParallelAdapter):
    # used to add an adapter to the attention block

    def __init__(
        self,
        module: nn.Module,
        dim: int,
        downsample_factor: int = 4,
        scaled: bool = False,
        add_layernorm: bool = False,
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__(
            module, dim, downsample_factor, scaled, add_layernorm, activation
        )

    def forward(self, x: TensorType["b", "s", "d"], *attn_args, **attn_kwargs):
        attn_outputs = self.module(x, *attn_args, **attn_kwargs)
        attn_output, outputs = (
            attn_outputs[0],
            attn_outputs[1:],
        )  # output_attn: a, present, (attentions)
        hidden_states = attn_output + (self.adapt(x) * self.adapter_scale)
        return (hidden_states,) + outputs


class AdapterWrapper(Adapter):
    # used to add an adapter to the attention block

    def __init__(
        self,
        attn_block: nn.Module,
        dim: int,
        downsample_factor: int = 4,
        activation: nn.Module = nn.ReLU,
        add_layernorm: bool = False,
    ):
        super().__init__(dim, downsample_factor, activation, add_layernorm)
        self.attn_block = attn_block

    def forward(self, x: TensorType["b", "s", "d"], *attn_args, **attn_kwargs):
        attn_outputs = self.attn_block(x, *attn_args, **attn_kwargs)
        
        attn_output, outputs = (
            attn_outputs[0],
            attn_outputs[1:],
        )  # outputs: output, bias
        hidden_states = self.adapt(attn_output) + attn_output
        return (hidden_states,) + outputs

class ParallelLinearPEFT(torch.nn.Module):
    
    def __init__(self, 
                 par_lin,
                 rank = 8,
                 is_lora = False,#Assume it's not a lora. If not, rank = downsampling
                 alpha = 1.0):
       
        super().__init__()
        self.in_features = par_lin.input_size
        self.out_features = par_lin.output_size
        
        self.par_lin = par_lin
        self.rank = rank
        
        self.downsample_adapter_lora = torch.nn.Parameter(torch.zeros((rank, self.in_features)))
        self.upsample_adapter_lora = torch.nn.Parameter(torch.zeros((self.out_features, rank)))
        self.scaling = alpha / self.rank
        
        # initialize A the same way as the default for nn.Linear and B to zero
        #This doesn't matter for domain adaptation
        #torch.nn.init.kaiming_uniform_(self.downsample_adapter_lora, a=math.sqrt(5))
        #torch.nn.init.zeros_(self.upsample_adapter_lora)
    
    def forward(self, x):
        
        result, bias = self.par_lin.forward(x)
        
        result = result + x @ self.downsample_adapter_lora.T @ self.upsample_adapter_lora.T * self.scaling
        
        return result, bias

def add_adapters(
        neox_args,
        model,
        downsample_factor: int = 4,
        # adapter_type: Literal["normal", "parallel", "scaled_parallel"] = "normal",
        location: Literal["mlp", "attention"] = "mlp",
        ff_attr: str = "mlp",
        attn_attr: str = "attention",
        **adapter_kwargs,    
):
    for names, module in model.named_modules():
        if 'image_prefix' in names:
          continue # no adapter for image_prefix transformers
        temp_names = [name for name,module in module.named_modules()]
        #Adapters aren't placed at the leaf node level, but one above it
        if False:#location in temp_names and location==ff_attr:
            mlp = getattr(module,ff_attr)
            adapter_layer = AdapterWrapper(
                        attn_block=mlp,
                        dim=neox_args.hidden_size,
                        downsample_factor=downsample_factor,
                        **adapter_kwargs
                        )
            setattr(module,ff_attr,adapter_layer)   
        elif False:#location in temp_names and location==attn_attr:
            attn = getattr(module,attn_attr)
            adapter_layer = AdapterWrapper(
                        attn_block=attn,
                        dim=neox_args.hidden_size,
                        downsample_factor=downsample_factor,
                        **adapter_kwargs
                        )
            setattr(module,attn_attr,adapter_layer)
    recursive_replace(model)
    print("model after recur", model)
    print("done", flush = True)
    return model

def recursive_replace(model):

    for child_name, child in model.named_children():
        print("recusrive child name", child_name)
        if 'image_prefix' in child_name:
            return
        if isinstance(child, nn.Linear):#I don't believe there are any regular leafs
            #Basically we need to be certain this doesn't trigger on the things we added
            if "adapter_lora" in child_name:
                return
            print("found linear", child_name)
            peft = ParallelLinearPEFT(child, is_lora = True)
            setattr(model, child_name, peft)
            
        if "ColumnParallel" in str(type(child)) or "RowParallel" in str(type(child)):
            print("found row/col")
            regular_parallel_linear = ["dense", "final_linear", "dense_4h_to_h","dense_h_to_4h"]
            if "query_key_value" in child_name:
                #For now we try not giving a shit
                peft = ParallelLinearPEFT(child, is_lora = True)
                setattr(model, child_name, peft)
                pass#Create a mergedlinearadapter here..
            elif child_name in regular_parallel_linear:
                peft = ParallelLinearPEFT(child, is_lora = True)
                setattr(model, child_name, peft)
            else:
                print("parallel fail")
                print(child_name)
        elif isinstance(child, Adapter):
            return
        else:
            #print("recursive")
            #print(child_name)
            #print(type(child))
            recursive_replace(child)

