import sys
import torch
from megatron.neox_arguments import NeoXArgs
from megatron.initialize import initialize_megatron
from megatron.training import get_model,get_optimizer,get_learning_rate_scheduler 

neox_args = NeoXArgs.from_ymls(['/home/lfsm/code/multimodal/configs/mytests/70m-openclipH.yml', '/home/lfsm/code/multimodal/configs/mytests/local_setup.yml'])
neox_args.configure_distributed_args()
neox_args.build_tokenizer()  # tokenizer needs to be build in training in order to set the padding vocab
initialize_megatron(neox_args=neox_args)
model = get_model(neox_args=neox_args)
optimizer, param_groups = get_optimizer(model=model, neox_args=neox_args)
lr_scheduler = get_learning_rate_scheduler(optimizer=optimizer, neox_args=neox_args)

print(param_groups[0])
print(param_groups[1])
print(param_groups[2])
print(param_groups[3])
