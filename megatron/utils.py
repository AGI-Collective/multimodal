# Copyright (c) 2021, EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""General utilities."""
import os
import sys
import re
import time
import socket
from typing import Dict, List

import requests

try:
    import wandb
except ModuleNotFoundError:
    pass

import torch

from deepspeed.launcher.runner import fetch_hostfile, parse_inclusion_exclusion

from megatron import print_rank_0
from megatron import mpu

from collections import deque

MODALITY_DICT = {
    "text":0, 
    "vision":1,
    "audio":2
}

def reduce_losses(losses):
    """Reduce a tensor of losses across all GPUs."""
    reduced_losses = torch.cat([loss.clone().detach().view(1) for loss in losses])
    torch.distributed.all_reduce(reduced_losses)
    reduced_losses = reduced_losses / torch.distributed.get_world_size()
    return reduced_losses


def report_memory(name):
    """Simple GPU memory report."""
    mega_bytes = 1024.0 * 1024.0
    string = name + " memory (MB)"
    string += " | allocated: {}".format(torch.cuda.memory_allocated() / mega_bytes)
    string += " | max allocated: {}".format(
        torch.cuda.max_memory_allocated() / mega_bytes
    )
    string += " | reserved: {}".format(torch.cuda.memory_reserved() / mega_bytes)
    string += " | max reserved: {}".format(
        torch.cuda.max_memory_reserved() / mega_bytes
    )
    print_rank_0(string)


def get_attn_mask(seq_length, device):
    """
    Get triangular attention mask for a given sequence length / device.
    """
    # lower triangular attention mask
    mask = torch.tril(torch.ones((1, seq_length, seq_length), device=device)).view(
        1, 1, seq_length, seq_length
    )

    # convert to binary
    return mask < 0.5

def get_ltor_masks_and_position_ids(
    data,
    eod_token,
    eod_mask_loss=False,
):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    attention_mask = get_attn_mask(
        seq_length=seq_length,
        device=data.device,
    )

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)

    return attention_mask, loss_mask, position_ids


def get_sequence_ids(tokens, eos_token_id, bos_token_id):
    if (eos_token_id is None) and (bos_token_id is None):
        raise ValueError(
            'Must supply a value for either eos_token_id or bos_token_id, but got None for both.'
        )
    if (eos_token_id is not None) and (bos_token_id is not None):
        raise ValueError(
            'Cannot use *both* EOS and BOS tokens for detecting sequence boundaries. ' +\
            'Please supply `eos_token_id` if sequences end with an EOS token, or use ' +\
            '`bos_token_id` if sequences start with a BOS token.'
        )
    
    split_token_id = eos_token_id
    bos_mode = False
    if eos_token_id is None:
        split_token_id = bos_token_id
        bos_mode = True

    is_separator = torch.eq(tokens,
                            split_token_id)  # type: ignore
    cumulative_sep = torch.cumsum(is_separator,
                                    dim=1).to(tokens.dtype)
    # If separator token is bos, we're already done
    if bos_mode:
        return cumulative_sep

    # If separator token is eos, right shift 1 space
    left_zeros = cumulative_sep.new_zeros((cumulative_sep.shape[0], 1))
    return torch.cat([left_zeros, cumulative_sep[:, :-1]], dim=1)
    
def get_shifted_multimodal_position_ids(input_info, position_pad_id=-1):
    
    text_positions = input_info["text"]["positions"]
    vision_positions = input_info["vision"]["positions"]
    # audio_positions = input_info["audio"]["positions"]

    vision_seq_length = input_info["vision"]["seq_length"]
    # audio_seq_length = input_info["audio"]["seq_length"]

    T_B, T_L = text_positions.size()
    V_B, V_L = vision_positions.size()
    # A_B, A_L = audio_positions.size()
    
    # Concatenate all positions
    # all_positions = torch.cat((text_positions, vision_positions, audio_positions), dim=1)
    all_positions = torch.cat((text_positions, vision_positions), dim=1)
    
    # Replace all -1s with increasing values after max value when concatenated
    mask = all_positions == position_pad_id
    max_vals, _ = all_positions.max(dim=-1)
    max_vals_extended = max_vals.view(-1,1).expand(-1, mask.shape[1])
    cumulative_counts = mask.cumsum(dim=-1)
    replacements = max_vals_extended + cumulative_counts
    all_positions[mask] = replacements[mask]

    # Replace all vision positoins with new vision positions
    vision_positions = all_positions[:, T_L:T_L+V_L]

    # Calculate total shifts
    img_shifts_total = (vision_positions.unsqueeze(1) < all_positions.unsqueeze(-1)).sum(dim=-1) * (vision_seq_length - 1)
    # aud_shifts_total = (audio_positions.unsqueeze(1) < all_positions.unsqueeze(-1)).sum(dim=-1) * (audio_seq_length - 1)
    
    total_shifts = img_shifts_total # TODO + aud_shifts_total

    # Add total shifts to all positions
    all_positions_shifted = all_positions + total_shifts

    # Split back and update vision_positions and audio_positions
    shifted_text_positions = all_positions_shifted[:, :T_L]
    shifted_vision_positions = (all_positions_shifted[:, T_L:T_L+V_L]).repeat_interleave(vision_seq_length, dim=1) + torch.arange(vision_seq_length).repeat(V_B, V_L).type_as(vision_positions)
    # shited_audio_positions = (all_positions_shifted[:, -A_L:]).repeat_interleave(audio_seq_length, dim=1) + torch.arange(audio_seq_length).repeat(A_B, A_L).type_as(audio_positions)
    shited_audio_positions = None
    return shifted_text_positions, shifted_vision_positions, shited_audio_positions

def get_proxy_tokens(position_ids, seq_length, pad_id, position_pad_id=-1, start_ind=100):
    multimodal_mask = position_ids != position_pad_id

    # All vision tokens are given a negative index
    proxy_tokens = -1*torch.arange(start_ind, start_ind + multimodal_mask.shape[1]).expand_as(multimodal_mask).to(multimodal_mask.device)
    proxy_tokens = torch.repeat_interleave(proxy_tokens, seq_length, dim=1)
    multimodal_mask = torch.repeat_interleave(multimodal_mask, seq_length, dim=1)
    proxy_masked_tokens = proxy_tokens*multimodal_mask
    proxy_masked_tokens[proxy_masked_tokens == 0] = pad_id # This is any random text token. This cannot be equal to eos, or eod: TODO, can this be padid?
    return proxy_masked_tokens

def get_multimodal_mask(interleaved_tokens, text_pad_id):
    # Find all the multimodal models (negative indices) and ignore pad id
    interleaved_mask = (interleaved_tokens < 0) * (interleaved_tokens != text_pad_id) # pad id is not generally needed since its >= 0
    interleaved_multimodal_tokens = interleaved_tokens*interleaved_mask

    interleaved_multimodal_tokens_3d_1 = interleaved_multimodal_tokens.unsqueeze(-1)
    interleaved_multimodal_tokens_3d_2 = interleaved_multimodal_tokens.unsqueeze(1)

    multimodal_mask = torch.eq(interleaved_multimodal_tokens_3d_1, interleaved_multimodal_tokens_3d_2) & (interleaved_multimodal_tokens_3d_1 < 0)
    return multimodal_mask

def get_multimodal_attn_mask(
        text_tokens,
        vision_positions,
        audio_positions,
        vision_seq_length,
        input_seq_length,
        shifted_multimodal_position_ids, 
        eos_token_id,
        bos_token_id,
        position_pad_token_id,
        text_pad_token_id,
        concat_data,
        attn_uses_sequence_id,
        device
    ):
    """
    Get multimodal attention mask
    """
    batch_size = text_tokens.shape[0]
    # lower triangular attention mask across all tokens
    mask = torch.tril(torch.ones((1, input_seq_length, input_seq_length), device=device)).expand((batch_size, -1, -1)).clone()

    # Form vision proxy tokens using shifted multimodal position ids. Use text_pad_token_id as pad_id
    proxy_vision_tokens = get_proxy_tokens(vision_positions, vision_seq_length, position_pad_id=position_pad_token_id, pad_id=text_pad_token_id, start_ind=100)
    # Do the same process for Audio #TODO

    # Concatenate vision proxy tokens with text tokens 
    concat_tokens = torch.cat((text_tokens, proxy_vision_tokens), dim=1)

    # Rearrrange tokens in interleaved format using shifted multimodal position ids
    interleaved_tokens = torch.zeros_like(concat_tokens, dtype=concat_tokens.dtype, device=concat_tokens.device)
    interleaved_tokens = interleaved_tokens.scatter_(1, shifted_multimodal_position_ids, concat_tokens)
    
    # assert that all tokens after input_seq_length are -1s (padding)
    assert torch.all(interleaved_tokens[:, input_seq_length:] == text_pad_token_id)
    
    interleaved_tokens = interleaved_tokens[:, :input_seq_length]

    # No masking across vision tokens 
    vision_mask = get_multimodal_mask(interleaved_tokens, text_pad_id=text_pad_token_id)
    mask = mask+vision_mask
    mask = torch.clip(mask, 0, 1)

    # if attn_uses_sequence_id, then mask across sequence ids to prevent cross sequence attention
    if concat_data:
        sequence_ids = get_sequence_ids(interleaved_tokens, eos_token_id, bos_token_id)
        if attn_uses_sequence_id:
            sequence_ids_3d_1 = sequence_ids.unsqueeze(-1)
            sequence_ids_3d_2 = sequence_ids.unsqueeze(1)
            matching_sequence_ids = torch.eq(sequence_ids_3d_1, sequence_ids_3d_2)
            mask = mask * matching_sequence_ids
    else:
        sequence_ids = None
    # convert to binary
    mask = mask.view(
        batch_size, 1, input_seq_length, input_seq_length
    )
    return mask < 0.5, sequence_ids


def get_multimodal_ltor_masks_and_position_ids(
    input_info,
    input_seq_length,
    eod_token,
    bos_token,
    pad_token,
    position_pad_token_id,
    vision_input_start_token,
    vision_input_end_token,
    vision_gen_start_token,
    concat_data=True,
    attn_uses_sequence_id=False,
):
    """Build masks and position id for left to right model."""

    # Extract batch size and label length
    batch_size = input_info["text"]["input"].shape[0]
    
    shifted_text_positions, shifted_vision_positions, shited_audio_positions = get_shifted_multimodal_position_ids(input_info, position_pad_id=-1)
    
    # Concatenate all of them
    # Include Audio
    # shifted_multimodal_position_ids = torch.cat((shifted_text_positions, shifted_vision_positions, shited_audio_positions), dim=1)
    shifted_multimodal_position_ids = torch.cat((shifted_text_positions, shifted_vision_positions), dim=1)
    
    # Attention mask (lower triangular). # TODO: INCLUDE Audio in this
    attention_mask, sequence_ids = get_multimodal_attn_mask(
        text_tokens=input_info["text"]["input"],
        vision_positions=input_info["vision"]["positions"],
        audio_positions=None,
        vision_seq_length=input_info["vision"]["seq_length"],
        input_seq_length=input_seq_length,
        shifted_multimodal_position_ids=shifted_multimodal_position_ids,
        eos_token_id=eod_token,
        bos_token_id=bos_token,
        position_pad_token_id=position_pad_token_id, 
        text_pad_token_id=pad_token,
        concat_data=concat_data,
        attn_uses_sequence_id=attn_uses_sequence_id,
        device=input_info["text"]["input"].device,
    )

    # Prepare labels 
    vision_labels = torch.repeat_interleave(input_info["vision"]["labels"], input_info["vision"]["seq_length"], dim=1)
    
    # Concatenate vision proxy tokens with text tokens 
    concat_labels = torch.cat((input_info["text"]["labels"], vision_labels), dim=1)

    # Rearrrange tokens in interleaved format using shifted multimodal position ids
    interleaved_labels = torch.zeros_like(concat_labels, dtype=concat_labels.dtype, device=concat_labels.device)
    labels = interleaved_labels.scatter_(1, shifted_multimodal_position_ids, concat_labels)[:,:input_seq_length]

    # Loss mask.
    label_length = labels.shape[1]
    loss_mask = torch.ones((batch_size, label_length), dtype=torch.float, device=labels.device)
    
    if pad_token is not None:
        loss_mask[labels == pad_token] = 0.0
        loss_mask[labels == bos_token] = 0.0
        loss_mask[labels == eod_token] = 0.0
        loss_mask[labels == vision_input_start_token] = 0.0
        loss_mask[labels == vision_input_end_token] = 0.0
        loss_mask[labels == vision_gen_start_token] = 0.0

    # if concat_data:
        
    # else:

    position_ids = torch.arange(input_seq_length, dtype=torch.long, device=labels.device) # FIX THIS #TODO
    position_ids = position_ids.unsqueeze(0).expand(batch_size, input_seq_length)
    return attention_mask, loss_mask, position_ids, shifted_multimodal_position_ids, labels

def local_rank():
    """Local rank of process"""
    local_rank = os.environ.get("LOCAL_RANK")

    if local_rank is None:
        local_rank = os.environ.get("SLURM_LOCALID")

    if local_rank is None:
        print(
            "utils.local_rank() environment variable LOCAL_RANK not set, defaulting to 0",
            flush=True,
        )
        local_rank = 0
    return int(local_rank)


def is_bnb_available():
    """True if bitsandbytes optimizers are available"""
    return importlib.util.find_spec("bitsandbytes") is not None


def is_local_main():
    """True if is the local main process"""
    return local_rank() == 0


def is_mp_rank_0():
    """True if mp rank == 0"""
    return mpu.get_model_parallel_rank() == 0


def get_wandb_api_key(neox_args):
    """Get Weights and Biases API key from ENV or .netrc file. Otherwise return None"""
    if "WANDB_LOCAL" in os.environ:
        return "LOCAL"
    if "WANDB_API_KEY" in os.environ:
        return os.environ["WANDB_API_KEY"]

    wandb_token = requests.utils.get_netrc_auth(neox_args.wandb_host)

    if wandb_token is not None:
        return wandb_token[1]


def init_wandb(neox_args):
    # Wandb. (one worker per machine)
    if neox_args.use_wandb == False:
        return

    if not neox_args.wandb_init_all_ranks:
        use_wandb = is_local_main() and (
            get_wandb_api_key(neox_args=neox_args) is not None
        )
        neox_args.update_value("use_wandb", use_wandb)
    if neox_args.use_wandb:
        group_name = neox_args.wandb_group
        name = f"{socket.gethostname()}-{local_rank()}" if group_name else None
        print("Logging wandb to:", neox_args.wandb_dir, flush=True)
        try:
            wandb.init(
                project=neox_args.wandb_project,
                group=group_name,
                name=name,
                save_code=False,
                force=False,
                entity=neox_args.wandb_team,
                dir=neox_args.wandb_dir,
            )
        except wandb.UsageError as e:
            neox_args.update_value("use_wandb", False)
            print(e)
            print(
                "Skipping wandb. Execute `wandb login` on local or main node machine to enable.",
                flush=True,
            )
        wandb.config.update(neox_args.all_config)


def obtain_resource_pool(
    hostfile_path, include_arg, exclude_arg
) -> Dict[str, List[int]]:
    """
    Get dict of `resource_pool[hostname] = [list of GPU ranks]` using hostfile, include and exclude args.
    Modified from: `deepspeed.launcher.runner.main`
    """
    resource_pool = fetch_hostfile(hostfile_path)
    if not resource_pool:
        resource_pool = {}
        device_count = torch.cuda.device_count()
        if device_count == 0:
            raise RuntimeError("Unable to proceed, no GPU resources available")
        resource_pool["localhost"] = device_count

    active_resources = parse_inclusion_exclusion(
        resource_pool, include_arg, exclude_arg
    )
    return active_resources


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)


def ddb(rank=0):
    """
    Distributed Debugger that will insert a py debugger on rank `rank` and
    pause all other distributed processes until debugging is complete.
    :param rank:
    """
    if torch.distributed.get_rank() == rank:
        from pdb import Pdb

        pdb = Pdb(skip=["torch.distributed.*"])
        pdb.set_trace(sys._getframe().f_back)
    torch.distributed.barrier()


class Timer:
    """Timer."""

    def __init__(self, name):
        self.name_ = name
        self.elapsed_ = 0.0
        self.started_ = False
        self.start_time = time.time()

    def start(self):
        """Start the timer."""
        assert not self.started_, "timer has already been started"
        torch.cuda.synchronize()
        self.start_time = time.time()
        self.started_ = True

    def stop(self):
        """Stop the timer."""
        assert self.started_, "timer is not started"
        torch.cuda.synchronize()
        self.elapsed_ += time.time() - self.start_time
        self.started_ = False

    def reset(self):
        """Reset timer."""
        self.elapsed_ = 0.0
        self.started_ = False

    def elapsed(self, reset=True):
        """Calculate the elapsed time."""
        started_ = self.started_
        # If the timing in progress, end it first.
        if self.started_:
            self.stop()
        # Get the elapsed time.
        elapsed_ = self.elapsed_
        # Reset the elapsed time
        if reset:
            self.reset()
        # If timing was in progress, set it back.
        if started_:
            self.start()
        return elapsed_


class Timers:
    """Group of timers."""

    def __init__(self, use_wandb, tensorboard_writer):
        self.timers = {}
        self.use_wandb = use_wandb
        self.tensorboard_writer = tensorboard_writer

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = Timer(name)
        return self.timers[name]

    def write(self, names, iteration, normalizer=1.0, reset=False):
        """Write timers to a tensorboard writer"""
        # currently when using add_scalars,
        # torch.utils.add_scalars makes each timer its own run, which
        # pollutes the runs list, so we just add each as a scalar
        assert normalizer > 0.0
        for name in names:
            value = self.timers[name].elapsed(reset=reset) / normalizer

            if self.tensorboard_writer:
                self.tensorboard_writer.add_scalar(f"timers/{name}", value, iteration)

            if self.use_wandb:
                wandb.log({f"timers/{name}": value}, step=iteration)

    def log(self, names, normalizer=1.0, reset=True):
        """Log a group of timers."""
        assert normalizer > 0.0
        string = "time (ms)"
        for name in names:
            elapsed_time = self.timers[name].elapsed(reset=reset) * 1000.0 / normalizer
            string += " | {}: {:.2f}".format(name, elapsed_time)
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                print(string, flush=True)
        else:
            print(string, flush=True)


def expand_attention_types(attention_config, num_layers):
    """
    Expands an `attention_config` list in the following format:

        [
        [['attention_type_1', ..., `attention_type_n`], 12]
        ]

    to a flattened list of length `num_layers`.

    :param params_list:
    :return:
    """
    # if only strings are found in the config, we assume it's already expanded
    if all([isinstance(i, str) for i in attention_config]):
        return attention_config
    newlist = []
    for item in attention_config:
        # instead of specifying a number - we can specify 'all' to extend this pattern across all layers
        if item[1] == "all":
            assert num_layers % len(item[0]) == 0, (
                f"Number of layers ({num_layers}) is not divisible by the length "
                f"of pattern: {item[0]}"
            )
            return item[0] * (num_layers // len(item[0]))
        for _ in range(item[1]):
            newlist.extend(item[0])
    return newlist


class OverflowMonitor:

    """
    Checks if the past n iterations have been skipped due to overflow, and exits
    training if that happens.
    """

    def __init__(self, optimizer, n=50):
        self.optimizer = optimizer
        self.n = n
        self.history = deque(maxlen=n)

    def check(self, skipped):
        self.history.append(skipped)
        if (
            self.optimizer.overflow
            and len(self.history) == self.n
            and all(self.history)
        ):
            raise Exception(
                f"Skipped {self.n} iterations in a row due to Overflow - Exiting training."
            )


def get_noise_scale_logger(neox_args):
    if neox_args.log_gradient_noise_scale:
        if neox_args.zero_stage >= 1:
            raise NotImplementedError(
                "Gradient Noise Scale logging does not work with zero stage 2+, as the "
                "gradients are distributed across ranks."
            )
        noise_scale_logger = GradientNoiseScale(
            model=model,
            batch_size_small=neox_args.train_batch_size,
            n_batches=neox_args.gradient_noise_scale_n_batches,
            cpu_offload=neox_args.gradient_noise_scale_cpu_offload,
            neox_args=neox_args,
            mpu=mpu,
        )
    else:
        noise_scale_logger = None
    return noise_scale_logger


def get_total_params(model):
    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        params = sum([p.nelement() for p in model.parameters()])
        print(
            " > number of parameters on model parallel rank {}: {}".format(
                mpu.get_model_parallel_rank(), params
            ),
            flush=True,
        )
    else:
        params = 0

    total_n_parameters = torch.tensor([params]).cuda(torch.cuda.current_device())
    torch.distributed.all_reduce(total_n_parameters)
    total_n_parameters = total_n_parameters.item()
    return total_n_parameters


def setup_for_inference_or_eval(
    use_cache=True,
    overwrite_values=None,
):
    """
    Initializes the model for evaluation or inference (doesn't load optimizer states, etc.) from command line args.

    use_cache: bool
        Whether to use key value caching in inference.
    overwrite_values: dict
        Optional Values to overwrite in the model config.
    """

    from megatron.neox_arguments import NeoXArgs
    from megatron.initialize import initialize_megatron
    from megatron.training import setup_model_and_optimizer

    _overwrite_values = {
        "checkpoint_activations": False,
        "partition_activations": False,
        "no_load_optim": True,
        "optimizer": None,  # prevent loading optimizer (no_load_optim alone won't work)
        "zero_optimization": None,  # disable zero optimization (won't be used in inference, and loading zero optimizer can cause errors)
    }
    if overwrite_values:
        _overwrite_values.update(overwrite_values)
    neox_args = NeoXArgs.consume_neox_args(overwrite_values=_overwrite_values)
    neox_args.configure_distributed_args()
    neox_args.build_tokenizer()

    if neox_args.load is None:
        raise ValueError("`load` parameter must be supplied to load a model`")

    # initialize wandb
    init_wandb(neox_args=neox_args)

    # initialize megatron
    initialize_megatron(neox_args)

    # set up model and load checkpoint.
    model, _, _ = setup_model_and_optimizer(
        neox_args=neox_args,
        use_cache=use_cache,
        iteration=neox_args.iteration,
    )  # we use setup_model_and_optimizer instead of get_model in order to initialize deepspeed
    print_rank_0("Finished loading model")

    model.module.inference_mode(use_cache=use_cache)
    return model, neox_args


class CharCounter:
    """
    Wraps the data_iterator to count the number of characters in a batch
    """

    def __init__(self, data_iterator, tokenizer):
        self.tokenizer = tokenizer
        self.data_iterator = data_iterator
        self.char_count = 0
        self.batch_count = 0
        self.token_count = 0
        self.total_time = 0

    def tokens_per_char(self):
        return self.token_count / self.char_count

    def __iter__(self):
        return self

    def __next__(self):
        start = time.time()
        batch = self.data_iterator.__next__()
        for b in batch["text_input"]:
            self.token_count += len(b)
            self.char_count += len(self.tokenizer.detokenize(b.tolist()))
        self.batch_count += 1
        end = time.time()
        self.total_time += end - start
        return batch
