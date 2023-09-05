# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import logging
from urllib.parse import urlparse

from . import vision_transformer as vits


logger = logging.getLogger("dinov2")

def load_pretrained_weights(model, pretrained_weights, checkpoint_key):
    if urlparse(pretrained_weights).scheme:  # If it looks like an URL
        state_dict = torch.hub.load_state_dict_from_url(pretrained_weights, map_location="cpu")
    else:
        state_dict = torch.load(pretrained_weights, map_location="cpu")
    if checkpoint_key is not None and checkpoint_key in state_dict:
        logger.info(f"Take key {checkpoint_key} in provided checkpoint dict")
        state_dict = state_dict[checkpoint_key]
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    logger.info("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))
    return model

def build_model(args, img_size=224):
    # args.arch = args.arch.removesuffix("_memeff")
    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            # init_values=args.layerscale,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            # qkv_bias=args.qkv_bias,
            # proj_bias=args.proj_bias,
            # ffn_bias=args.ffn_bias,
        )
        teacher = vits.__dict__[args.arch](**vit_kwargs)
        return teacher #, teacher.embed_dim

def get_pretrained_model_from_cfg(cfg):
    model = build_model(cfg, img_size=cfg.global_crops_size)
    model=load_pretrained_weights(model, cfg.pretrained_weights, "teacher")
    return model
