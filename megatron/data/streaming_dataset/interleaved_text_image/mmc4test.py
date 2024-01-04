# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming dataset conversion scripts for json files."""
import os
from argparse import ArgumentParser, Namespace
from enum import Enum
from glob import glob
from typing import Dict, Iterable, Optional, List

from streaming import MDSWriter
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
# from transformers import AutoTokenizer, PreTrainedTokenizerBase

# Import Image type class 
from PIL import Image

import io
import struct

import os
import warnings
from typing import Dict, Iterable, Union

import numpy as np
from torch.utils.data import IterableDataset
# from transformers import PreTrainedTokenizerBase

import torch
from torch.nn.utils.rnn import pad_sequence
# from hamcrest.core.core.isnone import none

import datasets as hf_datasets
import lm_dataformat as lmd
from threading import Semaphore
# Import webdataset
import webdataset as wds

from streaming.base.format.mds.encodings import Encoding, _encodings

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torchvision

from megatron.tokenizer.tokenizer import build_tokenizer

from multiprocessing import Pool
import multiprocessing

from functools import partial

import json

import tarfile

from PIL import Image

import pandas as pd


class SEEDGenerationDataset(IterableDataset):
    def __init__(self, path, group):
        # List all jsonl files in the folder mentioned by path:
        all_json_ls = glob(path + "/*.jsonl")
        # Sort the list of jsonl files:
        all_json_ls.sort()
        # Get the start and end indices of the group:
        start, end = group
        # Get the jsonl files in the group:
        if end > len(all_json_ls):
            end = len(all_json_ls)
        self.paths = all_json_ls[start:end]
            
    def __iter__(self):
        for jsonl_path in self.paths:
            with open(jsonl_path) as f:
                for line in f:
                    curr_json = json.loads(line)
                    text = curr_json["caption"]
                    seed_tokens = curr_json["seed"]
                    generation_caption = "hello"  #random.choice(IMAGE_GENERATION_TEXT_VARIANTS)
                    text_list = [generation_caption[0], text, "<|image_gen_start|>"]
                    image_list = [None, None, None]
                    for seed_token in seed_tokens:
                        text_list.append(f'<|seed_{seed_token}|>')
                        image_list.append(None)
                    text_list.append("<|image_gen_end|>")
                    image_list.append(None)
                    print(text_list)
                    yield {
                        "images": image_list,
                        "text": text_list
                    }

# Main function
if __name__ == '__main__':
    mmc4_class = SEEDGenerationDataset("/p/scratch/ccstdl/chen24/datasets/laion2b_seed/", (0, 1))
    path = "/p/scratch/ccstdl/chen24/datasets/laion2b_seed/"
    all_json_ls = glob(path + "split_2B-en-*.jsonl")
    print("all_json_ls", len(all_json_ls))
    exit()
    for i, batch in enumerate(mmc4_class):
        print(batch)
        if i == 10:
            break


'''
"<|image_start|>": 50277,
      "<|image_end|>": 50278,
      "<|image_gen_start|>": 50279,
      "<|image_gen_end|>": 50280,
'''