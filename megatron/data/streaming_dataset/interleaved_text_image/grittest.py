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


def get_bin_number(width, height, bins, x, y):
    """
    Get the bin number from the point 2D coordinates

    Parameters:
    width: the width of the image
    height: the height of the image
    bins: the number of bins per axis
    x: the x-coordinate of the point
    y: the y-coordinate of the point

    Return:
    bin_number: the bin number containing (x, y)
    """

    # Step 1: Calculate the width and height of each bin
    bin_width = width / bins
    bin_height = height / bins

    # Step 2: Determine the bin number in x-axis and y-axis
    bin_x = int(x // bin_width)
    bin_y = int(y // bin_height)

    # Account for edge cases where x or y equals width or height
    if x == width:
        bin_x -= 1
    if y == height:
        bin_y -= 1

    # Step 3: Combine the bin_x and bin_y to get the bin number
    # Numbering is from top to bottom, and then from left to right
    bin_number = bin_y * bins + bin_x
    
    return bin_number

class GritDatasetGeneration(IterableDataset):
    def __init__(self, path, group):
        self.path = path
        self.dataset = wds.WebDataset(self.path).decode("pilrgb").rename(image="jpg;png;jpeg;webp", text="txt", json="json").to_tuple("image", "text", "json")

    def __iter__(self):
        for sample in self.dataset:
            sample_json = sample[2]
            text = sample_json["caption"]
            image = sample[0]
            image = torchvision.transforms.functional.resize(image, [224, 224], interpolation=InterpolationMode.BICUBIC)
            image_list = [image, None]
            text_list = [None, "<|grounding|>"]
            
            # sort ref expressions based on the first value
            sample_json['ref_exps'].sort(key=lambda x: x[0])
            i = 1
            while i < len(sample_json['ref_exps']):
                if sample_json['ref_exps'][i][0] < sample_json['ref_exps'][i - 1][1]:
                    # remove this ref exp
                    sample_json['ref_exps'].pop(i)
                else:
                    i += 1

            last_text_index = 0
            for ref_exp in sample_json['ref_exps']:
                start_text = int(ref_exp[0])
                end_text = int(ref_exp[1])

                if start_text > last_text_index:
                    text_list.append(text[last_text_index:start_text])
                    image_list.append(None)
                
                top_left = [ref_exp[2], ref_exp[3]]
                bottom_right = [ref_exp[4], ref_exp[5]]

                top_left[0] = int(top_left[0]  * 224)
                top_left[1] = int(top_left[1]  * 224)
                bottom_right[0] = int(bottom_right[0]  * 224)
                bottom_right[1] = int(bottom_right[1]  * 224)

                top_left_bin = get_bin_number(224, 224, 32, top_left[0], top_left[1])
                bottom_right_bin = get_bin_number(224, 224, 32, bottom_right[0], bottom_right[1])

                referring_expression = text[start_text:end_text]
                text_list.append("<|p|>")
                text_list.append(referring_expression)
                text_list.append("<|/p|>")
                text_list.append("<|box|>")
                text_list.append(f"<|box_{top_left_bin}|>")
                text_list.append(f"<|box_{bottom_right_bin}|>")
                text_list.append("<|/box|>")
                # extend image list with 7 nones
                image_list.extend([None] * 7)
                last_text_index = end_text
            if last_text_index < len(text):
                text_list.append(text[last_text_index:])
                image_list.append(None)
            assert len(text_list) == len(image_list)
            yield {
                "images": image_list,
                "text": text_list
            }


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
    mmc4_class = GritDatasetGeneration("/p/fastdata/mmlaion/hummingbird/grit/00000.tar", (0, 1))
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