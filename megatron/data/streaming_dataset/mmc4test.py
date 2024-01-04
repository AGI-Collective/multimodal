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


class MMC4InterleavedDataset(IterableDataset):
    def __init__(self, path):
        dataset = wds.WebDataset(path).decode("pilrgb").to_tuple("image", "__key__")
        self.dataset = iter(dataset)

    def __iter__(self):
        for image, key in self.dataset:
            image = torchvision.transforms.functional.resize(image, [224, 224], interpolation=InterpolationMode.BICUBIC)
            yield {
                "images": [None, image],
                "text": [key, None]
            }

# Main
