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
import pandas as pd

import random

IMAGE_UNDERSTANDING_TEXT_VARIANTS = [
("Describe this image", " "),
("The caption of the image", "is "),
("The image", "shows "),
("The image", "depicts "),
("This illustration", "represents "),
("The snapshot", "captures "),
("The scene", "consists of "),
("In the photo", "is "),
("This visual", "displays "),
("A picture of", "has "),
("The image", "features "),
("This graphic", "presents "),
("The image", "consists of "),
("The representation", "is of "),
("The photo", "captures "),
("This depiction", "reveals "),
("The scene", "shows "),
("The picture", "represents "),
("The image", "demonstrates "),
("In the illustration", "is "),
("This visual representation", "displays "),
("The photograph", "features "),
("The image", "presents "),
("This snapshot", "depicts "),
("The artwork", "shows "),
("The scene", "portrays "),
("This graphic", "represents "),
("This picture", "contains "),
("The image", "portrays "),
("In this visual", "is "),
("The illustration", "depicts "),
("This photo", "shows "),
("The image", "reveals "),
("The snapshot", "displays "),
("This picture", "presents "),
("The image", "illustrates "),
("This scene", "features "),
("The photograph", "represents "),
("The graphic", "depicts "),
("This illustration", "displays "),
("The picture", "demonstrates "),
("In the image", "is "),
("The visual", "presents "),
("This representation", "portrays "),
("The snapshot", "illustrates "),
("This photograph", "captures "),
("Can you describe what's in the image?", " "),
("What do you see in this picture?", " "),
("Tell me what's happening in the photo", " "),
("Explain the scene in the illustration", " "),
("What does this image represent?", " "),
("Provide a description of this snapshot", " "),
("What's going on in this graphic?", " "),
("Please give a brief summary of the photo", " "),
("What elements can you identify in this picture?", " "),
]

IMAGE_GENERATION_TEXT_VARIANTS = [
("Create an illustration of ", " "),
("Produce a visual depicting ", " "),
("Design a picture to show ", " "),
("Draw an image representing ", " "),
("Construct a visual representation of ", " "),
("Make a rendered image of ", " "),
("Put together a picture showcasing ", " "),
("Develop a visual piece on ", " "),
("Render an image about ", " "),
("Craft an illustration that embodies ", " "),
("Generate an artistic representation of ", " "),
("Create a visual render of ", " "),
("Build a visual image around ", " "),
("Put out an image illustrating ", " "),
("Produce a image of ", " "),
("Devise an image capturing ", " "),
("Whip up an image showcasing ", " "),
("Bring into life an image of ", " "),
]

class ListPIL(Encoding):
    """Store PIL image raw.

    Format: [width: 4] [height: 4] [mode size: 4] [mode] [raw image].
    """

    def encode(self, images: List[Image.Image]) -> bytes:
        # self._validate(images, List[Image.Image])
        final_bytes = b''
        for obj in images:
            mode = obj.mode.encode('utf-8')
            width, height = obj.size
            raw = obj.tobytes()
            ints = np.array([width, height, len(mode), len(raw)], np.uint32)
            final_bytes += ints.tobytes() + mode + raw
        return final_bytes

    def decode(self, data: bytes) -> List[Image.Image]:
        images = []
        idx = 4 * 4
        start = 0
        # print("Data length", len(data))
        while True:
            if start == len(data):
                break
            width, height, mode_size, raw_size = np.frombuffer(data[start:start+idx], np.uint32)
            # print("width, height, mode_size, raw_size", width, height, mode_size, raw_size)
            start = start + idx
            idx2 = start + mode_size
            # print("start", start, " idx2", idx2)
            mode = data[start:idx2].decode('utf-8')
            start = idx2
            size = width, height
            idx3 = start + raw_size
            raw = data[start:idx3]
            start = idx3
            images.append(Image.frombytes(mode, size, raw))  # pyright: ignore
        return images

_encodings['listpil'] = ListPIL

class ImageEncoding(Encoding):
    def encode(self, images: List[Image.Image]) -> bytes:
        bytes_arr = []
        for image in images:
            byte_io = io.BytesIO()
            image.save(byte_io, format='png') 
            bytes_arr.append(byte_io.getvalue())
        return b''.join(bytes_arr) 

    def decode(self, data: bytes) -> List[Image.Image]:
        images = []
        image_stream = io.BytesIO(data)
        while True:
            try:
                image = Image.open(image_stream)
                images.append(image)
                # Find the end of current img and skip \xFF\xD9 marker  
                image_stream.seek(image_stream.tell() + 2)
            except Exception:
                break
        return images

_encodings['ImageEncoding'] = ImageEncoding

class MultipleImageEncoding:
    def encode(self, images: List[Image.Image]) -> bytes:
        bytes_arr = []

        # Save number of images as first 4 bytes
        bytes_arr.append(struct.pack(">I", len(images))) 

        for image in images:
            byte_io = io.BytesIO()
            image.save(byte_io, format='JPEG')
            img_bytes = byte_io.getvalue()

            # Save size of image as 4 bytes before image bytes
            bytes_arr.append(struct.pack(">I", len(img_bytes))) 
            bytes_arr.append(img_bytes)

        return b''.join(bytes_arr)

    def decode(self, data: bytes) -> List[Image.Image]:
        images = []
        data_stream = io.BytesIO(data)

        # Read number of images
        num_images = struct.unpack(">I", data_stream.read(4))[0] 

        for _ in range(num_images):
            # Read size of image
            size_img = struct.unpack(">I", data_stream.read(4))[0] 

            # Read image bytes and open as PIL Image
            img_bytes = data_stream.read(size_img)
            image = Image.open(io.BytesIO(img_bytes))
            images.append(image)

        return images
    
_encodings['MultipleImageEncoding'] = MultipleImageEncoding

import pickle

class PickleEncoding(Encoding):
    def encode(self, data: List[Image.Image]) -> bytes:
        return pickle.dumps(data)

    def decode(self, data: bytes) -> np.ndarray:
        data = pickle.loads(data)
        # Convert PIL Images to numpy arrays
        data = map(lambda x: np.array(x), data)
        return np.stack(list(data))

_encodings['pickleencoding'] = PickleEncoding

class simple_encoding(Encoding):
    def encode(self, data: List[Image.Image]) -> bytes:
        if data == []:
            return np.array([]).tobytes()
        # Read all images into numpy array
        data = map(lambda x: np.array(x), data)
        data = np.stack(list(data))
        assert data.shape == (len(data), 256, 256, 3), f'Expected shape (N, 256, 256, 3), got {data.shape}'
        for img in data:
            assert img.dtype == np.uint8, f'Expected dtype np.uint8, got {img.dtype}'
        return data.tobytes()
    
    def decode(self, data: bytes) -> np.ndarray:
        # convert bytes to numpy array
        data = np.frombuffer(data, dtype=np.uint8)
        # reshape to original shape
        data = data.reshape(-1, 256, 256, 3)
        return data

_encodings['simple_encoding'] = simple_encoding

class NoConcatDataset(IterableDataset):
    """An IterableDataset that returns text samples for MDSWriter.

    Returns dicts of {'text': bytes}
    """

    def __init__(self, dataset: IterableDataset):
        self.dataset = dataset

    def __iter__(self) -> Iterable[Dict[str, bytes]]:
        for sample in self.dataset:
            # print(sample)
            # convert to bytes to store in MDS binary format
            yield {'text': sample["text"].encode('utf-8'), 'image': sample["images"]}


class ConcatTokensDataset(IterableDataset):
    """An IterableDataset that returns token samples for MDSWriter.

    Returns dicts of {'tokens': bytes}

    To use data created by this class and written to MDS format:

    ```python
        import torch
        from streaming.base import StreamingDataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained('your/tokenizer')
        ds = StreamingDataset(local='mds-data-folder', split='val')

        # note, you need to copy the numpy array because the original is non-writeable
        # and torch does not support non-writeable tensors, so you get a scary warning and
        # if you do try to write to the tensor you get undefined behavior
        tokens = torch.from_numpy(np.frombuffer(ds[0]['tokens'], dtype=np.int64).copy())
        print(tokenizer.decode(tokens))
    ```
    """

    def __init__(
        self,
        dataset: IterableDataset,
        tokenizer,
        max_length: int,
        image_seq_length: int,
        bos_text: str,
        eos_text: str,
        image_start_text: str,
        image_end_text: str,
        image_gen_start_text: str,
        no_wrap: bool,
        after_image_extra_tokens: int = 10, 
        position_pad_id: int = -1
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        self.max_length = max_length
        self.image_seq_length = image_seq_length
        self.image_start_text = image_start_text
        self.image_end_text = image_end_text
        self.after_image_extra_tokens = after_image_extra_tokens    
        self.bos_text = bos_text
        self.eos_text = eos_text
        self.pad_token_id = self.tokenizer.pad_id
        self.position_pad_id = position_pad_id
        self.should_wrap = not no_wrap
        self.image_gen_start_text = image_gen_start_text
        self.bos_tokens = self.tokenizer.tokenize(self.bos_text)
        if len(self.bos_tokens) > 1:
            warnings.warn(
                f'You specified --concat_tokens with --bos_text, but your BOS text is not tokenizing to one token\
                , instead we got {self.bos_tokens}. Quit if this was in error.')

        self.eos_tokens = self.tokenizer.tokenize(self.eos_text)
        print("eos token", self.eos_tokens)
        if len(self.eos_tokens) > 1:
            warnings.warn(
                f'You specified --concat_tokens with --eos_text, but your EOS text is not tokenizing to one token\
                , instead we got {self.eos_tokens}. Quit if this was in error.')
        
        self.image_start_token = self.tokenizer.tokenize(self.image_start_text)[0]
        
        self.image_end_token = self.tokenizer.tokenize(self.image_end_text)[0]
        
        eos_text_provided = self.eos_text != ''
        bos_text_provided = self.bos_text != ''
        test_text = self.tokenizer.tokenize('')
        if len(test_text) > 0 and (eos_text_provided or
                                                bos_text_provided):
            message = 'both eos and bos' if eos_text_provided and bos_text_provided else (
                'eos_text' if eos_text_provided else 'bos_text')
            warnings.warn(
                f'The provided tokenizer adds special tokens, but you also specified {message}. This may result '
                +
                'in duplicated special tokens. Please be sure this is what you intend.'
            )
        self.image_buffer = []
        self.text_buffer = []
        self.total_buffer = 0

    def __iter__(self) -> Iterable[Dict[str, bytes]]:

        current_length = 0
        curr_text = []
        curr_image = []
        
        for sample in self.dataset:     
            sample_text = sample["text"]
            sample_images = sample["images"]
            
            self.text_buffer.append(self.bos_tokens)
            self.image_buffer.insert(0, None) # To batch bos
            
            for section in sample_text:
                if section != None:
                    # Need to check this for max length however
                    self.text_buffer.append(self.tokenizer.tokenize(section))

                else:
                    self.text_buffer.append(None)
            
            self.text_buffer.append(self.eos_tokens)
            self.image_buffer.extend(sample_images)
            self.image_buffer.append(None)

            #We want to add text and image to our upcoming output (setup), and remove them from the buffer.
            while True:
                
                if len(self.text_buffer) == 0 and len(self.image_buffer) == 0:
                    break
                
                # Add assert with raise error
                assert (len(self.text_buffer) == len(self.image_buffer)), "Text and image buffer lengths do not match"
                
                text = self.text_buffer[0]
                image = self.image_buffer[0]

                assert (text != None and image == None) or (text == None and image != None), "Text and image are both none or both not None in the same sample"

                if text != None and image == None:
                    if self.image_gen_start_text in text and current_length + len(text) > self.max_length:
                        current_length = self.max_length
                    else:
                        # Fix saving the missing/original part of the text
                        if current_length + len(text) > self.max_length: # Too long, that's fine for text, just grab what we can
                            text_append = text[:self.max_length-current_length] # Changes the actual list in the thing
                            self.text_buffer[0] = text[self.max_length-current_length:]
                            # We do NOT pop an image here because we haven't finished the current text
                            # We also naturally do not pop text.
                            current_length = self.max_length
                            
                        else: # Not greater, remove entire text and entire image
                            text_append = self.text_buffer.pop(0)
                            self.image_buffer.pop(0)#Just remove the None for image
                            current_length += len(text_append)
                            
                        curr_text.extend(text_append)
                        curr_image.append(None)
                    
                elif text == None and image != None:
                    if current_length + self.image_seq_length + 2 + self.after_image_extra_tokens > self.max_length: # TODO: Make sure there is text remaining from current sample. Make sure in cases like grounding boxes and seed tokens, things dont break off in the middle 
                        current_length = self.max_length
                        
                    else: # So this includes that EOS case...
                        curr_image.extend([None, self.image_buffer.pop(0), None])
                        curr_text.extend([self.image_start_token, self.text_buffer.pop(0), self.image_end_token])
                        current_length += self.image_seq_length + 2
                else:
                    raise ValueError("Text and image are both none or both not None in the same sample")

                if current_length == self.max_length:
                    np_text = np.array(curr_text)

                    # length is total number of non None tokens 
                    text_length = len(np_text[np_text != None])
                    vision_length = len(np_text[np_text==None])
                    total_sample_length = text_length + vision_length*self.image_seq_length

                    if total_sample_length != self.max_length:
                        # Pad rest of the text tokens 
                        np_text = np.pad(np_text, (0, self.max_length - total_sample_length), constant_values = self.pad_token_id)

                    text_ids = np_text[np_text != None]
                    text_tokens = text_ids
                    text_positions = torch.from_numpy(np.where(np_text != None)[0])
                    
                    images = list(filter(lambda a: a != None, curr_image)) # FIX THIS
                    image_positions = torch.from_numpy(np.where(np_text == None)[0])
                    labels = np.roll(np_text, -1, axis = 0)
                    labels[-1] = self.pad_token_id

                    text_labels = labels[np_text != None]
                    image_labels = labels[np_text == None]

                    # Replace None with pad token in labels 
                    text_labels = np.where(text_labels == None, self.pad_token_id, text_labels).astype(np.int64)
                    image_labels = np.where(image_labels == None, self.pad_token_id, image_labels).astype(np.int64)

                    multimodal_position_ids = torch.nn.utils.rnn.pad_sequence([text_positions, image_positions], batch_first = True, padding_value = self.position_pad_id)

                    labels = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(text_labels), torch.from_numpy(image_labels)], batch_first = True, padding_value = self.pad_token_id)
                    
                    # convert tensor to numpy array
                    labels = labels.numpy().tobytes()
                    text_tokens = text_tokens.astype(np.int64)
                    text_tokens = text_tokens.tobytes()
                    multimodal_position_ids = multimodal_position_ids.numpy().tobytes()

                    yield {
                        'images': images,
                        'tokens': text_tokens,
                        'multimodal_position_ids' : multimodal_position_ids,
                        'labels': labels
                    }
                    
                    curr_image.clear()
                    curr_text.clear()
                    current_length = 0
                    
                elif current_length > self.max_length:
                    raise ValueError("Current length is greater than max length")
    
class ConcatMode(Enum):
    NO_CONCAT = 'NO_CONCAT'
    CONCAT_TOKENS = 'CONCAT_TOKENS'

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
        start, end = group
        fpath = f"{path}/{{{str(start).zfill(5)}..{str(end).zfill(5)}}}.tar"
        self.dataset = wds.WebDataset(fpath).decode("pilrgb").rename(image="jpg;png;jpeg;webp", text="txt", json="json").to_tuple("image", "text", "json")

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
            
class MMC4InterleavedDataset(IterableDataset):
    def __init__(self, path, group):
        start, end = group
        self.jsonl_paths = []
        for i in range(start, end):
            self.jsonl_paths.append(os.path.join(path, f"docs_shard_{i}_v2.jsonl"))
            
    def __iter__(self):
        for jsonl_path in self.jsonl_paths:

            jsonl_file_name = os.path.basename(jsonl_path)
            tar_file_name = jsonl_file_name.replace("docs_", "").replace("_v2.jsonl", "_images_v2.tar")
            tar_path = os.path.join("/p/fastdata/mmlaion/mmc4/images", tar_file_name)
            tar_file_name = tar_file_name.replace(".tar", "")
            with tarfile.open(tar_path) as tar:
                with open(jsonl_path) as f:
                    for line in f:
                        curr_json = json.loads(line)
                        text_list = curr_json["text_list"]
                        image_info_list = curr_json["image_info"]
                        for image_info in image_info_list:
                            image_path_name = image_info["image_name"]
                            matching_index = image_info["matched_text_index"]
                            image_path_name = os.path.join(tar_file_name, image_path_name)                            
                            file = tar.extractfile(image_path_name)
                            image_data = file.read()    

                            # Create a BytesIO object and load the image data
                            image = Image.open(io.BytesIO(image_data))
                            if image.mode != 'RGB':
                                image = image.convert('RGB')
                            
                            image = torchvision.transforms.functional.resize(image, [224, 224], interpolation=InterpolationMode.BICUBIC)
                            

                            text_list.insert(matching_index+1, image)
                        
                        final_text_list = []
                        final_image_list = []
                        for item in text_list:
                            if isinstance(item, str):
                                final_text_list.append(item)
                                final_image_list.append(None)
                            else:
                                final_image_list.append(item)
                                final_text_list.append(None)
                        yield {
                            "images": final_image_list,
                            "text": final_text_list
                        }

class SEEDGenerationDataset(IterableDataset):
    def __init__(self, path, group, image_gen_start_text, image_gen_end_text):
        # List all jsonl files in the folder mentioned by path:
        all_json_ls = glob(path + "split_2B-en-*.jsonl")
        # Sort the list of jsonl files:
        all_json_ls.sort()
        # Get the start and end indices of the group:
        start, end = group
        # Get the jsonl files in the group:
        if end > len(all_json_ls):
            end = len(all_json_ls)
        self.paths = all_json_ls[start:end]
        self.image_gen_start_text = image_gen_start_text
        self.image_gen_end_text = image_gen_end_text
            
    def __iter__(self):
        for jsonl_path in self.paths:
            with open(jsonl_path) as f:
                for line in f:
                    curr_json = json.loads(line)
                    text = curr_json["caption"]
                    seed_tokens = curr_json["seed"]
                    generation_caption = random.choice(IMAGE_GENERATION_TEXT_VARIANTS)
                    text_portion = generation_caption[0] + text + generation_caption[1]
                    for seed_token in seed_tokens:
                        text_portion+=f'<|seed_{seed_token}|>'
                    text_portion += self.image_gen_end_text
                    text_list = [text_portion]
                    image_list = [None]
                    yield {
                        "images": image_list,
                        "text": text_list
                    }

class InterleavedSEEDGenerationDataset(IterableDataset):
    def __init__(self, path, group):
        self.parquet_folder_list = []
        start, end = group
        for i in range(start, end):
            first_digit = i//4+1
            second_digit = i%4+1
            self.parquet_folder_list.append(path + "/split_2B-en-4_5_" + str(first_digit) + "_parquet_" + str(second_digit) + "_parquet")
        # pass
        self.seed_folder_path = '/p/scratch/ccstdl/chen24/datasets/laion2b_seed/'


    def __iter__(self):
        for parquet_folder in self.parquet_folder_list:
            # Get the brace notation of all available tar files 
            tar_files = glob(parquet_folder + "/*.tar")
            # brace notation
            brace_notation = "{" + ",".join(tar_files) + "}"
            # Get complete path
            dataset = wds.WebDataset(brace_notation).decode("pilrgb").rename(image="jpg;png;jpeg;webp", text="txt", id="__key__").to_tuple("image", "text", "id")
            seed_dataset_path = self.seed_folder_path + os.path.basename(parquet_folder) + ".jsonl"
            seed_dataset = pd.read_json(seed_dataset_path, lines=True)
            # make the key as the index
            seed_dataset = seed_dataset.set_index('key')
            # convert index to str
            seed_dataset.index = seed_dataset.index.map(str)
            for sample in dataset:
                image, text, id = sample
                image = torchvision.transforms.functional.resize(image, [224, 224], interpolation=InterpolationMode.BICUBIC)
                # search sample in seed dataset based on id
                id = id.lstrip('0')
                sample = seed_dataset.loc[id]
                generation_caption = random.choice(IMAGE_GENERATION_TEXT_VARIANTS)
                text_list = [generation_caption[0], text, "<|image_gen_start|>"]
                image_list = [None, None, None]
                for seed_token in sample['seed']:
                    text_list.append(f'<|seed_{seed_token}|>')
                    image_list.append(None)
                text_list.append("<|image_gen_end|>")
                image_list.append(None)
                yield {
                    "images": image_list,
                    "text": text_list
                }

class TextConcatDataset(IterableDataset):
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
        for fname in self.paths:
            print("Processing file: ", fname)
            for doc in filter(lambda x: x, lmd.Reader(fname).stream_data()):
                sample = {
                    "images": [None],
                    "text": [doc]
                }
                yield sample

class ImageCaptionDataset(IterableDataset):
    def __init__(self, path, group):
        start, end = group
        fpath = f"{path}/{{{str(start).zfill(5)}..{str(end).zfill(5)}}}.tar"
        self.dataset = iter(wds.WebDataset(fpath).decode("pilrgb").rename(image="jpg;png;jpeg;webp", text="txt").to_tuple("image", "text"))

    def __iter__(self):
        while True:
            try:
                image, text = next(self.dataset)
                image = torchvision.transforms.functional.resize(image, [224, 224], interpolation=InterpolationMode.BICUBIC)
                if text is None:
                    print("key 'text' not found in the sample, skipping this datapoint")
                    continue
                text_understanding_caption = random.choice(IMAGE_UNDERSTANDING_TEXT_VARIANTS)
                yield {
                    "images": [None, image, None, None],
                    "text": [text_understanding_caption[0], None, text_understanding_caption[1], text]
                }
            except StopIteration:
                break
            except ValueError as e:
                print(f"Error encountered: {e}. Skipping this datapoint.")
                continue
            except Exception as e:
                print(f"Unexpected Error encountered: {e}. Skipping this datapoint.")
                continue

def build_interleaved_multimodal_dataset(   
    path: str,
    group: tuple, 
    mode: ConcatMode,
    max_length: Optional[int] = None,
    bos_text: str = '',
    eos_text: str = '',
    image_start_text: str = '<|image_start|>',
    image_end_text: str = '<|image_end|>',
    image_gen_start_text: str = '<|image_gen_start|>',
    image_gen_end_text: str = '<|image_gen_end|>',
    no_wrap: bool = False,
    tokenizer = None,
    vision_seq_length: int = 64,
    after_image_extra_tokens: int = 10,
    position_pad_id: int = -1
):
    """Build an IterableDataset over the HF C4 or pile source data.

    Args:
        dataset_name (str): Dataset name
        mode (ConcatMode): NO_CONCAT, or CONCAT_TOKENS
        max_length (int): The length of concatenated tokens
        bos_text (str): text to insert at the beginning of each sequence
        eos_text (str): text to insert at the end of each sequence
        no_wrap (bool): if concatenating, whether to wrap text across `max_length` boundaries
        tokenizer (PreTrainedTokenizerBase): if mode is CONCAT_TOKENS, the tokenizer to use
        data_subset (str): Referred to as "name" in HuggingFace datasets.load_dataset.
            Typically "all" (The Pile) or "en" (c4).

    Returns:
        An IterableDataset.
    """

    # dataset = ImageCaptionDataset(path, group)
    # dataset = TextConcatDataset(path, group)
    dataset = MMC4InterleavedDataset(path, group)
    # dataset = SEEDGenerationDataset(path, group, image_gen_start_text, image_gen_end_text)
    # dataset = GritDatasetGeneration(path, group)

    if mode == ConcatMode.NO_CONCAT:
        dataset = NoConcatDataset(dataset)
    else:
        if max_length is None:
            raise ValueError(f'max_length must be set.')
        if bos_text + eos_text == '':
            test_tokens = tokenizer('test')
            if test_tokens[
                    0] != tokenizer.bos_token_id and test_tokens[
                        -1] != tokenizer.eos_token_id:
                tok_error_msg = 'This tokenizer does not insert an EOS nor BOS token. '
                tok_error_msg += 'Concatenating with this tokenizer will result in sequences being '
                tok_error_msg += 'attached without a separating token. Please use another tokenizer, '
                tok_error_msg += 'such as facebook/opt-125m, or specify EOS/BOS text with e.g. '
                tok_error_msg += '--bos_text=<|endoftext|>.'
                raise ValueError(tok_error_msg)
            
        dataset = ConcatTokensDataset(
            dataset=dataset,
            tokenizer=tokenizer,
            max_length=max_length,
            image_seq_length=vision_seq_length,
            bos_text=bos_text,
            eos_text=eos_text,
            image_start_text=image_start_text,
            image_end_text=image_end_text,
            image_gen_start_text=image_gen_start_text,
            no_wrap=no_wrap, 
            after_image_extra_tokens=after_image_extra_tokens,
            position_pad_id=position_pad_id
        )
    for sample in tqdm(dataset):
        yield sample

def data_generator(task_queue, data_queue, args, worker_id):

    if args.concat_tokens is not None:
        mode = ConcatMode.CONCAT_TOKENS
        args.rank = 0
        args.model_parallel_size = 1
        args.make_vocab_size_divisible_by = 128
        tokenizer = build_tokenizer(args)
        # we will enforce length, so suppress warnings about sequences too long for the model
        tokenizer.model_max_length = int(1e30)
    else:
        mode = ConcatMode.NO_CONCAT
        tokenizer = None

    partial_builder = partial(build_interleaved_multimodal_dataset, 
                                path=args.path,
                                mode=mode,
                                max_length=args.concat_tokens,
                                bos_text=tokenizer.bos_text,
                                eos_text=tokenizer.eos_text,
                                image_start_text=tokenizer.image_start_text,
                                image_end_text=tokenizer.image_end_text,
                                image_gen_start_text=tokenizer.image_gen_start_text,
                                image_gen_end_text=tokenizer.image_gen_end_text,
                                no_wrap=args.no_wrap,
                                tokenizer=tokenizer, 
                                vision_seq_length=args.vision_seq_length,
                                after_image_extra_tokens=args.after_image_extra_tokens,
                                position_pad_id=args.position_pad_id)
    
    while not task_queue.empty():
        group = task_queue.get()
        start, end = group
        print(f'Worker {worker_id} started processing data: {start}-{end}')
        for data in partial_builder(group=group):
            data_queue.put(data)
        print(f'Worker {worker_id} finished processed data: {start}-{end}')

def data_writer(data_queue, args, index):
    if args.concat_tokens is not None:
        columns = {'tokens': 'bytes', 'images': 'listpil', 'multimodal_position_ids': 'bytes', 'labels': 'bytes'}
    else:
        columns = {'text': 'str', 'images': 'ndarray'} 

    with MDSWriter(columns=columns,
                out=os.path.join(f"{args.out_root}/{index}"),
                compression=args.compression, size_limit=1e+9) as out:
        
        total_samples = 0
        total_images = 0
        while True:    
            print("The queue size is", data_queue.qsize())
            try:
                sample = data_queue.get(timeout=100)
                total_samples += 1
                total_images += len(sample["images"])
                out.write(sample)
                print(f'\rWriter {index} Writing sample {total_samples} with {total_images} images.........', flush=True, end='')
                if total_samples > 3000:
                    break
            except multiprocessing.queues.Empty:
                print(f'\rNo more data to write. Exiting. {index}')
                break

def get_dataset_groups(start_ind:int, end_ind:int, groups: int):
    """Get the sub-directory path and the sample range.

    Args:
        out_root (str): base output mds directory
        groups (int): Number of sub-directories to create

    Yields:
        Iterator[Tuple[str, int, int]]: Each argument tuple
    """
    group_size = (end_ind - start_ind) // groups
    for group_start in range(start_ind, end_ind, group_size):
        yield (group_start, group_start + group_size)

def main(args: Namespace) -> None:
    """Main: create C4/pile streaming dataset.

    Args:
        args (Namespace): Commandline arguments.
    """
    # Write samples
    print(f'Converting to MDS format...')
    print(
        f'Note that the progress bar is based on the dataset length before tokenization.'
    )
    print(f'It will finish at a value below 100% if tokenizing')

    dataset_group_iterator = get_dataset_groups(args.start_ind, args.end_ind, args.num_groups)

    # args.rank = 0
    # args.model_parallel_size = 1
    # args.make_vocab_size_divisible_by = 128
    # tokenizer = build_tokenizer(args)
    # tokenizer.tokenizer.add_special_tokens([f'<|p|>'])
    # tokenizer.tokenizer.add_special_tokens([f'<|/p|>'])
    # tokenizer.tokenizer.add_special_tokens([f'<|box|>'])
    # tokenizer.tokenizer.add_special_tokens([f'<|/box|>'])
    # tokenizer.tokenizer.add_special_tokens([f'<|grounding|>'])

    # for i in range(1024):
    #     tokenizer.tokenizer.add_special_tokens([f'<|box_{i}|>'])
    # tokenizer.tokenizer.save("/p/project/ccstdl/gupta6/multimodal/20B_tokenizer_new.json")
    # exit()

    task_queue = multiprocessing.Queue()
    for index_range in dataset_group_iterator:
        task_queue.put(index_range)

    data_queue = multiprocessing.Queue(maxsize=args.queue_size)

    workers = []
    for i in range(args.workers): 
        worker_process = multiprocessing.Process(target=data_generator, args=(task_queue, data_queue, args, i))
        worker_process.start()
        workers.append(worker_process)

    # writers 
    writers = []

    for i in range(args.num_writers):
        writer_process = multiprocessing.Process(target=data_writer, args=(data_queue, args, i))
        writer_process.start()
        writers.append(writer_process)

    # Wait for all the workers to finish
    for worker in workers:
        worker.join()

    # Now the master can terminate
    for writer in writers:
        writer.join()

def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description=
        'Convert dataset into MDS format, optionally concatenating and tokenizing'
    )
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--out_root', type=str, required=True)
    parser.add_argument('--compression', type=str, default=None)

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        '--concat_tokens',
        type=int,
        help='Convert text to tokens and concatenate up to this many tokens')
    parser.add_argument('--queue_size', type=int, default=5000)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--num_groups', type=int, default=100)
    parser.add_argument('--workers', type=int, default=80) # 44
    parser.add_argument('--num_writers', type=int, default=40) # 2
    parser.add_argument('--start_ind', type=int, default=250)
    parser.add_argument('--end_ind', type=int, default=6000) #150
    parser.add_argument('--tokenizer_type', type=str, required=False, default=None)
    parser.add_argument('--vocab_file', type=str, required=False, default=None)
    parser.add_argument('--merge_file', type=str, required=False, default=None)
    parser.add_argument('--no_wrap', default=False, action='store_true')
    parser.add_argument('--vision_seq_length', type=int, default=64)
    parser.add_argument('--after_image_extra_tokens', type=int, default=10)
    parser.add_argument('--position_pad_id', type=int, default=-1)

    parsed = parser.parse_args()

    if os.path.isdir(parsed.out_root) and len(
            set(os.listdir(parsed.out_root)).intersection(set(
                parsed.split))) > 0:
        raise ValueError(
            f'--out_root={parsed.out_root} contains {os.listdir(parsed.out_root)} which cannot overlap with the requested splits {parsed.splits}.'
        )
    else:
        os.makedirs(parsed.out_root)
    # Make sure we have needed concat options
    if (parsed.concat_tokens is not None and
            isinstance(parsed.concat_tokens, int) and parsed.tokenizer_type is None):
        parser.error(
            'When setting --concat_tokens, you must specify a tokenizer')

    return parsed

if __name__ == '__main__':
    main(parse_args())

'''
python megatron/data/streaming_dataset/interleaved_text_image/create_interleaved_dataset.py --path /p/fastdata/mmlaion/laion-400m/LAION-400m-webdataset/data --compression zstd --concat_tokens 2048 --tokenizer_type HFTokenizer --vocab_file /p/project/ccstdl/gupta6/multimodal/20B_tokenizer.json --out_root /p/fastdata/mmlaion/hummingbird/hummingbird_dataset_final/laion_400m_train

python megatron/data/streaming_dataset/interleaved_text_image/create_interleaved_dataset.py --path /p/fastdata/mmlaion/mmc4/unzipped_docs/ --compression zstd --concat_tokens 2048 --tokenizer_type HFTokenizer --vocab_file /p/project/ccstdl/gupta6/multimodal/20B_tokenizer.json --out_root /p/fastdata/mmlaion/hummingbird/hummingbird_dataset_final/mmc4_train

python megatron/data/streaming_dataset/interleaved_text_image/create_interleaved_dataset.py --path /p/fastdata/mmlaion/EVOBITS --compression zstd --concat_tokens 2048 --tokenizer_type HFTokenizer --vocab_file /p/project/ccstdl/gupta6/multimodal/20B_tokenizer.json --out_root /p/fastdata/mmlaion/hummingbird/hummingbird_dataset/seed_train

python megatron/data/streaming_dataset/interleaved_text_image/create_interleaved_dataset.py --path /p/scratch/ccstdl/chen24/datasets/laion2b_seed/ --compression zstd --concat_tokens 2048 --tokenizer_type HFTokenizer --vocab_file /p/project/ccstdl/gupta6/multimodal/20B_tokenizer.json --out_root /p/fastdata/mmlaion/hummingbird/hummingbird_dataset/seed_train

python megatron/data/streaming_dataset/interleaved_text_image/create_interleaved_dataset.py --path /p/fastdata/mmlaion/hummingbird/grit --compression zstd --concat_tokens 2048 --tokenizer_type HFTokenizer --vocab_file /p/project/ccstdl/gupta6/multimodal/20B_tokenizer.json --out_root /p/fastdata/mmlaion/hummingbird/hummingbird_dataset/grit_train

python megatron/data/streaming_dataset/interleaved_text_image/create_interleaved_dataset.py --path /p/fastdata/mmlaion/hummingbird/SlimPajama-627B/train/chunk1 --compression zstd --concat_tokens 2048 --tokenizer_type HFTokenizer --vocab_file /p/project/ccstdl/gupta6/multimodal/20B_tokenizer.json --out_root /p/fastdata/mmlaion/hummingbird/hummingbird_dataset_final/text_train_chunk1
'''
