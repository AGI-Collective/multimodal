# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming dataset conversion scripts for json files."""
import os
from argparse import ArgumentParser, Namespace
from typing import Dict, Iterable, Optional, List
from enum import Enum
import glob
from glob import glob
from tqdm import tqdm
import warnings
import io
import multiprocessing
from functools import partial
import tarfile
import pandas as pd
import random
import time

import torch
from torch.utils.data import IterableDataset
from torchvision.transforms.functional import InterpolationMode
import torchvision

import lm_dataformat as lmd

from PIL import Image
import numpy as np

import webdataset as wds

from streaming import MDSWriter
from streaming.base.format.mds.encodings import Encoding, _encodings

from megatron.tokenizer.tokenizer import build_tokenizer

IMAGE_SIZE = 336

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
    ("Create an illustration of ", ""),
    ("Produce a visual depicting ", ""),
    ("Design a picture to show ", ""),
    ("Draw an image representing ", ""),
    ("Construct a visual representation of ", ""),
    ("Make a rendered image of ", ""),
    ("Put together a picture showcasing ", ""),
    ("Develop a visual piece on ", ""),
    ("Render an image about ", ""),
    ("Craft an illustration that embodies ", ""),
    ("Generate an artistic representation of ", ""),
    ("Create a visual render of ", ""),
    ("Build a visual image around ", ""),
    ("Put out an image illustrating ", ""),
    ("Produce a image of ", ""),
    ("Devise an image capturing ", ""),
    ("Whip up an image showcasing ", ""),
    ("Bring into life an image of ", ""),
]


class ListPIL(Encoding):
    def encode(self, images: List[Image.Image], quality: int = 95) -> bytes:
        final_bytes = b""
        for obj in images:
            byte_arr = io.BytesIO()
            obj.save(byte_arr, format='JPEG', quality=quality)
            raw = byte_arr.getvalue()
            raw_len = len(raw)
            ints = np.array([raw_len], np.uint32)
            final_bytes += ints.tobytes() + raw
        return final_bytes

    def decode(self, data: bytes) -> List[Image.Image]:
        images = []
        start = 0
        while start < len(data):
            raw_size = np.frombuffer(data[start : start + 4], np.uint32)[0]
            start += 4
            raw = data[start:start + raw_size]
            start += raw_size
            byte_arr = io.BytesIO(raw)
            img = Image.open(byte_arr)
            images.append(img)
        return images


def check_image(
    image, min_image_size=0, max_image_area=float("inf"), max_aspect_ratio=float("inf")
):
    width, height = image.size
    if min(height, width) < min_image_size:
        return False
    if height * width > max_image_area:
        return False
    if max(height, width) / min(height, width) > max_aspect_ratio:
        return False
    return True


DIMENSIONS = [(1, 1), (2, 2), (1, 2), (1, 3), (1, 4), (2, 1), (3, 1), (4, 1)]
MAX_SPLITS = 4

def zigzag_order(scaled_image_np, new_height, new_width, box_length):
    sub_images = []

    for y in range(0, new_height, box_length):
        if y // box_length % 2 == 0:  # Even rows are left-to-right
            x_values = range(0, new_width, box_length)
        else:  # Odd rows are right-to-left
            x_values = range(new_width - box_length, -box_length, -box_length)
        
        for x in x_values:
            sub_images.append(scaled_image_np[y:y+box_length, x:x+box_length])
    
    return sub_images

def split_images_fn(image, box_length):

    width, height = image.size

    # Scaling the image to fit box_length
    dimensions = DIMENSIONS
    original_aspect_ratio = width / height
    scaled_configurations = [
        {
            "dim": (w * box_length, h * box_length),
            "area_diff": abs(w * h * box_length**2 - width * height),
            "aspect_ratio_diff": abs(original_aspect_ratio - w / h),
        }
        for w, h in dimensions
    ]

    # Sort list of configurations: first by area difference, then aspect ratio difference
    sorted_configurations = sorted(
        scaled_configurations,
        key=lambda config: (config["area_diff"], config["aspect_ratio_diff"]),
    )

    # Choose the first configuration (the one with lowest area difference and aspect ratio difference)
    chosen_configuration = sorted_configurations[0]["dim"]

    if chosen_configuration != (1*box_length, 1*box_length):
        if random.random() < 0.5: # Split less than 50% of the time
            new_width, new_height = chosen_configuration
            scaled_image = torchvision.transforms.functional.resize(
                image, [new_height, new_width], interpolation=InterpolationMode.BICUBIC
            )
            # Splitting the image into a grid of sub-images of size: box_length X box_length
            scaled_image_np = np.array(scaled_image)
            sub_images = zigzag_order(scaled_image_np, new_height, new_width, box_length)

            # Converting numpy arrays back to images
            sub_images = [Image.fromarray(sub_image) for sub_image in sub_images]
        else:
            sub_images = []
    else:
        sub_images = []
    
    sub_images.append(torchvision.transforms.functional.resize(image, [box_length, box_length], interpolation=InterpolationMode.BICUBIC))

    # Returns a list of image objects
    return sub_images


_encodings["listpil"] = ListPIL


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
            yield {"text": sample["text"].encode("utf-8"), "image": sample["images"]}


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
        no_wrap: bool,
        after_image_extra_tokens: int = 32,
        before_image_extra_tokens: int = 32,
        position_pad_id: int = -1,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.max_length = max_length
        self.image_seq_length = image_seq_length
        self.image_start_text = image_start_text
        self.image_end_text = image_end_text
        self.after_image_extra_tokens = after_image_extra_tokens
        self.before_image_extra_tokens = before_image_extra_tokens
        self.bos_text = bos_text
        self.eos_text = eos_text
        self.pad_token_id = self.tokenizer.pad_id
        self.position_pad_id = position_pad_id
        self.should_wrap = not no_wrap
        self.bos_tokens = self.tokenizer.tokenize(self.bos_text)
        self.seed_token_length = 32
        if len(self.bos_tokens) > 1:
            warnings.warn(
                f"You specified --concat_tokens with --bos_text, but your BOS text is not tokenizing to one token\
                , instead we got {self.bos_tokens}. Quit if this was in error."
            )

        self.eos_tokens = self.tokenizer.tokenize(self.eos_text)
        print("eos token", self.eos_tokens)
        if len(self.eos_tokens) > 1:
            warnings.warn(
                f"You specified --concat_tokens with --eos_text, but your EOS text is not tokenizing to one token\
                , instead we got {self.eos_tokens}. Quit if this was in error."
            )

        self.image_start_token = self.tokenizer.tokenize(self.image_start_text)[0]
        self.image_end_token = self.tokenizer.tokenize(self.image_end_text)[0]
        eos_text_provided = self.eos_text != ""
        bos_text_provided = self.bos_text != ""
        test_text = self.tokenizer.tokenize("")
        if len(test_text) > 0 and (eos_text_provided or bos_text_provided):
            message = (
                "both eos and bos"
                if eos_text_provided and bos_text_provided
                else ("eos_text" if eos_text_provided else "bos_text")
            )
            warnings.warn(
                f"The provided tokenizer adds special tokens, but you also specified {message}. This may result "
                + "in duplicated special tokens. Please be sure this is what you intend."
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
            self.image_buffer.insert(0, None)  # To batch bos

            for section in sample_text:
                if section != None:
                    # Need to check this for max length however
                    self.text_buffer.append(self.tokenizer.tokenize(section))

                else:
                    self.text_buffer.append(None)

            self.text_buffer.append(self.eos_tokens)
            self.image_buffer.extend(sample_images)
            self.image_buffer.append(None)

            # We want to add text and image to our upcoming output (setup), and remove them from the buffer.
            while True:

                if len(self.text_buffer) == 0 and len(self.image_buffer) == 0:
                    break

                # Add assert with raise error
                assert len(self.text_buffer) == len(
                    self.image_buffer
                ), "Text and image buffer lengths do not match"

                text = self.text_buffer[0]
                image = self.image_buffer[0]

                assert (text != None and image == None) or (
                    text == None and image != None
                ), "Text and image are both none or both not None in the same sample"

                def calculate_image_splits(image_buffer, index):
                    image_splits = 0
                    cur_i = index
                    while image_buffer[cur_i] != None:
                        image_splits += 1
                        cur_i += 1
                    return image_splits

                if text != None and image == None:
                    index_to_add_till = len(text)
                    if self.image_start_token in text:
                        image_ind = text.index(self.image_start_token)
                        if (
                            current_length
                            + image_ind
                            + self.seed_token_length
                            + self.image_seq_length
                            * calculate_image_splits(self.image_buffer, 1)
                            + 2
                            + self.after_image_extra_tokens
                        ) > self.max_length:
                            if text[0] == self.image_end_token:
                                balance = 1
                            else:
                                balance = 0
                            index_to_add_till = max(
                                balance, image_ind - self.before_image_extra_tokens
                            )  # Add max number of tokens possible (with minimum 1 (image end token) if the previous one was an image)

                    if (
                        current_length + len(text) > self.max_length
                    ):  # Too long, that's fine for text, just grab what we can
                        index_to_add_till = min(
                            index_to_add_till, self.max_length - current_length
                        )

                    if index_to_add_till < len(text):
                        text_append = text[:index_to_add_till]
                        self.text_buffer[0] = text[
                            index_to_add_till:
                        ]  # We do NOT pop an image here because we haven't finished the current text
                        current_length = self.max_length
                    else:
                        text_append = self.text_buffer.pop(0)
                        self.image_buffer.pop(0)
                        current_length += len(text_append)
                    curr_text.extend(text_append)
                    curr_image.append(None)

                elif text == None and image != None:
                    if (
                        current_length
                        + self.image_seq_length
                        * calculate_image_splits(self.image_buffer, 0)
                        + self.after_image_extra_tokens
                        + 1
                        > self.max_length
                    ):  # TODO: Make sure in cases like grounding boxes things dont break off in the middle
                        current_length = self.max_length

                    else:  # So this includes that EOS case...
                        curr_image.extend([self.image_buffer.pop(0)])
                        curr_text.extend([self.text_buffer.pop(0)])
                        current_length += self.image_seq_length
                else:
                    raise ValueError(
                        "Text and image are both none or both not None in the same sample"
                    )

                if current_length == self.max_length:
                    np_text = np.array(curr_text)

                    # length is total number of non None tokens
                    text_length = len(np_text[np_text != None])
                    vision_length = len(np_text[np_text == None])
                    total_sample_length = (
                        text_length + vision_length * self.image_seq_length
                    )

                    if total_sample_length != self.max_length:
                        # Pad rest of the text tokens
                        np_text = np.pad(
                            np_text,
                            (0, self.max_length - total_sample_length),
                            constant_values=self.pad_token_id,
                        )

                    text_ids = np_text[np_text != None]
                    text_tokens = text_ids
                    text_positions = torch.from_numpy(np.where(np_text != None)[0])

                    images = list(filter(lambda a: a != None, curr_image))  # FIX THIS
                    image_positions = torch.from_numpy(np.where(np_text == None)[0])
                    labels = np.roll(np_text, -1, axis=0)
                    labels[-1] = self.pad_token_id

                    text_labels = labels[np_text != None]
                    image_labels = labels[np_text == None]

                    # Replace None with pad token in labels
                    text_labels = np.where(
                        text_labels == None, self.pad_token_id, text_labels
                    ).astype(np.int64)
                    image_labels = np.where(
                        image_labels == None, self.pad_token_id, image_labels
                    ).astype(np.int64)

                    multimodal_position_ids = torch.nn.utils.rnn.pad_sequence(
                        [text_positions, image_positions],
                        batch_first=True,
                        padding_value=self.position_pad_id,
                    )

                    labels = torch.nn.utils.rnn.pad_sequence(
                        [torch.from_numpy(text_labels), torch.from_numpy(image_labels)],
                        batch_first=True,
                        padding_value=self.pad_token_id,
                    )

                    # convert tensor to numpy array
                    labels = labels.numpy().tobytes()
                    text_tokens = text_tokens.astype(np.int64)
                    text_tokens = text_tokens.tobytes()
                    multimodal_position_ids = multimodal_position_ids.numpy().tobytes()

                    yield {
                        "images": images,
                        "tokens": text_tokens,
                        "multimodal_position_ids": multimodal_position_ids,
                        "labels": labels,
                    }

                    curr_image.clear()
                    curr_text.clear()
                    current_length = 0

                elif current_length > self.max_length:
                    raise ValueError("Current length is greater than max length")


class ConcatMode(Enum):
    NO_CONCAT = "NO_CONCAT"
    CONCAT_TOKENS = "CONCAT_TOKENS"


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
                sample = {"images": [None], "text": [doc]}
                yield sample

class LLaVADataset(IterableDataset):
    def __init__(self, path, group, image_start_text, image_end_text):
        start, end = group
        fpath = f"{path}/{{{str(start).zfill(5)}..{str(end).zfill(5)}}}.tar"
        self.dataset = iter(
            wds.WebDataset(fpath)
            .decode("pilrgb")
            .to_tuple("jpg;png;jpeg;webp", "json", "__key__", "__url__")
        )

        self.seed_parquet_folder = "/p/fastdata/mmlaion/hummingbird/temp_llava_seed"
        self.image_start_text = image_start_text
        self.image_end_text = image_end_text
        self.current_parquet_path = None
        self.current_loaded_parquet = None
    
    def __iter__(self):
        while True:
            try:
                image, metadata, key, url = next(self.dataset)
                id = str(metadata["id"])
                basename = os.path.basename(url)
                parquet_shard = int(os.path.splitext(basename)[0])
                parquet_path = f"{self.seed_parquet_folder}/{str(parquet_shard).zfill(5)}.parquet"
                if self.current_parquet_path != parquet_path:
                    self.current_parquet_path = parquet_path
                    self.current_loaded_parquet = pd.read_parquet(parquet_path)
                    self.current_loaded_parquet.set_index("id", inplace=True)
                seed_tokens = np.frombuffer(self.current_loaded_parquet.loc[[id]]["seed_tokens"].iloc[0], dtype=np.int64)
                split_images = split_images_fn(image, IMAGE_SIZE)
                conversations = metadata["conversations"]

                for i in range(0, len(conversations), 2):

                    question = conversations[i]["value"]
                    answer = conversations[i + 1]["value"]
                    if "<image>" in question:
                        question = question.replace("<image>", "")
                    
                    text_portion = self.image_start_text + "".join(
                        [f"<|seed_{seed_token}|>" for seed_token in seed_tokens]
                    )

                    image_list = [None]
                    text_list = [text_portion]
                    for split_image in split_images:
                        image_list.append(split_image)
                        text_list.append(None)
                    
                    image_list.append(None)
                    text_list.append(self.image_end_text + question + " " + answer)
                    yield {"images": image_list, "text": text_list}
            except StopIteration:
                break
            except ValueError as e:
                print(f"Error encountered: {e}. Skipping this datapoint.")
                continue
            except Exception as e:
                print(f"Unexpected Error encountered: {e}. Skipping this datapoint.")
                continue

class OBELICSDataset(IterableDataset):
    def __init__(self, group, image_start_text, image_end_text):
        start, end = group
        obelics_original_path = "/p/fastdata/mmlaion/OBELICS_parquet"
        self.parquets = []
        for i in range(start, end):
            self.parquets.append(
                os.path.join(
                    obelics_original_path,
                    f"{obelics_original_path}/obelics-train-{str(i).zfill(5)}-of-01335.parquet",
                )
            )
        self.images_path = "/p/fastdata/mmlaion/obelics_img"
        self.converted_parquets = "/p/fastdata/mmlaion/OBELICS_converted_sample"
        self.current_tar_file = None
        self.current_tar = None
        self.seed_folder = "/p/fastdata/mmlaion/obelics_seed_all"
        self.current_seed_parquet_file = None
        self.current_seed_parquet = None
        self.image_start_text = image_start_text
        self.image_end_text = image_end_text

    def __iter__(self):
        for parquet in self.parquets:
            image_folder = parquet.replace(
                "/p/fastdata/mmlaion/OBELICS_parquet", f"{self.images_path}/"
            )
            image_folder = image_folder.replace(".parquet", "")
            original_df = pd.read_parquet(parquet)
            converted_df = pd.read_parquet(
                parquet.replace(
                    "/p/fastdata/mmlaion/OBELICS_parquet", f"{self.converted_parquets}"
                )
            )
            # Set the "images" column in converted_df to be the key
            converted_df = converted_df.set_index("images")
            for idx, row in tqdm(original_df.iterrows()):
                images = row["images"]
                final_images = []
                final_texts = []
                assert len(images) == len(row["texts"])
                for i, image_url in enumerate(images):

                    if image_url is None:
                        final_images.append(None)
                        final_texts.append(row["texts"][i])
                        continue

                    if i == 0:
                        final_images.append(None)
                        final_texts.append("")

                    row_number = converted_df.index.get_loc(image_url)
                    # if row number is array then use the first element, else if its slice then use the start of the slice
                    if isinstance(row_number, np.ndarray):
                        row_number = row_number[0]
                    elif isinstance(row_number, slice):
                        row_number = row_number.start
                    else:
                        row_number = row_number

                    shard_id = row_number // 10000
                    idx_in_shard = row_number % 10000
                    image_id = f"{shard_id:05d}{idx_in_shard:04d}"

                    tar_file = image_folder + f"/{shard_id:05d}.tar"

                    if self.current_tar_file != tar_file:
                        self.current_tar_file = tar_file
                        if self.current_tar is not None:
                            self.current_tar.close()
                        self.current_tar = tarfile.open(tar_file)

                    seed_path = tar_file.replace(self.images_path, self.seed_folder)
                    seed_path = seed_path.replace(".tar", ".parquet")
                    if self.current_seed_parquet_file != seed_path:
                        self.current_seed_parquet_file = seed_path
                        self.current_seed_parquet = pd.read_parquet(seed_path)
                        self.current_seed_parquet = self.current_seed_parquet.set_index(
                            "key"
                        )
                    # Get seed tokens
                    try:
                        seed_tokens = np.frombuffer(
                            self.current_seed_parquet.loc[image_id]["seed_tokens"],
                            dtype=np.int64,
                        )
                        image = self.current_tar.extractfile(f"{image_id}.jpg")  # FIX
                        image_data = image.read()
                        image = Image.open(io.BytesIO(image_data))
                        if image.mode != "RGB":
                            image = image.convert("RGB")
                        split_images = split_images_fn(image, IMAGE_SIZE)

                        final_texts[-1] = (
                            final_texts[-1].rstrip()
                            + self.image_start_text
                            + "".join(
                                [f"<|seed_{seed_token}|>" for seed_token in seed_tokens]
                            )
                        )

                        for split_image in split_images:
                            final_images.append(split_image)
                            final_texts.append(None)

                        if row["texts"][i + 1] is not None:
                            row["texts"][i + 1] = (
                                self.image_end_text + row["texts"][i + 1]
                            )
                        else:
                            final_images.append(None)
                            final_texts.append(self.image_end_text)

                    except Exception as e:
                        continue
                yield {"images": final_images, "text": final_texts}


class DatacompDataset(IterableDataset):
    def __init__(self, path, group, image_start_text, image_end_text, mode):
        start, end = group
        fpath = f"{path}/{{{str(start).zfill(7)}..{str(end).zfill(7)}}}.tar"
        self.dataset = iter(
            wds.WebDataset(fpath)
            .decode("pilrgb")
            .to_tuple("jpg;png;jpeg;webp", "json", "__key__", "__url__", "txt")
        )

        self.seed_datacomp_folder_map = {
            0: "/p/fastdata/mmlaion/seed_tokens_datacomp1b_0_to_10",
            1: "/p/scratch/laionize/datacomp_seed_jz",
            2: "/p/fastdata/mmlaion/seed_tokens_datacomp1b_20_to_30",
            3: "/p/scratch/laionize/datacomp_seed_jz",
            4: "/p/fastdata/mmlaion/seed_tokens_datacomp1b_40_to_50",
        }
        self.image_start_text = image_start_text
        self.image_end_text = image_end_text
        self.current_parquet_path = None
        self.current_loaded_parquet = None
        self.mode = mode

    def __iter__(self):
        while True:
            try:
                image, metadata, key, url, text = next(self.dataset)

                split_images = split_images_fn(image, IMAGE_SIZE)

                if text is None:
                    print("key 'text' not found in the sample, skipping this datapoint")
                    continue

                basename = os.path.basename(url)
                parquet_shard = int(os.path.splitext(basename)[0])
                folder_path = self.seed_datacomp_folder_map[int(parquet_shard / 10000)]
                parquet_path = f"{folder_path}/{str(parquet_shard).zfill(7)}.parquet"

                if self.current_parquet_path != parquet_path:
                    self.current_parquet_path = parquet_path
                    self.current_loaded_parquet = pd.read_parquet(parquet_path)
                    self.current_loaded_parquet.set_index("key", inplace=True)

                seed_tokens = np.frombuffer(
                    self.current_loaded_parquet.loc[metadata["key"]]["seed_tokens"],
                    dtype=np.int64,
                )

                if self.mode == "understanding":
                    text_understanding_caption = random.choice(
                        IMAGE_UNDERSTANDING_TEXT_VARIANTS
                    )
                    text_portion = (
                        text_understanding_caption[0].rstrip()
                        + self.image_start_text
                        + "".join(
                            [f"<|seed_{seed_token}|>" for seed_token in seed_tokens]
                        )
                    )

                    images = [None]
                    texts = [text_portion]
                    for split_image in split_images:
                        images.append(split_image)
                        texts.append(None)
                    images.append(None)
                    texts.append(
                        self.image_end_text + text_understanding_caption[1] + text
                    )

                elif self.mode == "generation":
                    generation_caption = random.choice(IMAGE_GENERATION_TEXT_VARIANTS)
                    text_portion = (
                        generation_caption[0]
                        + text
                        + generation_caption[1].rstrip()
                        + self.image_start_text
                        + "".join(
                            [f"<|seed_{seed_token}|>" for seed_token in seed_tokens]
                        )
                    )

                    images = [None]
                    texts = [text_portion]
                    for split_image in split_images:
                        images.append(split_image)
                        texts.append(None)
                    images.append(None)
                    texts.append(self.image_end_text)

                else:
                    raise ValueError("Mode not understood")

                yield {"images": images, "text": texts}
            except StopIteration:
                break
            except ValueError as e:
                print(f"Error encountered: {e}. Skipping this datapoint.")
                continue
            except Exception as e:
                print(f"Unexpected Error encountered: {e}. Skipping this datapoint.")
                continue


class GritDatasetGeneration(IterableDataset):
    def __init__(self, group, image_start_text, image_end_text):
        start, end = group

        base_path = "/p/fastdata/mmlaion/GRIT_img"
        all_files = []

        # iterate over the range
        for i in range(start, end + 1):
            # construct the folder path with padded zero
            folder = f"{base_path}/grit-train-{str(i).zfill(5)}-of-00021"

            # get list of .parquet files in the folder
            # and add to the overall list
            all_files += glob(os.path.join(folder, "*.tar"))

        self.dataset = iter(
            wds.WebDataset(all_files)
            .decode("pilrgb")
            .rename(image="jpg;png;jpeg;webp", text="txt", json="json")
            .to_tuple("image", "text", "json", "__url__")
        )

        self.seed_folder = "/p/fastdata/mmlaion/seed_tokens_grit"
        self.image_start_text = image_start_text
        self.image_end_text = image_end_text
        self.current_parquet_path = None
        self.current_loaded_parquet = None

    def get_bin_number(self, width, height, bins, x, y):
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

    def __iter__(self):
        while True:
            try:
                image, text, sample_json, url = next(self.dataset)
                text = sample_json["caption"]
                
                image = torchvision.transforms.functional.resize(
                    image, [IMAGE_SIZE, IMAGE_SIZE], interpolation=InterpolationMode.BICUBIC
                )

                current_path = url

                # replace "GRIT_img" with "seed_tokens_grit"
                seed_folder = current_path.replace("GRIT_img", "seed_tokens_grit")

                # replace .tar with .parquet
                parquet_path = seed_folder.replace(".tar", ".parquet")

                # use self.seed_folder to get appropriate parquet file.

                if self.current_parquet_path != parquet_path:
                    self.current_parquet_path = parquet_path
                    self.current_loaded_parquet = pd.read_parquet(parquet_path)
                    self.current_loaded_parquet.set_index("key", inplace=True)

                seed_tokens = np.frombuffer(
                    self.current_loaded_parquet.loc[sample_json["key"]]["seed_tokens"],
                    dtype=np.int64,
                )

                text_portion = self.image_start_text + "".join(
                    [f"<|seed_{seed_token}|>" for seed_token in seed_tokens]
                )

                temp_text = self.image_end_text + "<|grounding|>"

                image_list = [None, image]
                text_list = [text_portion, None]

                # sort ref expressions based on the first value
                sample_json["ref_exps"].sort(key=lambda x: x[0])
                i = 1
                while i < len(sample_json["ref_exps"]):
                    if sample_json["ref_exps"][i][0] < sample_json["ref_exps"][i - 1][1]:
                        # remove this ref exp
                        sample_json["ref_exps"].pop(i)
                    else:
                        i += 1

                last_text_index = 0
                for ref_exp in sample_json["ref_exps"]:
                    start_text = int(ref_exp[0])
                    end_text = int(ref_exp[1])

                    if start_text > last_text_index:
                        # text_list.append(text[last_text_index:start_text])
                        temp_text += text[last_text_index:start_text]
                        # image_list.append(None)

                    top_left = [ref_exp[2], ref_exp[3]]
                    bottom_right = [ref_exp[4], ref_exp[5]]

                    top_left[0] = int(top_left[0] * IMAGE_SIZE)
                    top_left[1] = int(top_left[1] * IMAGE_SIZE)
                    bottom_right[0] = int(bottom_right[0] * IMAGE_SIZE)
                    bottom_right[1] = int(bottom_right[1] * IMAGE_SIZE)

                    top_left_bin = self.get_bin_number(
                        IMAGE_SIZE, IMAGE_SIZE, 32, top_left[0], top_left[1]
                    )
                    bottom_right_bin = self.get_bin_number(
                        IMAGE_SIZE, IMAGE_SIZE, 32, bottom_right[0], bottom_right[1]
                    )

                    referring_expression = text[start_text:end_text]

                    temp_text += (
                        "<|p|>"
                        + referring_expression
                        + "<|/p|><|box|>"
                        + f"<|box_{top_left_bin}|>"
                        + f"<|box_{bottom_right_bin}|>"
                        + "<|/box|>"
                    )

                    last_text_index = end_text
                if last_text_index < len(text):
                    temp_text += text[last_text_index:]

                text_list.append(temp_text)
                image_list.append(None)
                assert len(text_list) == len(image_list)
                yield {"images": image_list, "text": text_list}

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
    dataset_type: str,
    group: tuple,
    mode: ConcatMode,
    max_length: Optional[int] = None,
    bos_text: str = "",
    eos_text: str = "",
    image_start_text: str = "<|image_start|>",
    image_end_text: str = "<|image_end|>",
    no_wrap: bool = False,
    tokenizer=None,
    vision_seq_length: int = 64,
    after_image_extra_tokens: int = 10,
    position_pad_id: int = -1,
    datacomp_mode=None,
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
    if dataset_type == "datacomp":
        dataset = DatacompDataset(
            path, group, image_start_text, image_end_text, datacomp_mode
        )
    elif dataset_type == "text":
        dataset = TextConcatDataset(path, group)
    elif dataset_type == "grit":
        dataset = GritDatasetGeneration(group, image_start_text, image_end_text)
    elif dataset_type == "obelics":
        dataset = OBELICSDataset(group, image_start_text, image_end_text)
    elif dataset_type == "llava":
        dataset = LLaVADataset(path, group, image_start_text, image_end_text)
    else:
        raise ValueError(f"Dataset {dataset} not recognized.")

    if mode == ConcatMode.NO_CONCAT:
        dataset = NoConcatDataset(dataset)
    else:
        if max_length is None:
            raise ValueError(f"max_length must be set.")
        if bos_text + eos_text == "":
            test_tokens = tokenizer("test")
            if (
                test_tokens[0] != tokenizer.bos_token_id
                and test_tokens[-1] != tokenizer.eos_token_id
            ):
                tok_error_msg = "This tokenizer does not insert an EOS nor BOS token. "
                tok_error_msg += (
                    "Concatenating with this tokenizer will result in sequences being "
                )
                tok_error_msg += "attached without a separating token. Please use another tokenizer, "
                tok_error_msg += (
                    "such as facebook/opt-125m, or specify EOS/BOS text with e.g. "
                )
                tok_error_msg += "--bos_text=<|endoftext|>."
                raise ValueError(tok_error_msg)

        dataset = ConcatTokensDataset(
            dataset=dataset,
            tokenizer=tokenizer,
            max_length=max_length,
            image_seq_length=vision_seq_length,
            bos_text=bos_text,
            image_start_text=image_start_text,
            image_end_text=image_end_text,
            eos_text=eos_text,
            no_wrap=no_wrap,
            after_image_extra_tokens=after_image_extra_tokens,
            position_pad_id=position_pad_id,
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

    partial_builder = partial(
        build_interleaved_multimodal_dataset,
        path=args.path,
        dataset_type=args.dataset_type,
        mode=mode,
        max_length=args.concat_tokens,
        bos_text=tokenizer.bos_text,
        eos_text=tokenizer.eos_text,
        image_start_text=tokenizer.image_start_text,
        image_end_text=tokenizer.image_end_text,
        no_wrap=args.no_wrap,
        tokenizer=tokenizer,
        vision_seq_length=args.vision_seq_length,
        after_image_extra_tokens=args.after_image_extra_tokens,
        position_pad_id=args.position_pad_id,
        datacomp_mode=args.datacomp_mode,
    )

    while not task_queue.empty():
        group = task_queue.get()
        start, end = group
        print(f"Worker {worker_id} started processing data: {start}-{end}")
        for data in partial_builder(group=group):
            data_queue.put(data)
        print(f"Worker {worker_id} finished processed data: {start}-{end}")


def data_writer(data_queue, args, index):
    if args.concat_tokens is not None:
        columns = {
            "tokens": "bytes",
            "images": "listpil",
            "multimodal_position_ids": "bytes",
            "labels": "bytes",
        }
    else:
        columns = {"text": "str", "images": "ndarray"}

    with MDSWriter(
        columns=columns,
        out=os.path.join(f"{args.out_root}/{index}"),
        compression=args.compression,
        size_limit=5e8,
    ) as out:

        total_samples = 0
        total_images = 0
        while True:
            # print("The queue size is", data_queue.qsize())
            try:
                sample = data_queue.get(timeout=100)
                total_samples += 1
                total_images += len(sample["images"])
                out.write(sample)
                # print(
                #     f"\rWriter {index} Writing sample {total_samples} with {total_images} images.........",
                #     flush=True,
                #     end="",
                # )
                # if total_samples % 500 == 0:
                #     end_time = time.time()
                #     time_taken = end_time - start_time
                #     print(f"\nTime taken for 1000 samples: {time_taken} seconds")
                #     start_time = time.time()  # reset start time
                # if total_samples > 1000:
                #     break
            except multiprocessing.queues.Empty:
                print(f"\rNo more data to write. Exiting. {index}")
                break


def get_dataset_groups(start_ind: int, end_ind: int, groups: int):
    """Get the sub-directory path and the sample range.

    Args:
        out_root (str): base output mds directory
        groups (int): Number of sub-directories to create

    Yields:
        Iterator[Tuple[str, int, int]]: Each argument tuple
    """
    group_size = (end_ind - start_ind) // groups
    for group_start in range(start_ind, end_ind+1, group_size):
        yield (group_start, group_start + group_size - 1)


def main(args: Namespace) -> None:
    """Main: create C4/pile streaming dataset.

    Args:
        args (Namespace): Commandline arguments.
    """
    # Write samples
    print(f"Converting to MDS format...")
    print(
        f"Note that the progress bar is based on the dataset length before tokenization."
    )
    print(f"It will finish at a value below 100% if tokenizing")

    dataset_group_iterator = get_dataset_groups(
        args.start_ind, args.end_ind, args.num_groups
    )

    task_queue = multiprocessing.Queue()
    for index_range in dataset_group_iterator:
        task_queue.put(index_range)

    data_queue = multiprocessing.Queue(maxsize=args.queue_size)

    workers = []
    for i in range(args.workers):
        worker_process = multiprocessing.Process(
            target=data_generator, args=(task_queue, data_queue, args, i)
        )
        worker_process.start()
        workers.append(worker_process)

    # writers
    writers = []

    for i in range(args.num_writers):
        writer_process = multiprocessing.Process(
            target=data_writer, args=(data_queue, args, i)
        )
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
        description="Convert dataset into MDS format, optionally concatenating and tokenizing"
    )
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--compression", type=str, default=None)
    parser.add_argument("--dataset_type", type=str, default=None)
    parser.add_argument("--datacomp_mode", type=str, default=None)

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--concat_tokens",
        type=int,
        help="Convert text to tokens and concatenate up to this many tokens",
    )
    parser.add_argument("--queue_size", type=int, default=5000)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num_groups", type=int, default=22)
    parser.add_argument("--workers", type=int, default=22)  # 44       # 80
    parser.add_argument("--num_writers", type=int, default=26)  # 2
    parser.add_argument("--start_ind", type=int, default=0)
    parser.add_argument("--end_ind", type=int, default=62)  # 150
    parser.add_argument("--tokenizer_type", type=str, required=False, default=None)
    parser.add_argument("--vocab_file", type=str, required=False, default=None)
    parser.add_argument("--merge_file", type=str, required=False, default=None)
    parser.add_argument("--no_wrap", default=False, action="store_true")
    parser.add_argument("--vision_seq_length", type=int, default=32)
    parser.add_argument("--after_image_extra_tokens", type=int, default=10)
    parser.add_argument("--position_pad_id", type=int, default=-1)

    parsed = parser.parse_args()

    if (
        os.path.isdir(parsed.out_root)
        and len(set(os.listdir(parsed.out_root)).intersection(set(parsed.split))) > 0
    ):
        raise ValueError(
            f"--out_root={parsed.out_root} contains {os.listdir(parsed.out_root)} which cannot overlap with the requested splits {parsed.splits}."
        )
    else:
        os.makedirs(parsed.out_root)
    # Make sure we have needed concat options
    if (
        parsed.concat_tokens is not None
        and isinstance(parsed.concat_tokens, int)
        and parsed.tokenizer_type is None
    ):
        parser.error("When setting --concat_tokens, you must specify a tokenizer")

    return parsed


if __name__ == "__main__":
    main(parse_args())

"""
number of groups 100
1e9, 10, 46
python megatron/data/streaming_dataset/interleaved_text_image/create_unified_interleaved_dataset.py --path /p/fastdata/mmlaion/hummingbird/SlimPajama-627B/train/chunk2 --dataset_type text --compression zstd --concat_tokens 2048 --tokenizer_type HFTokenizer --vocab_file /p/project/ccstdl/gupta6/multimodal/20B_tokenizer.json --out_root /p/fastdata/mmlaion/hummingbird/hummingbird_dataset_final/text_val_chunk2

5e8, 22, 26
python megatron/data/streaming_dataset/interleaved_text_image/create_unified_interleaved_dataset.py --path /p/fastdata/mmlaion/datacomp/datacomp_1B/flat --dataset_type datacomp --datacomp_mode understanding --compression zstd --concat_tokens 2048 --tokenizer_type HFTokenizer --vocab_file /p/project/ccstdl/gupta6/multimodal/20B_tokenizer.json --out_root /p/fastdata/mmlaion/hummingbird/hummingbird_dataset_final/datacomp_val_understanding

5e8, 22, 26
python megatron/data/streaming_dataset/interleaved_text_image/create_unified_interleaved_dataset.py --path /p/fastdata/mmlaion/datacomp/datacomp_1B/flat --dataset_type datacomp --datacomp_mode generation --compression zstd --concat_tokens 2048 --tokenizer_type HFTokenizer --vocab_file /p/project/ccstdl/gupta6/multimodal/20B_tokenizer.json --out_root /p/fastdata/mmlaion/hummingbird/hummingbird_dataset_final/datacomp_val_generation

5e8, 30, 18
python megatron/data/streaming_dataset/interleaved_text_image/create_unified_interleaved_dataset.py --path /p/fastdata/mmlaion/OBELICS_parquet --dataset_type obelics --compression zstd --concat_tokens 2048 --tokenizer_type HFTokenizer --vocab_file /p/project/ccstdl/gupta6/multimodal/20B_tokenizer.json --out_root /p/fastdata/mmlaion/hummingbird/hummingbird_dataset_final/obelics_val

20 groups
5e8, 22, 26
python megatron/data/streaming_dataset/interleaved_text_image/create_unified_interleaved_dataset.py --path /p/fastdata/mmlaion/hummingbird/grit --dataset_type grit --compression zstd --concat_tokens 2048 --tokenizer_type HFTokenizer --vocab_file /p/project/ccstdl/gupta6/multimodal/20B_tokenizer.json --out_root /p/fastdata/mmlaion/hummingbird/hummingbird_dataset_final/grit_val


python megatron/data/streaming_dataset/interleaved_text_image/create_unified_interleaved_dataset.py --path /p/fastdata/mmlaion/llava_v1_5_mix665k --dataset_type llava --compression zstd --concat_tokens 2048 --tokenizer_type HFTokenizer --vocab_file /p/project/ccstdl/gupta6/multimodal/20B_tokenizer.json --out_root /p/fastdata/mmlaion/hummingbird/hummingbird_dataset_final/test_llava
"""
