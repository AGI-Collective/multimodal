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
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# Import Image type class 
from PIL import Image

import io
import struct

import os
import warnings
from typing import Dict, Iterable, Union

import numpy as np
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase

import torch
from torch.nn.utils.rnn import pad_sequence
# from hamcrest.core.core.isnone import none

# Import webdataset
import webdataset as wds

from streaming.base.format.mds.encodings import Encoding, _encodings

class ImageEncoding:
    def encode(self, images: List[Image.Image]) -> bytes:
        bytes_arr = []
        for image in images:
            byte_io = io.BytesIO()
            image.save(byte_io, format='JPEG') 
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
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        image_seq_length: int,
        bos_text: str,
        eos_text: str,
        image_start_text: str,
        image_end_text: str,
        no_wrap: bool,
        after_image_extra_tokens: int = 10 
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
        self.should_wrap = not no_wrap

        self.bos_tokens = self.tokenizer(self.bos_text,
                                         truncation=False,
                                         padding=False,
                                         add_special_tokens=False)['input_ids']
        if len(self.bos_tokens) > 1:
            warnings.warn(
                f'You specified --concat_tokens with --bos_text, but your BOS text is not tokenizing to one token\
                , instead we got {self.bos_tokens}. Quit if this was in error.')

        self.eos_tokens = self.tokenizer(self.eos_text,
                                         truncation=False,
                                         padding=False,
                                         add_special_tokens=False)['input_ids']
        print("eos token", self.eos_tokens)
        if len(self.eos_tokens) > 1:
            warnings.warn(
                f'You specified --concat_tokens with --eos_text, but your EOS text is not tokenizing to one token\
                , instead we got {self.eos_tokens}. Quit if this was in error.')

        eos_text_provided = self.eos_text != ''
        bos_text_provided = self.bos_text != ''
        test_text = self.tokenizer('')
        if len(test_text['input_ids']) > 0 and (eos_text_provided or
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
            text = sample["text"]
            images = sample["images"]
            
            self.text_buffer.append(self.bos_tokens)
            self.image_buffer.insert(0, None) # To batch bos
            
            for section in text:
                if section != None:
                    # Need to check this for max length however
                    self.text_buffer.append(self.tokenizer(section, truncation=False, padding=False)["input_ids"])

                else:
                    self.text_buffer.append(None)
            
            self.text_buffer.append(self.eos_tokens)
            self.image_buffer.extend(images)
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
                    # Fix saving the missing/original part of the text
                    if current_length + len(text) > self.max_length: # Too long, that's fine for text, just grab what we can
                        current_length = self.max_length
                        text_append = text[:self.max_length-current_length] # Changes the actual list in the thing
                        text[0] = text[self.max_length-current_length:]
                        # We do NOT pop an image here because we haven't finished the current text
                        # We also naturally do not pop text.
                        
                    else: # Not greater, remove entire text and entire image
                        text_append = self.text_buffer.pop(0)
                        self.image_buffer.pop(0)#Just remove the None for image
                        current_length += len(text_append)
                        
                    curr_text.extend(text_append)
                    curr_image.append(None)
                    
                elif text == None and image != None:
                    if current_length + self.image_seq_length + 2 + self.after_image_extra_tokens > self.max_length: # TODO: Make sure there is text remaining from current sample
                        current_length = self.max_length
                        
                    else: # So this includes that EOS case...
                        curr_image.extend([None, self.image_buffer.pop(0), None])
                        curr_text.extend([self.image_start_text, self.text_buffer.pop(0), self.image_end_text])
                        current_length += self.image_seq_length + 2
                else:
                    raise ValueError("Text and image are both none or both not None in the same sample")

                if current_length == self.max_length:
                    np_text = np.array(curr_text)
                    text_ids = np_text[np_text != None]
                    text_tokens = text_ids[:-1]
                    text_positions = torch.from_numpy(np.where(np_text != None)[0])
                    
                    images = list(filter(lambda a: a != None, curr_image))
                    image_positions = torch.from_numpy(np.where(np_text == None)[0])
                    labels = text_ids[1:]

                    multimodal_position_ids = torch.nn.utils.rnn.pad_sequence([text_positions, image_positions], batch_first = True, padding_value = -1)

                    print("text_id", text_tokens)
                    print("text_positions", text_positions)
                    print("image_positions", image_positions)
                    print("multimodal_position_ids", multimodal_position_ids)
                    print("labels", labels)
                    print("images", images)

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


'''
python create_dataset.py \
  --path /p/fastdata/mmlaion/hummingbird/streaming/arxiv.jsonl \
  --out_root /p/fastdata/mmlaion/hummingbird/streaming/text/train --split train \
  --concat_tokens 2048 --tokenizer EleutherAI/gpt-neox-20b --eos_text '<|endoftext|>' \
  --compression zstd
'''
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
    parser.add_argument('--split', type=str, default='train')

    parser.add_argument('--tokenizer', type=str, required=False, default=None)
    parser.add_argument('--bos_text', type=str, required=False, default=None)
    parser.add_argument('--eos_text', type=str, required=False, default=None)
    parser.add_argument('--no_wrap', default=False, action='store_true')

    parsed = parser.parse_args()

    if os.path.isdir(parsed.out_root) and len(
            set(os.listdir(parsed.out_root)).intersection(set(
                parsed.split))) > 0:
        raise ValueError(
            f'--out_root={parsed.out_root} contains {os.listdir(parsed.out_root)} which cannot overlap with the requested splits {parsed.splits}.'
        )

    # Make sure we have needed concat options
    if (parsed.concat_tokens is not None and
            isinstance(parsed.concat_tokens, int) and parsed.tokenizer is None):
        parser.error(
            'When setting --concat_tokens, you must specify a --tokenizer')

    # now that we have validated them, change BOS/EOS to strings
    if parsed.bos_text is None:
        parsed.bos_text = ''
    if parsed.eos_text is None:
        parsed.eos_text = ''
    return parsed


class ImageCaptionDataset(IterableDataset):
    def __init__(self, path):
        fpath = path + "/{00000..00001}.tar"
        self.dataset = wds.WebDataset(fpath).decode("pilrgb").rename(image="jpg;png;jpeg;webp", text="txt").to_tuple("image", "text") 

    def __iter__(self):
        for image, text in self.dataset:
            sample = {
                "images": [None, image],
                "text": [text, None]
            }
            yield sample

def build_image_caption_dataset(
    path: str,
    split: str,
    mode: ConcatMode,
    max_length: Optional[int] = None,
    bos_text: str = '',
    eos_text: str = '',
    no_wrap: bool = False,
    tokenizer: PreTrainedTokenizerBase = None,
) -> IterableDataset:
    """Build an IterableDataset over the HF C4 or pile source data.

    Args:
        dataset_name (str): Dataset name
        split (str): Split name.
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

    dataset = ImageCaptionDataset(path)

    if mode == ConcatMode.NO_CONCAT:
        dataset = NoConcatDataset(dataset)
    else:
        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise ValueError(
                f'{tokenizer} must be of type PreTrainedTokenizerBase')
        if max_length is None:
            raise ValueError(f'max_length must be set.')
        if bos_text + eos_text == '':
            test_tokens = tokenizer('test')
            if test_tokens['input_ids'][
                    0] != tokenizer.bos_token_id and test_tokens['input_ids'][
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
            image_seq_length=10,
            bos_text=bos_text,
            eos_text=eos_text,
            image_start_text='hello',
            image_end_text='world',
            no_wrap=no_wrap
        )
    return dataset

def generate_samples(
        loader: DataLoader,
        truncate_num_samples: Optional[int] = None
) -> Iterable[Dict[str, bytes]]:
    """Generator over samples of a dataloader.

    Args:
       loader (DataLoader): A dataloader emitting batches like {key: [sample0_bytes, sample1_bytes, sample2_bytes, ...]}
       truncate_num_samples (Optional[int]): An optional # of samples to stop at.

    Yields:
        Sample dicts.
    """
    n_samples = 0
    for batch in loader:
        keys = list(batch.keys())
        print(keys)
        current_bs = len(batch[keys[0]])
        for idx in range(current_bs):
            if truncate_num_samples is not None and n_samples == truncate_num_samples:
                return
            n_samples += 1
            yield {k: v[idx] for k, v in batch.items()}


def main(args: Namespace) -> None:
    """Main: create C4/pile streaming dataset.

    Args:
        args (Namespace): Commandline arguments.
    """
    if args.concat_tokens is not None:
        mode = ConcatMode.CONCAT_TOKENS
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        # we will enforce length, so suppress warnings about sequences too long for the model
        tokenizer.model_max_length = int(1e30)
        columns = {'tokens': 'bytes', 'images': 'pickleencoding', 'multimodal_position_ids': 'ndarray', 'labels': 'ndarray'}
    else:
        mode = ConcatMode.NO_CONCAT
        tokenizer = None
        columns = {'text': 'str', 'images': 'pickleencoding'}

    print('here')

    # Write samples
    print(f'Converting to MDS format...')
    print(
        f'Note that the progress bar is based on the dataset length before tokenization.'
    )
    print(f'It will finish at a value below 100% if tokenizing')
    with MDSWriter(columns=columns,
                   out=os.path.join(args.out_root),
                   compression=args.compression, size_limit=5.12e+8) as out:
        for i in range(1):
            # Get samples
            dataset = build_image_caption_dataset(path='/p/fastdata/mmlaion/laion-400m/LAION-400m-webdataset/data',
                                    split=args.split,
                                    mode=mode,
                                    max_length=args.concat_tokens,
                                    bos_text=args.bos_text,
                                    eos_text=args.eos_text,
                                    no_wrap=args.no_wrap,
                                    tokenizer=tokenizer)
            for sample in tqdm(dataset):
                out.write(sample)


if __name__ == '__main__':
    main(parse_args())


'''
python create_dataset.py   --path /p/fastdata/mmlaion/hummingbird/streaming/arxiv.jsonl   --out_root /p/fastdata/mmlaion/hummingbird/streaming/interleaved/train --split train   --concat_tokens 2048 --tokenizer EleutherAI/gpt-neox-20b --eos_text '<|endoftext|>'   --compression zstd'''