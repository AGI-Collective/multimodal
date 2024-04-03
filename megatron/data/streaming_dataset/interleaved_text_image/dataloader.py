# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Build a StreamingTextDataset dataset and dataloader for training."""

import os
from itertools import islice
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

from PIL import Image
import pickle
import numpy as np
import torch
import transformers
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from streaming import Stream, StreamingDataset, StreamingDataLoader
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase
import argparse
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers import AutoImageProcessor, AutoModel

from streaming.base.format.mds.encodings import Encoding, _encodings
from einops import rearrange

# from megatron.data.streaming_dataset.interleaved_text_image.create_interleaved_dataset import simple_encoding, ListPIL, PickleEncoding

from megatron.data.streaming_dataset.interleaved_text_image.create_unified_interleaved_dataset import ListPIL
# _encodings['pickleencoding'] = PickleEncoding
_encodings['listpil'] = ListPIL
# _encodings['simple_encoding'] = simple_encoding
from megatron.tokenizer.tokenizer import build_tokenizer

class StreamingInterleavedDataset(StreamingDataset):
    """Generic text dataset using MosaicML's StreamingDataset.

    Args:
        tokenizer (Tokenizer): HuggingFace tokenizer to
            tokenize samples.
        max_seq_length (int): The max sequence length of each sample.
        streams (Sequence[Stream], optional): One or more Streams to stream/cache samples from,
            which may be upsampled or downsampled. StreamingDataset uses either ``streams`` or
            ``remote``/``local``. Defaults to ``None``.
        remote (str, optional): Remote path or directory to download the dataset from. If ``None``,
            its data must exist locally. StreamingDataset uses either ``streams`` or
            ``remote``/``local``. Defaults to ``None``.
        local (str, optional): Local working directory to download shards to. This is where shards
            are cached while they are being used. Uses a temp directory if not set.
            StreamingDataset uses either ``streams`` or ``remote``/``local``. Defaults to ``None``.
        split (str, optional): Which dataset split to use, if any. If provided, we stream from/to
            the ``split`` subdirs of  ``remote`` and ``local``. Defaults to ``None``.
        download_retry (int): Number of download re-attempts before giving up. Defaults to ``2``.
        download_timeout (float): Number of seconds to wait for a shard to download before raising
            an exception. Defaults to ``60``.
        validate_hash (str, optional): Optional hash or checksum algorithm to use to validate
            shards. Defaults to ``None``.
        keep_zip (bool): Whether to keep or delete the compressed form when decompressing
            downloaded shards. If ``False``, keep iff remote is local or no remote. Defaults to
            `False``.
        epoch_size (int, optional): Number of samples to draw per epoch balanced across all
            streams. If ``None``, takes its value from the total number of underlying samples.
            Provide this field if you are weighting streams relatively to target a larger or
            smaller epoch size. Defaults to ``None``.
        predownload (int, optional): Target number of samples ahead to download the shards of while
            iterating. Defaults to ``100_000``.
        cache_limit (Union[int, str], optional) - Maximum size in bytes of this StreamingDataset's
            shard cache. Before downloading a shard, the least recently used resident shard(s) may
            be evicted (deleted from the local cache) in order to stay under the limit. Set to None
            to disable shard eviction. Supports integer bytes as well as string human-readable
            bytes (e.g., 100b, 64kb, 77mb, and so on). Defaults to None.
        partition_algo (str): Which partitioning algorithm to use. Defaults to ``orig``.
        num_canonical_nodes (int, optional): Canonical number of nodes for shuffling with
            resumption. Defaults to ``None``, which is interpreted as the number of nodes of the
            initial run.
        batch_size (int, optional): Batch size of its DataLoader, which affects how the dataset is
            partitioned over the workers. Defaults to ``None``.
        shuffle (bool): Whether to iterate over the samples in randomized order. Defaults to
            ``False``.
        shuffle_algo (str): Which shuffling algorithm to use. Defaults to ``py1b``.
        shuffle_seed (int): Seed for Deterministic data shuffling. Defaults to ``9176``.
        shuffle_block_size (int): Unit of shuffle. Defaults to ``1 << 18``.
    """

    def __init__(self,
                 tokenizer,
                 max_seq_length: int,
                 streams: Optional[Sequence[Stream]] = None,
                 remote: Optional[str] = None,
                 local: Optional[str] = None,
                 split: Optional[str] = None,
                 download_retry: int = 2,
                 download_timeout: float = 60,
                 validate_hash: Optional[str] = None,
                 keep_zip: bool = False,
                 epoch_size: Optional[int] = None,
                 predownload: int = 100_000,
                 cache_limit: Optional[Union[int, str]] = None,
                 partition_algo: str = 'orig',
                 num_canonical_nodes: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 shuffle: bool = False,
                 shuffle_algo: str = 'py1b',
                 shuffle_seed: int = 9176,
                 shuffle_block_size: int = 1 << 18,
                 batching_method: str = 'random',
                 vision_pad_id: int = -100,
                 **kwargs: Any):

        group_method = kwargs.pop('group_method', None)
        if group_method is not None:
            raise NotImplementedError(
                'group_method is deprecated and has been removed.\nTo ' +
                'concatenate, use the --concat_tokens ' +
                'argument when creating your MDS dataset with concat_c4.py')

        if len(kwargs) > 0:
            raise ValueError(
                f'StreamingTextDataset() got an unexpected keyword argument: {kwargs}'
            )

        if local is not None and (remote is None or (local == remote)):
            if os.path.isdir(local):
                contents = set(os.listdir(local))
                if split not in contents:
                    raise ValueError(
                        f'local directory {local} does not contain split {split}'
                    )

        # Build Dataset
        super().__init__(
            streams=streams,
            remote=remote,
            local=local,
            split=split,
            download_retry=download_retry,
            download_timeout=download_timeout,
            validate_hash=validate_hash,
            keep_zip=keep_zip,
            epoch_size=epoch_size,
            predownload=predownload,
            cache_limit=cache_limit,
            partition_algo=partition_algo,
            num_canonical_nodes=num_canonical_nodes,
            batch_size=batch_size,
            shuffle=shuffle,
            shuffle_algo=shuffle_algo,
            shuffle_seed=shuffle_seed,
            shuffle_block_size=shuffle_block_size,
            batching_method=batching_method,
        )
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.vision_pad_id = vision_pad_id

    # How to tokenize a text sample to a token sample
    def _tokenize(self, text_sample: Mapping):
        if self.tokenizer._pad_token is None:
            # Some tokenizers (e.g. GPT2 tokenizer) have no padding token which causes bugs
            raise RuntimeError(
                'If tokenizing on-the-fly, tokenizer must have a pad_token_id')

        return self.tokenizer(text_sample['text'],
                              truncation=True,
                              padding='max_length',
                              max_length=self.max_seq_length)

    def _read_binary_tokenized_sample(self, sample: Dict[str, Any]):
        return torch.from_numpy(
            np.frombuffer(sample['tokens'],
                          dtype=np.int64)[:self.max_seq_length].copy())
    
    def _read_binary_data(self, sample):
        return torch.from_numpy(
            np.frombuffer(sample,
                          dtype=np.int64).copy())

    # How to process a sample
    def __getitem__(self, idx: int):
        sample = super().__getitem__(idx)
        if 'text' in sample:
            token_sample = self._tokenize(sample)
        elif 'tokens' in sample:
            token_sample = self._read_binary_tokenized_sample(sample)
        else:
            raise RuntimeError(
                'StreamingTextDataset needs samples to have a `text` or `tokens` column'
            )
        images = list(map(lambda x: np.array(x), sample.get('images', None)))
        if images != []:
            images = np.stack(images)
        else:
            images = np.array([])
        vision_input = images.reshape(-1, 224, 224, 3)    
        is_vision_empty = vision_input.shape[0] == 0
        if is_vision_empty:
            vision_input = np.zeros((1, 224, 224, 3), dtype=np.uint8)
        
        vision_input = torch.from_numpy(vision_input).to(torch.int64)
        vision_input = vision_input.unsqueeze(1) # TODO: Fix for num_frames > 1
        vision_input = rearrange(vision_input, "t f h w c -> t f c h w")
        
        if is_vision_empty:
            vision_input = torch.ones_like(vision_input) * self.vision_pad_id
        multimodal_position_ids = torch.from_numpy(np.frombuffer(sample.get('multimodal_position_ids', None), dtype=np.int64).copy()).reshape(2, -1)
        labels = torch.from_numpy(np.frombuffer(sample.get('labels', None), dtype=np.int64).copy()).reshape(2, -1) 
        return (token_sample, vision_input, multimodal_position_ids, labels)


# Multimodal Collate Function
class MultimodalCollateWrapper:
    def __init__(self, text_collator, vision_collator, video_collator, audio_collator, multimodal_position_ids_collator, label_collator) -> None:
        self.text_collator = text_collator
        self.vision_collator = vision_collator
        # self.video_collator = video_collator
        # self.audio_collator = audio_collator
        self.multimodal_position_ids_collator = multimodal_position_ids_collator
        self.label_collator = label_collator
    
    def __call__(self, examples: List[Any]) -> Dict[str, torch.Tensor]:
        # Convert examples (list of tuples) to a list of lists (one list per modality)
        examples = list(zip(*examples))
        batch = self.text_collator(examples[0])
        batch['vision_input'] = self.vision_collator(examples[1])
        # batch['video'] = self.video_collator(examples[2])
        # batch['audio'] = self.audio_collator(examples[3])
        batch['multimodal_position_ids'] = self.multimodal_position_ids_collator(examples[2])
        batch['labels'] = self.label_collator(examples[3])
        return batch

class TextNeoXCollateWrapper:

    def __init__(self, collator) -> None:
        self.collator = collator
    
    def __call__(self, examples: List[Any]) -> Dict[str, torch.Tensor]:
        batch = self.collator(examples)
        batch['text_input'] = batch['input_ids'] # TODO Delete input_ids to save comm overhead and space
        return batch

class PaddedCollateWrapper:
    def __init__(self, pad_token_id, take_transpose=False) -> None:
        self.pad_token_id = pad_token_id
        self.take_transpose = take_transpose

    def __call__(self, examples: List[Any]) -> Dict[str, torch.Tensor]:
        if self.take_transpose:
            # Apply transpose to each example in batch parallely using map function
            examples = list(map(lambda x: x.transpose(0, 1), examples))
        batch = torch.nn.utils.rnn.pad_sequence(examples, batch_first=True, padding_value=self.pad_token_id)

        if self.take_transpose:
            batch = batch.transpose(1, 2)
        return batch
    
def build_interleaved_dataloader(
    cfg: DictConfig,
    tokenizer,
    device_batch_size: int,
):
    assert cfg.name == 'text', f'Tried to build text dataloader with cfg.name={cfg.name}'
    if cfg.dataset.get('group_method', None) is not None:
        raise NotImplementedError(
            'group_method is deprecated and has been removed.\nTo ' +
            'concatenate, use the --concat_tokens ' +
            'argument when creating your MDS dataset with convert_dataset_hf.py'
        )

    # get kwargs
    streams_dict = cfg.dataset.pop('streams', None)
    mlm_probability = cfg.dataset.pop('mlm_probability', None)
    position_pad_id = cfg.dataset.pop('position_pad_id', None)
    pad_token_id = tokenizer.pad_token_id
    vision_pad_id = cfg.dataset.pop('vision_pad_id', None)

    # build streams
    streams = None
    if streams_dict is not None:
        streams = []
        for _, stream in streams_dict.items():
            # stream is the streams kwargs
            # fwd all kwargs with **stream allows streaming to check args
            streams.append(Stream(**stream))

    # build dataset potentially with streams
    dataset = StreamingInterleavedDataset(
        tokenizer=tokenizer,
        streams=streams,
        vision_pad_id=vision_pad_id,
        batch_size=device_batch_size,
        epoch_size=cfg.get('epoch_size', None),
        **cfg.dataset,
    )

    text_collate_fn = transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=mlm_probability is not None,
        mlm_probability=mlm_probability)

    text_collate_fn = TextNeoXCollateWrapper(text_collate_fn)
    
    vision_collate_fn = PaddedCollateWrapper(pad_token_id=vision_pad_id) # Each sample: (timesteps, num_vision, H, W, C)

    multimodal_position_ids_collate_fn = PaddedCollateWrapper(pad_token_id=position_pad_id, take_transpose=True) # Each sample: (num_modalities, max_seq_length)
    
    label_collate_fn = PaddedCollateWrapper(pad_token_id=pad_token_id, take_transpose=True) # Each sample: (num_modalities, max_seq_length)

    collate_fn = MultimodalCollateWrapper(text_collator=text_collate_fn, 
                                          vision_collator=vision_collate_fn, 
                                          video_collator=None, 
                                          audio_collator=None, 
                                          multimodal_position_ids_collator=multimodal_position_ids_collate_fn, 
                                          label_collator=label_collate_fn)
    
    return StreamingDataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=device_batch_size,
        drop_last=cfg.drop_last,
        num_workers=cfg.num_workers,
        pin_memory=cfg.get('pin_memory', True),
        prefetch_factor=cfg.get('prefetch_factor', 2),
        persistent_workers=cfg.get('persistent_workers', True),
        timeout=cfg.get('timeout', 0),
    )


# Helpful to test if your dataloader is working locally
# Run `python data.py  --local_path [local] [--remote_path remote, optional]` and verify that batches are printed out
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--local_path',
                        type=str,
                        required=True,
                        help='the path to the local copy of the dataset')
    parser.add_argument(
        '--remote_path',
        type=str,
        default=None,
        help='the path to the remote copy to stream from (optional)')
    parser.add_argument('--split',
                        type=str,
                        default='validation',
                        help='which split of the dataset to use')
    parser.add_argument('--max_seq_length',
                        type=int,
                        default=2048,
                        help='max sequence length to test')

    parser.add_argument('--tokenizer_type', type=str, required=False, default=None)
    parser.add_argument('--vocab_file', type=str, required=False, default=None)
    parser.add_argument('--merge_file', type=str, required=False, default=None)

    args = parser.parse_args()

    if args.remote_path is not None:
        print(
            f'Reading {args.split} split from {args.local_path} <- streamed from <- {args.remote_path}'
        )
    else:
        print(f'Reading {args.split} split from {args.local_path}')

    cfg = {
        'name': 'text',
        'dataset': {
            'local': args.local_path,
            'remote': None,
            'split': args.split,
            'shuffle': False,
            'max_seq_length': args.max_seq_length,
            'keep_zip': True,  # in case we need compressed files after testing
            'position_pad_id': -1,
            'vision_pad_id': 0, 
        },
        'drop_last': True,
        'num_workers': 5,
    }
    cfg = om.create(cfg)

    device_batch_size = 2

    tokenizer_args = {
        "tokenizer_type": args.tokenizer_type,
        "vocab_file": args.vocab_file,
        "rank": 0,
        "model_parallel_size": 1,
        "make_vocab_size_divisible_by": 128,
    }
    class Config:
        def __init__(self, dictionary):
            for key, value in dictionary.items():
                setattr(self, key, value)
    
    tokenizer_args = Config(tokenizer_args)
    # tokenizer_config = om.create(tokenizer_args)
    tokenizer = build_tokenizer(tokenizer_args)
    
    loader = build_interleaved_dataloader(cfg, tokenizer, device_batch_size)
    print("I am ready")
    for sample in loader:
        print(sample)
    for batch_ix, batch in enumerate(loader):
        print('\n')
        print('#' * 20, f'Batch {batch_ix}', '#' * 20)
        for k, v in batch.items():
            print(k, v.shape, v.dtype)
        for sample_ix, token_sample in enumerate(batch['input_ids']):
            print('-' * 20, f' Sample {sample_ix} ', '-' * 20)
            print(tokenizer.decode(token_sample))



'''
python /p/project/ccstdl/gupta6/multimodal/megatron/data/streaming_dataset/interleaved_text_image/dataloader.py --tokenizer_type HFTokenizer --vocab_file /p/project/ccstdl/gupta6/multimodal/20B_tokenizer.json --local_path /p/fastdata/mmlaion/hummingbird/hummingbird_dataset --split text_train_chunk1
'''