import ast
import json
import logging
import math
import os
import random
import sys
from dataclasses import dataclass
from multiprocessing import Value

import braceexpand
import numpy as np
import pandas as pd
import torch
import webdataset as wds
from PIL import Image
from torch.utils.data import  DataLoader,  IterableDataset, get_worker_info
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample

from megatron import mpu

class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        expanded_urls=[url for url in expanded_urls if os.path.exists(url)]
        ### go save existed url
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist),\
            f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_urls=[url for url in expanded_urls if os.path.exists(url)] 
            ### go save existed url
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights

def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches

def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True

def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000

class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        weights=None,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(self.weights),\
                f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        epoch = self.epoch.get_value()
        print(f'start at {epoch} sub epoch')
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])

class SimpleShardList2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(self, 
                 urls,
                 epoch=-1,
                 seed=0,
                 num_sub_epochs=None
                 ):
        """Iterate through the list of shards.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls,_ = expand_urls(urls)
        self.urls = urls
        assert isinstance(self.urls[0], str)
        self.seed = seed
        self.epoch = epoch
        self.num_sub_epochs = num_sub_epochs
        print(f'total urls shards number is {len(urls)}')

    def __len__(self):
        return len(self.urls)

    def __iter__(self):
        """Return an iterator over the shards."""
        urls = self.urls.copy()
        epoch = self.epoch.get_value()
        if self.num_sub_epochs is None:
            seed = self.seed + epoch
        else:
            seed = self.seed + (epoch // self.num_sub_epochs)
        random.Random(self.seed).shuffle(urls)
        
        assert(
            len(urls) >= self.num_sub_epochs
        ),f'shards number is {len(urls)}, smaller than num_sub_epochs {self.num_sub_epochs}'\
            'please increase train shareds num or reduce num_sub_epochs'\
            'num_sub_epoch = train_num_samples / global_batch_size / checkpoint_factor'
        
        if self.num_sub_epochs is not None:
            urls = urls[epoch % self.num_sub_epochs::self.num_sub_epochs]

        print(f'start at {epoch} sub epoch with number {self.num_sub_epochs} with {len(urls)} shards and total shards{len(self.urls.copy())}')
         
        for url in urls:
            print(f'get url {url} now from {os.environ["RANK"]}')
            yield dict(url=url)

def image_text_dict_collation_fn(samples):
    """Customize collation_fn to generate dict batch """
    assert isinstance(samples[0], (list, tuple)), type(samples[0])
    batched = list(zip(*samples))
    result = dict()
    import torch
    import numpy as np
    for b in batched:
        b = torch.stack(list(b))
        if b.dim()>=3: # dim means image
            result['img']=b
        else:
            result['text']=b
    
    return result

def get_data_parallel_info():    
    if False:
        rank = mpu.get_data_parallel_rank()
        world_size = mpu.get_data_parallel_world_size()
    else:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    return rank, world_size

def get_dp_world_size_wrap(original_function):
    
    def wrapper(*args, **kwargs):
        dp_rank, dp_world_size = get_data_parallel_info()
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        os.environ['RANK'] = str(dp_rank) 
        os.environ['WORLD_SIZE'] = str(dp_world_size) 
        # change the worldsize to data-parallel world-size for group by node function. 
        result = original_function(*args, **kwargs)
        # change back to aviod potenital issues caused by wrong WORLD_SIZE
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        return result
    
    return wrapper


def get_wds_data(args, is_train, floor=False):
    input_shards = args.train_data_paths if is_train else args.valid_data_paths
    rank, world_size = get_data_parallel_info() # get world_size as data-parallel-world-size
    resampled = getattr(args, 'dataset_resampled', False) and is_train
    round_fn = math.floor if floor else math.ceil
    assert input_shards is not None
    
    # get the num_samples info
    if is_train:
        if args.train_num_samples is not None:
            num_samples = args.train_num_samples
        else:
            raise RuntimeError(
                'Currently, the number of dataset samples must be specified for the training dataset. '
                'Please specify it via `--train-num-samples` if no dataset length info is present.')
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        num_samples = args.val_num_samples or 0 

    if resampled:
        pipeline = [ResampledShards2(
            input_shards,
            weights=args.train_data_upsampling_factors,
            deterministic=True,
            epoch=args.shared_epoch,
        )]
    else:
        if is_train:
            global_batch_size=world_size*args.batch_size
            num_batches = round_fn(args.train_num_samples / global_batch_size)
            num_workers = max(1, args.num_workers)
            num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
            num_batches = num_worker_batches * num_workers
            train_one_epoch_iters = num_batches 
            num_sub_epochs = round_fn(train_one_epoch_iters / args.checkpoint_factor)
            assert (
                args.checkpoint_scale == 'linear'
                    ), f'webdataset only works with linear checkpoint saving way'
        else:
            num_sub_epochs = 1 # val don't need to recover from iter
        pipeline = [SimpleShardList2(input_shards, epoch=args.shared_epoch, num_sub_epochs=num_sub_epochs)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    ### build preprocess_img and preprocess_text from args
    from .transforms import get_clip_transforms
    preprocess_img = get_clip_transforms(image_size=args.image_size)
    
    assert (
        args.tokenizer.name in ['HFGPT2Tokenizer','HFGPT2TokenizerFast']
        ), f"Webdataset only support HFGPT2Tokenizer or HFGPT2TokenizerFast"
    
    tokenize = args.tokenizer.tokenize
    seq_length = args.seq_length
    
    pipeline.extend([
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp", text="txt"),
        wds.map_dict(image=preprocess_img,  text=lambda text: tokenize(text,seq_length)[0]),
        wds.to_tuple("image", "text"),
        wds.batched(args.batch_size, collation_fn=image_text_dict_collation_fn, partial=not is_train)
    ])

    # wrap with data-parallel-world-size
    dataset = get_dp_world_size_wrap(wds.DataPipeline)(*pipeline)

    if is_train:
        if not resampled:
            num_shards = len(expand_urls(input_shards)[0])
            assert num_shards/num_sub_epochs >= args.num_workers * world_size, 'number of shards of subset must be >= total workers'
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=not (args.num_workers == 0),  # set persistent_workers to false if num_workers is 0
    )
    
    return dataloader
