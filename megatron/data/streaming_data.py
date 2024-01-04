import os
import shutil
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from streaming import MDSWriter, StreamingDataset


# the location of the "remote" streaming dataset (`sds`).
# Upload `out_root` to your cloud storage provider of choice. If `out_root` is a cloud provider
# path, shard files are automatically uploaded.
out_root = "/p/project/ccstdl/gupta6/multimodal/data/synthstreaming2"
out_train = "/p/project/ccstdl/gupta6/multimodal/data/synthstreaming2/train"
out_val = "/p/project/ccstdl/gupta6/multimodal/data/synthstreaming2/val"
out_test = "/p/project/ccstdl/gupta6/multimodal/data/synthstreaming2/test"

# the location to download the streaming dataset during training
local = '/p/project/ccstdl/gupta6/multimodal/data/streamingcache'
local_train = '/p/project/ccstdl/gupta6/multimodal/data/streamingcache/train'
local_val = '/p/project/ccstdl/gupta6/multimodal/data/streamingcache/val'
local_test = '/p/project/ccstdl/gupta6/multimodal/data/streamingcache/test'

# toggle shuffling in dataloader
shuffle_train = True
shuffle_val = False

# training batch size
batch_size = 512

# upload location for the dataset splits (change this if you want to upload to a different location, for example, AWS S3 bucket location)
upload_location = None

if upload_location is None:
    upload_train_location = None
    upload_val_location = None
    upload_test_location = None
else:
    upload_train_location = os.path.join(upload_location, 'train')
    upload_val_location = os.path.join(upload_location, 'val')
    upload_test_location = os.path.join(upload_location, 'test')

# # Word representation of a number
ones = ('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen ' +
        'fifteen sixteen seventeen eighteen nineteen').split()

tens = 'twenty thirty forty fifty sixty seventy eighty ninety'.split()


def say(i: int) -> List[str]:
    """Get the word form of a number.

    Args:
        i (int): The number.

    Returns:
        List[str]: The number in word form.
    """
    if i < 0:
        return ['negative'] + say(-i)
    elif i <= 19:
        return [ones[i]]
    elif i < 100:
        return [tens[i // 10 - 2]] + ([ones[i % 10]] if i % 10 else [])
    elif i < 1_000:
        return [ones[i // 100], 'hundred'] + (say(i % 100) if i % 100 else [])
    elif i < 1_000_000:
        return say(i // 1_000) + ['thousand'] + (say(i % 1_000) if i % 1_000 else [])
    elif i < 1_000_000_000:
        return say(i // 1_000_000) + ['million'] + (say(i % 1_000_000) if i % 1_000_000 else [])
    else:
        assert False

def get_numbers(num_train: int, num_val: int, num_test: int) -> Tuple[List[int], List[int]]:
    """Get two non-overlapping splits of a sequential random numbers.

    The train sample indices goes from [0, num_train] and val sample indices goes
    from [num_train, num_val].

    Args:
        num_train (int): Number of training samples.
        num_val (int): Number of validation samples.

    Returns:
        Tuple[List[int], List[int]]: The two generated splits.
    """
    total = num_train + num_val + num_test
    numbers = []
    bar = tqdm(total=total, leave=False)
    i = 0
    while i < total:
        was = len(numbers)
        sign = (np.random.random() < 0.8) * 2 - 1
        numbers.append(sign * i)
        bar.update(len(numbers) - was)
        i += 1
    return numbers[:num_train], numbers[num_train: num_train + num_val], numbers[num_train + num_val:]

def generate_samples(numbers: List[int]) -> List[Dict[str, Any]]:
    """Generate samples from a list of numbers.

    Args:
        numbers (List[int]): The numbers.

    Returns:
        List[Dict[str, Any]]: The corresponding samples.
    """
    samples = []
    for num in numbers:
        words = ' '.join(say(num))
        sample = {'number': num, 'text': words}
        samples.append(sample)
    return samples


def get_dataset(num_train: int, num_val: int, num_test: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generate a number-saying dataset of the given size.

    Args:
        num_train (int): Number of training samples.
        num_val (int): Number of validation samples.

    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: The two generated splits.
    """
    train_nums, val_nums, test_nums = get_numbers(num_train, num_val, num_test)
    train_samples = generate_samples(train_nums)
    val_samples = generate_samples(val_nums)
    test_samples = generate_samples(test_nums)
    return train_samples, val_samples, test_samples

# # Number of training and validation samples
num_train_samples = 10000 # 10k samples
num_val_samples = 2000    # 2k samples
num_test_samples = 2000   # 2k samples

# # Create the samples.
print(f'Generating synthetic dataset ({num_train_samples} train, {num_val_samples} val)...')
train_samples, val_samples, test_samples = get_dataset(num_train_samples, num_val_samples, num_test_samples)
splits = [
    ('train', train_samples),
    ('val', val_samples),
    ('test', test_samples)
]

print(f'Train sample: {train_samples[0]}')
print(f'Val sample: {val_samples[0]}')
print(f'Test sample: {test_samples[0]}')


# Mapping of sample keyword with their data type
columns = {
    'number': 'int',
    'text': 'str',
}

# Compression algorithm to use for dataset
compression = 'zstd:12'

# Hashing algorithm to use for dataset
hashes = ['sha1', 'xxh3_64']

# shard size limit, in bytes
size_limit = 1 << 16  # Override to a small number for more shards.

print(f'Saving dataset (to {out_root})...')
for split, samples in splits:
    print(f'* {split}')
    dirname = os.path.join(out_root, split)
    with MDSWriter(out=dirname, columns=columns, compression=compression,
                   hashes=hashes, size_limit=size_limit) as out:
        for sample in tqdm(samples, leave=False):
            out.write(sample)


# # wait for 15 seconds
# print('Waiting for 15 seconds...')
# import time
# time.sleep(15)

# remote_train = upload_train_location or out_train # replace this with your URL for cloud streaming
# remote_val  = upload_val_location or out_val

# # Load the samples back.
# print('Walking the dataset:')

# print(f'verifying samples for train split')
# train_dataset = StreamingDataset(remote=upload_location or out_root, local=local, split='train', shuffle=False)

# # for old, new in tqdm(zip(train_samples, train_dataset), total=len(train_samples), leave=False):
# #     print(old, new)
#     # assert old == new

# # print(f'verifying samples for val split')
# val_dataset = StreamingDataset(remote=upload_location or out_root, local=local, split='val', shuffle=False)
# # for old, new in tqdm(zip(val_samples, val_dataset), total=len(val_samples), leave=False):
# #     # assert old == new
# #     print(old, new)

# # # Fetch the 10th sample and print it on a console
# # print(f'Sample 10: {train_dataset[10]}')

# # # Fetch multiple samples
# # indices = [-1, 30, [12, -14], slice(-1, -10, -2), np.array([10, -20])]
# # for indx in indices:
# #     print(f'Sample {indx}: {train_dataset[indx]}')

# # # Get the total number of samples
# # print(f'Total number of samples: {train_dataset.num_samples}')

# # # Get the number of shard files
# # print(f'Total number of shards: {len(train_dataset.shards)}')

# # # Get the number of samples inside each shard files.
# # # Number of samples in each shard can vary based on each sample size.
# # print(f'Number of samples inside each shards: {train_dataset.samples_per_shard}')

# train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
# val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# # Instantiate a streaming Dataset class without a `batch_size` parameter
# # dataset_without_bs = StreamingDataset(remote=remote_train, local=os.path.join(local_train, 'wo_bs'), shuffle=shuffle_train)
# # dataloader_ds_wo_bs = DataLoader(dataset_without_bs, batch_size=batch_size, num_workers=8, drop_last=True)

# # # Instantiate a streaming Dataset class with a `batch_size` parameter
# # dataset_with_bs = StreamingDataset(remote=remote_train, local=os.path.join(local_train, 'w_bs'), shuffle=shuffle_train, batch_size=batch_size)
# # dataloader_ds_with_bs = DataLoader(dataset_with_bs, batch_size=batch_size, num_workers=8, drop_last=True)

# total_samples = 0
# for idx, batch in enumerate(train_dataloader):
#     print(batch)
#     total_samples += len(batch["number"])
# # print(f'Total number of samples processed by the dataloader is {total_samples} out of {num_train_samples}')

# total_samples = 0
# for idx, batch in enumerate(val_dataloader):
#     total_samples += len(batch["number"])
# # print(f'Total number of samples processed by the dataloader is {total_samples} out of {num_train_samples}')