import argparse
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import random

import random
from torch.utils.data import Subset
import collections
from collections.abc import Sequence
from functools import partial
import os
import warnings
import math
from monai.data import *
import itertools
from scipy import ndimage
import pydicom
import medpy.io as medio
import glob
from sklearn.model_selection import train_test_split
from monai.transforms import *


warnings.filterwarnings("ignore")



def get_cardiacMRI(args):
    base_dir = '../dataset/CardiacMRI/'

    data_dir1 = base_dir + 'UKBB/'
    data_list1 = get_volume_UKBB(data_dir1)
    train_files1, test_files1 = dataset_split(data_list1, args.seed)
    train_dicts1 = [{"image": image_name} for image_name in train_files1]
    test_dicts1 = [{"image": image_name} for image_name in test_files1]

    data_dir2 = base_dir + 'MESA/'
    data_list2 = get_volume_MESA(data_dir2)
    train_files2, test_files2 = dataset_split(data_list2, args.seed)
    train_dicts2 = [{"image": image_name} for image_name in train_files2]
    test_dicts2 = [{"image": image_name} for image_name in test_files2]

    data_dir3 = base_dir + 'ACDC/database/'
    data_list3 = get_volume_ACDC(data_dir3)
    train_files3, test_files3 = dataset_split(data_list3, args.seed)
    train_dicts3 = [{"image": image_name} for image_name in train_files3]
    test_dicts3 = [{"image": image_name} for image_name in test_files3]

    data_dir4 = base_dir + 'MSCMR/image/'
    data_list4 = get_volume_MSCMR(data_dir4)
    train_files4, test_files4 = dataset_split(data_list4, args.seed)
    train_dicts4 = [{"image": image_name} for image_name in train_files4]
    test_dicts4 = [{"image": image_name} for image_name in test_files4]

    data_list = data_list1 + data_list2 + data_list3 + data_list4

    datasets = {
        1: {
            "data_name": "UKBB",
            "train_files": train_dicts1,
            "test_files": test_dicts1,
            "modality": "MRI",
        },
        2: {
            "data_name": "MESA",
            "train_files": train_dicts2,
            "test_files": test_dicts2,
            "modality": "MRI",
        },
        3: {
            "data_name": "ACDC",
            "train_files": train_dicts3,
            "test_files": test_dicts3,
            "modality": "MRI",
        },
        4: {
            "data_name": "MSCMR",
            "train_files": train_dicts4,
            "test_files": test_dicts4,
            "modality": "MRI",
        },
    }
    return datasets


def dataset_split(data_list, seed):
    train_files, test_files = train_test_split(data_list, test_size=0.2, random_state=seed)
    return train_files, test_files

def get_volume_UKBB(data_dir):
    volume_list0 = sorted(glob.glob(data_dir + 'UKB_0/*.nii.gz'))
    volume_list1 = sorted(glob.glob(data_dir + 'UKB_1/*.nii.gz'))
    return volume_list0 + volume_list1

def get_volume_MESA(data_dir):
    volume_list = sorted(glob.glob(data_dir + '*.nii.gz'))
    return volume_list

def get_volume_ACDC(data_dir):
    volume_list = sorted(glob.glob(data_dir + '/*/*/*_frame**.nii.gz'))
    volume_list = [f for f in volume_list if '_gt' not in f]
    return volume_list

def get_volume_MSCMR(data_dir):
    volume_list = sorted(glob.glob(data_dir + '*_frame**.nii.gz'))
    return volume_list

def get_data(datasets, dataset_name):
    train_files = {"MRI": []}
    test_files = {"MRI": []}

    def add_assigned_class_to_datalist(datalist, classname):
        for item in datalist:
            item["class"] = classname
        return datalist

    data_by_name  = {}
    for key, dataset in datasets.items():
        data_name = dataset["data_name"]
        data_by_name[data_name] = dataset
    dataset_single = data_by_name[dataset_name]

    train_files_i = dataset_single["train_files"]
    test_files_i = dataset_single["test_files"]
    print(f"{dataset_single['data_name']}: number of training data is {len(train_files_i)}.")
    print(f"{dataset_single['data_name']}: number of test data is {len(test_files_i)}.")

    modality = dataset_single["modality"]
    train_files[modality] += add_assigned_class_to_datalist(train_files_i, modality)
    test_files[modality] += add_assigned_class_to_datalist(test_files_i, modality)

    for modality in train_files.keys():
        print(f"Total number of training data for {modality} is {len(train_files[modality])}.")
        print(f"Total number of test data for {modality} is {len(test_files[modality])}.")
    return train_files["MRI"], test_files["MRI"]

def get_combined_data(datasets):
    train_files = {"MRI": []}
    test_files = {"MRI": []}
    def add_assigned_class_to_datalist(datalist, classname):
        for item in datalist:
            item["class"] = classname
        return datalist
    for _, dataset in datasets.items():
        train_files_i = dataset["train_files"]
        test_files_i = dataset["test_files"]
        modality = dataset["modality"]
        train_files[modality] += add_assigned_class_to_datalist(train_files_i, modality)
        test_files[modality] += add_assigned_class_to_datalist(test_files_i, modality)
    return train_files["MRI"], test_files["MRI"]


def get_transforms(args):
    common_transform = [
                LoadImaged(keys=['image'], image_only=True, allow_missing_keys=True),
                EnsureChannelFirstd(keys=['image'], allow_missing_keys=True),
                ScaleIntensityRangePercentilesd(keys=['image'], lower=0, upper=99.5, b_min=0, b_max=1),
                SpatialPadd(keys=['image'], spatial_size=args.cardiac_roi, mode='edge', allow_missing_keys=True),
            ]

    train_transform = Compose(
        common_transform
        + [
            RandSpatialCropd(keys=['image'], roi_size=args.cardiac_roi, allow_missing_keys=True, random_size=False, random_center=True),
            RandFlipd(keys=["image"], prob=0.3, spatial_axis=0, allow_missing_keys=True),
            RandFlipd(keys=["image"], prob=0.3, spatial_axis=1, allow_missing_keys=True),
            RandFlipd(keys=["image"], prob=0.3, spatial_axis=2, allow_missing_keys=True),
            RandRotate90d(keys=["image"], prob=0.3, max_k=3, allow_missing_keys=True),
            RandScaleIntensityd(keys=["image"], prob=0.2, factors=(0.9, 1.1), allow_missing_keys=True),
            RandShiftIntensityd(keys=["image"], prob=0.2, offsets=0.05, allow_missing_keys=True),
            Resized(keys=["image"], spatial_size=args.cardiac_size, mode="trilinear", align_corners=True, allow_missing_keys=True),
            ToTensord(keys=["image"], allow_missing_keys=True),
        ])

    test_transform = Compose(
        common_transform
        + [
            Resized(keys=["image"], spatial_size=args.cardiac_size, mode="trilinear", align_corners=True, allow_missing_keys=True),
            ToTensord(keys=["image"], allow_missing_keys=True),
        ])
    return train_transform, test_transform


def missing_sample(x_in, missing_length, seed):
    torch.manual_seed(seed)
    length = x_in.shape[-1]
    comp_list = list(range(length))
    start_idx = torch.randint(0, length - missing_length + 1, (1,)).item()
    indices = list(range(start_idx, start_idx + missing_length))
    missing_idx = sorted([comp_list[i] for i in indices])
    incomp_idx = list(set(comp_list) - set(missing_idx))
    x_missing = x_in[..., missing_idx]
    x_incomp = x_in[..., incomp_idx]
    missing_label = get_label(length, missing_idx)
    return x_incomp, x_missing, missing_label


def get_label(length, missing_idx):
    missing_label = np.zeros(length, dtype=int)
    for i in range(len(missing_idx)):
        missing_label[missing_idx[i]] = 1
    missing_label = torch.FloatTensor(missing_label)
    return missing_label


def train_collate_fn(batch, missing_length):
    data, idx = zip(*batch)
    x_incomp_list, x_missing_list = [], []
    missing_label_list = []

    for i, (sample_idx, img) in enumerate(zip(idx, data)):
        seed = sample_idx + 2025
        x_incomp, x_missing, missing_label = missing_sample(img["image"], missing_length, seed)
        x_incomp_list.append(x_incomp)
        x_missing_list.append(x_missing)
        missing_label_list.append(missing_label)
    return (
        torch.stack(x_incomp_list),
        torch.stack(x_missing_list),
        torch.stack(missing_label_list),
    )

def inference_collate_fn(batch, missing_length):
    data, idx = zip(*batch)
    x_incomp_list, x_missing_list = [], []
    missing_label_list = []
    for i, (sample_idx, img) in enumerate(zip(idx, data)):
        seed = sample_idx + 2025
        x_incomp, x_missing, missing_label = missing_sample(img["image"], missing_length, seed)
        x_incomp_list.append(x_incomp)
        x_missing_list.append(x_missing)
        missing_label_list.append(missing_label)
    return (
        torch.stack(x_incomp_list),
        torch.stack(x_missing_list),
        torch.stack(missing_label_list),
    )


class MonaiDataset(CacheDataset):

    def __getitem__(self, index: int | slice | Sequence[int]):
        """
        Returns a `Subset` if `index` is a slice or Sequence, a data item otherwise.
        """
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)
        if isinstance(index, collections.abc.Sequence):
            return Subset(dataset=self, indices=index)
        return self._transform(index), index

def get_loader(args, rank, world_size, train_files_combined, test_files_combined, train_transform, test_transform):
    inf_collate_fn = partial(inference_collate_fn, missing_length=args.missing_num)
    print(f"Total number of training data is {len(train_files_combined)}.")
    dataset_train = MonaiDataset(data=train_files_combined, transform=train_transform, cache_rate=args.cache, num_workers=8)
    if args.distributed:
        train_sampler = DistributedSampler(dataset=dataset_train, even_divisible=True, shuffle=True, rank=rank, num_replicas=world_size)
        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, collate_fn=inf_collate_fn, num_workers=8, shuffle=False, drop_last=True, sampler=train_sampler,
                                      persistent_workers=True, pin_memory=True, prefetch_factor=2)
    else:
        train_sampler = None
        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, collate_fn=inf_collate_fn, num_workers=4, shuffle=False, drop_last=True, sampler=train_sampler,
                                      persistent_workers=True, pin_memory=False, prefetch_factor=2)
    print(f"Total number of test data is {len(test_files_combined)}.")
    dataset_test = MonaiDataset(data=test_files_combined, transform=test_transform, cache_rate=args.cache, num_workers=8)
    if args.distributed:
        test_sampler = DistributedSampler(dataset=dataset_test, even_divisible=True, shuffle=False, rank=rank, num_replicas=world_size)
        dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, collate_fn=inf_collate_fn, num_workers=8, shuffle=False, drop_last=True, sampler=test_sampler,
                                     persistent_workers=True, pin_memory=True, prefetch_factor=2)
    else:
        test_sampler = None
        dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, collate_fn=inf_collate_fn, num_workers=4, shuffle=False, drop_last=True, sampler=test_sampler,
                                     persistent_workers=True, pin_memory=False, prefetch_factor=2)
    return dataloader_train, dataloader_test, train_sampler, test_sampler