import argparse
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import random

from PIL import ImageEnhance, Image
from torch.utils.data import Subset
import collections
from collections.abc import Sequence
import os
import warnings
from monai.data import *
import itertools
from scipy import ndimage
import pydicom
import medpy.io as medio
import glob
import math
from sklearn.model_selection import train_test_split
from monai.transforms import *


warnings.filterwarnings("ignore")



def get_brainMRI(args):
    # base_dir = 'dataset/BrainMRI/'
    base_dir = '../dataset/BrainMRI/'

    data_dir1 = base_dir + 'BraTS2021/'
    data_list1 = get_volume_BraTS(data_dir1)
    train_dicts1, test_dicts1 = dataset_split_BraTS(data_list1, args.seed)

    data_dir2 = base_dir + 'IXI/'
    data_list2 = get_volume_IXI(data_dir2)
    train_dicts2, test_dicts2 = dataset_split_IXI(data_list2, args.seed)

    datasets = {
        1: {
            "data_name": "BraTS",
            "train_files": train_dicts1,
            "test_files": test_dicts1,
            "modality": "MRI",
        },
        2: {
            "data_name": "IXI",
            "train_files": train_dicts2,
            "test_files": test_dicts2,
            "modality": "MRI",
        },
    }

    return datasets


def dataset_split_BraTS(data_list, seed):
    comnined_list = list(zip(data_list["id"], data_list["t1"],data_list["t2"],data_list["t1ce"],data_list["flair"]))
    train_files, test_files = train_test_split(comnined_list, test_size=0.2, random_state=seed)

    id_list, t1_list, t2_list, flair_list, t1ce_list = zip(*train_files)
    train_data = []
    for i in range(len(id_list)):
        train_data.append({"t1": t1_list[i], "t1ce": t1ce_list[i], "t2": t2_list[i], "flair": flair_list[i], "subject_id": id_list[i]})

    id_list, t1_list, t2_list, flair_list, t1ce_list = zip(*test_files)
    test_data = []
    for i in range(len(id_list)):
        test_data.append({"t1": t1_list[i], "t1ce": t1ce_list[i], "t2": t2_list[i], "flair": flair_list[i], "subject_id": id_list[i]})

    return train_data, test_data

def dataset_split_IXI(data_list, seed):
    comnined_list = list(zip(data_list["id"], data_list["t1"],data_list["t2"],data_list["pd"]))
    train_files, test_files = train_test_split(comnined_list, test_size=0.2, random_state=seed)

    id_list, t1_list, t2_list, pd_list = zip(*train_files)
    train_data = []
    for i in range(len(id_list)):
        train_data.append({"t1": t1_list[i], "t2": t2_list[i], "pd": pd_list[i], "subject_id": id_list[i]})

    id_list, t1_list, t2_list, pd_list = zip(*test_files)
    test_data = []
    for i in range(len(id_list)):
        test_data.append({"t1": t1_list[i], "t2": t2_list[i], "pd": pd_list[i], "subject_id": id_list[i]})

    return train_data, test_data

def get_volume_BraTS(data_dir):
    id_list = [os.path.basename(folder) for folder in glob.glob(os.path.expanduser(data_dir + '/*'))]
    t1_list = sorted(glob.glob(data_dir + 'BraTS2021_*/*_t1.nii.gz'))
    t2_list = sorted(glob.glob(data_dir + 'BraTS2021_*/*_t2.nii.gz'))
    t1ce_list = sorted(glob.glob(data_dir + 'BraTS2021_*/*_t1ce.nii.gz'))
    flair_list = sorted(glob.glob(data_dir + 'BraTS2021_*/*_flair.nii.gz'))
    volume_list = {
        "t1": t1_list,
        "t1ce": t1ce_list,
        "t2": t2_list,
        "flair": flair_list,
        "id": id_list
    }
    return volume_list


def get_volume_IXI(data_dir):
    t1_list = sorted(glob.glob(data_dir + 'IXI-T1/*-T1.nii.gz'))
    t2_list = sorted(glob.glob(data_dir + 'IXI-T2/*-T2.nii.gz'))
    pd_list = sorted(glob.glob(data_dir + 'IXI-PD/*-PD.nii.gz'))
    t1_id = [s.replace('-T1.nii.gz', '').split('\\')[-1] for s in t1_list]
    t2_id = [s.replace('-T2.nii.gz', '').split('\\')[-1] for s in t2_list]
    pd_id = [s.replace('-PD.nii.gz', '').split('\\')[-1] for s in pd_list]
    id_list = sorted(set(t1_id) & set(t2_id) & set(pd_id))
    t1_list = [data_dir + 'IXI-T1/' + s + '-T1.nii.gz' for s in id_list]
    t2_list = [data_dir + 'IXI-T2/' + s + '-T2.nii.gz' for s in id_list]
    pd_list = [data_dir + 'IXI-PD/' + s + '-PD.nii.gz' for s in id_list]
    volume_list = {
        "t1": t1_list,
        "t2": t2_list,
        "pd": pd_list,
        "id": id_list
    }
    return volume_list


def get_data(datasets, dataset_name):
    data_by_name  = {}
    for key, dataset in datasets.items():
        data_name = dataset["data_name"]
        data_by_name[data_name] = dataset
    dataset_single = data_by_name[dataset_name]
    return dataset_single['train_files'], dataset_single['test_files']


def get_transforms(args, keys):

    if args.dataset == 'BraTS':
        common_transform = [
                    LoadImaged(keys=keys, image_only=True, allow_missing_keys=True),
                    EnsureChannelFirstd(keys=keys, allow_missing_keys=True),
                    EnsureTyped(keys=keys),
                    Orientationd(keys=keys, axcodes="RAS", allow_missing_keys=True),
                    CropForegroundd(keys=keys, source_key=keys[0], allow_missing_keys=True),
                    ScaleIntensityRangePercentilesd(keys=keys, lower=0, upper=99.5, b_min=0, b_max=1),
                    CenterSpatialCropd(keys=keys, roi_size=args.brain_roi, allow_missing_keys=True)
                ]
    else:
        common_transform = [
                    LoadImaged(keys=keys, image_only=True, allow_missing_keys=True),
                    EnsureChannelFirstd(keys=keys, allow_missing_keys=True),
                    EnsureTyped(keys=keys),
                    Orientationd(keys=keys, axcodes="RAS", allow_missing_keys=True),
                    ScaleIntensityRangePercentilesd(keys=keys, lower=0, upper=99.5, b_min=0, b_max=1),
                    CenterSpatialCropd(keys=keys, roi_size=args.brain_roi, allow_missing_keys=True)
                ]

    train_transform = Compose(
        common_transform
        + [
            RandSpatialCropd(keys=keys, roi_size=args.brain_roi, allow_missing_keys=True, random_size=False, random_center=True),
            RandFlipd(keys=keys, prob=0.3, spatial_axis=0, allow_missing_keys=True),
            RandFlipd(keys=keys, prob=0.3, spatial_axis=1, allow_missing_keys=True),
            RandFlipd(keys=keys, prob=0.3, spatial_axis=2, allow_missing_keys=True),
            RandRotate90d(keys=keys, prob=0.3, max_k=3, allow_missing_keys=True),
            RandScaleIntensityd(keys=keys, prob=0.2, factors=(0.9, 1.1), allow_missing_keys=True),
            RandShiftIntensityd(keys=keys, prob=0.2, offsets=0.05, allow_missing_keys=True),
            Resized(keys=keys, spatial_size=args.brain_size, mode="trilinear", align_corners=True, allow_missing_keys=True),
            ToTensord(keys=keys, allow_missing_keys=True)
        ])

    test_transform = Compose(
        common_transform
        + [
            Resized(keys=keys, spatial_size=args.brain_size, mode="trilinear", align_corners=True, allow_missing_keys=True),
            ToTensord(keys=keys, allow_missing_keys=True)
        ])

    return train_transform, test_transform


def modality_sample(x_in, seed0, seed1):
    torch.manual_seed(seed0)
    modality = x_in.shape[1]

    missing_values = list(range(1, modality))
    idx = int(torch.randint(0, len(missing_values), (1,)).item())
    missing_num = missing_values[idx]

    missing_num_list = np.zeros(len(missing_values), dtype=int)
    missing_num_list[idx] = 1

    torch.manual_seed(seed1)

    comp_list = list(range(modality))
    indices = torch.randperm(modality)[:missing_num]
    missing_idx = sorted([comp_list[i] for i in indices])
    incomp_idx = list(set(comp_list) - set(missing_idx))
    x_missing = x_in[:, missing_idx, ...]
    x_incomp = x_in[:, incomp_idx, ...]
    incomp_num = modality - missing_num
    missing_label = get_label(modality, incomp_num, missing_idx)
    return x_incomp, x_missing, torch.tensor(missing_num_list, dtype=torch.float32), missing_label


def get_label(number, incomp_length, missing_idx):
    missing_label = np.zeros(number, dtype=int)
    for i in range(len(missing_idx)):
        missing_label[missing_idx[i]] = 1
    missing_label = torch.FloatTensor(missing_label)
    return missing_label


def collate_fn_BraTS(batch):
    data, idx = zip(*batch)
    seed0 = sum(idx) + 2025
    x_incomp_list, x_missing_list = [], []
    missing_length_list, missing_label_list = [], []

    for i, (sample_idx, item) in enumerate(zip(idx, data)):

        seed1 = sample_idx + 2025
        img = torch.stack((item['t1'], item['t2'], item['t1ce'], item['flair']), dim=1)
        x_incomp, x_missing, missing_length, missing_label = modality_sample(img, seed0, seed1)

        x_incomp_list.append(x_incomp)
        x_missing_list.append(x_missing)
        missing_length_list.append(missing_length)
        missing_label_list.append(missing_label)

    return (
        torch.cat(x_incomp_list),
        torch.cat(x_missing_list),
        torch.stack(missing_length_list),
        torch.stack(missing_label_list),
    )

def collate_fn_IXI(batch):
    data, idx = zip(*batch)
    seed0 = sum(idx) + 2025
    x_incomp_list, x_missing_list = [], []
    missing_length_list, missing_label_list = [], []

    for i, (sample_idx, item) in enumerate(zip(idx, data)):

        seed1 = sample_idx + 2025
        img = torch.stack((item['t1'], item['t2'], item['pd']), dim=1)
        x_incomp, x_missing, missing_length, missing_label = modality_sample(img, seed0, seed1)

        x_incomp_list.append(x_incomp)
        x_missing_list.append(x_missing)
        missing_length_list.append(missing_length)
        missing_label_list.append(missing_label)

    return (
        torch.cat(x_incomp_list),
        torch.cat(x_missing_list),
        torch.stack(missing_length_list),
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
    collate_fn = collate_fn_BraTS if args.dataset == 'BraTS' else collate_fn_IXI

    dataset_train = MonaiDataset(data=train_files_combined, transform=train_transform, cache_rate=args.cache, num_workers=8)
    if args.distributed:
        train_sampler = DistributedSampler(dataset=dataset_train, even_divisible=True, shuffle=True, rank=rank, num_replicas=world_size)
        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=8, shuffle=False, drop_last=True, sampler=train_sampler,
                                      persistent_workers=True, pin_memory=True, prefetch_factor=2)
    else:
        train_sampler = None
        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=4, shuffle=False, drop_last=True, sampler=train_sampler,
                                      persistent_workers=True, pin_memory=False, prefetch_factor=2)

    dataset_test = MonaiDataset(data=test_files_combined, transform=test_transform, cache_rate=args.cache, num_workers=8)
    if args.distributed:
        test_sampler = DistributedSampler(dataset=dataset_test, even_divisible=True, shuffle=False, rank=rank, num_replicas=world_size)
        dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=8, shuffle=False, drop_last=True, sampler=test_sampler,
                                     persistent_workers=True, pin_memory=True, prefetch_factor=2)
    else:
        test_sampler = None
        dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=4, shuffle=False, drop_last=True, sampler=test_sampler,
                                     persistent_workers=True, pin_memory=False, prefetch_factor=2)
    return dataloader_train, dataloader_test, train_sampler, test_sampler


