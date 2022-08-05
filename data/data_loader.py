#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import glob
import h5py
import torch
import numpy as np
from typing import List
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision

import data.data_transform as Transforms
from utils.config import cfg


def download():
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www + ' --no-check-certificate', zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition):
    # download()
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))
    all_data = []
    all_label = []
    for h5_name in sorted(glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition))):
        f = h5py.File(h5_name, 'r')
        data = np.concatenate([f['data'][:], f['normal'][:]], axis=-1)
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def get_transforms(partition: str, num_points: int = 1024,
                   noise_type: str = 'clean', rot_mag: float = 45.0,
                   trans_mag: float = 0.5, partial_p_keep: List = None):
    """Get the list of transformation to be used for training or evaluating RegNet

    Args:
        noise_type: Either 'clean', 'jitter', 'crop'.
          Depending on the option, some of the subsequent arguments may be ignored.
        rot_mag: Magnitude of rotation perturbation to apply to source, in degrees.
          Default: 45.0 (same as Deep Closest Point)
        trans_mag: Magnitude of translation perturbation to apply to source.
          Default: 0.5 (same as Deep Closest Point)
        num_points: Number of points to uniformly resample to.
          Note that this is with respect to the full point cloud. The number of
          points will be proportionally less if cropped
        partial_p_keep: Proportion to keep during cropping, [src_p, ref_p]
          Default: [0.7, 0.7], i.e. Crop both source and reference to ~70%

    Returns:
        train_transforms, test_transforms: Both contain list of transformations to be applied
    """

    partial_p_keep = partial_p_keep if partial_p_keep is not None else [0.7, 0.7]

    if noise_type == "clean":
        # 1-1 correspondence for each point (resample first before splitting), no noise
        if partition == 'train':
            transforms = [Transforms.Resampler(num_points),
                          Transforms.SplitSourceRef(),
                          Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                          Transforms.ShufflePoints()]
        else:
            transforms = [Transforms.SetDeterministic(),
                          Transforms.Resampler(num_points),
                          Transforms.SplitSourceRef(),
                          Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                          Transforms.ShufflePoints()]

    elif noise_type == "jitter":
        # Points randomly sampled (might not have perfect correspondence), gaussian noise to position
        if partition == 'train':
            transforms = [Transforms.SetJitterFlag(),
                          Transforms.SplitSourceRef(),
                          Transforms.Resampler(num_points),
                          Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                          Transforms.RandomJitter(),
                          Transforms.ShufflePoints()]
        else:
            transforms = [Transforms.SetJitterFlag(),
                          Transforms.SetDeterministic(),
                          Transforms.SplitSourceRef(),
                          Transforms.Resampler(num_points),
                          Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                          Transforms.RandomJitter(),
                          Transforms.ShufflePoints()]

    elif noise_type == "crop":
        # Both source and reference point clouds cropped, plus same noise in "jitter"
        if partition == 'train':
            transforms = [Transforms.SetCropFlag(),
                          Transforms.SplitSourceRef(),
                          Transforms.Resampler(num_points),
                          Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                          Transforms.RandomCrop(partial_p_keep),
                          Transforms.RandomJitter(),
                          Transforms.ShufflePoints()]
        else:
            transforms = [Transforms.SetCropFlag(),
                          Transforms.SetDeterministic(),
                          Transforms.SplitSourceRef(),
                          Transforms.Resampler(num_points),
                          Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                          Transforms.RandomCrop(partial_p_keep),
                          Transforms.RandomJitter(),
                          Transforms.ShufflePoints()]
    else:
        raise NotImplementedError

    return transforms


class ModelNet40(Dataset):
    def __init__(self, partition='train', unseen=False, transform=None, crossval=False, train_part=False,
                 proportion=0.8):
        # data_shape:[B, N, 3]
        self.data, self.label = load_data(partition)
        if unseen and partition == 'train' and train_part is False:
            self.data, self.label = load_data('test')
        self.partition = partition
        self.unseen = unseen
        self.label = self.label.squeeze()
        self.transform = transform
        self.crossval = crossval
        self.train_part = train_part
        if self.unseen:
            # simulate training on first 20 categories while testing on last 20 categories
            if self.partition == 'test':
                self.data = self.data[self.label >= 20]
                self.label = self.label[self.label >= 20]
            elif self.partition == 'train':
                if self.train_part:
                    self.data = self.data[self.label < 20]
                    self.label = self.label[self.label < 20]
                else:
                    self.data = self.data[self.label < 20]
                    self.label = self.label[self.label < 20]
        else:
            if self.crossval:
                if self.train_part:
                    self.data = self.data[0:int(self.label.shape[0] * proportion)]
                    self.label = self.label[0:int(self.label.shape[0] * proportion)]
                else:
                    self.data = self.data[int(self.label.shape[0] * proportion):-1]
                    self.label = self.label[int(self.label.shape[0] * proportion):-1]

    def __getitem__(self, item):
        sample = {'points': self.data[item, :, :3], 'label': self.label[item], 'idx': np.array(item, dtype=np.int32)}

        if self.transform:
            sample = self.transform(sample)
        transform_gt = sample['transform_gt']
        transform_igt = np.concatenate(
            (transform_gt[:, :3].T, np.expand_dims(-(transform_gt[:, :3].T).dot(transform_gt[:, 3]), axis=1)), axis=-1)
        num_src, num_ref = sample['perm_mat'].shape

        ret_dict = {
            'points': [torch.Tensor(x) for x in [sample['points_src'], sample['points_ref']]],
            'num': [torch.tensor(x) for x in [num_src, num_ref]],
            'perm_mat_gt': torch.tensor(sample['perm_mat'].astype('float32')),
            'transform_gt': [torch.Tensor(x) for x in
                             [transform_gt.astype('float32'), transform_igt.astype('float32')]],
            'overlap_gt': [torch.Tensor(x) for x in [sample['src_overlap_gt'], sample['ref_overlap_gt']]],
            'label': torch.tensor(sample['label']),
            'points_src_raw': torch.Tensor(sample['points_src_raw']),
            'points_ref_raw': torch.Tensor(sample['points_ref_raw'])
        }
        return ret_dict

    def __len__(self):
        return self.data.shape[0]


def get_datasets(partition='train', num_points=1024, unseen=False,
                 noise_type="clean", rot_mag=45.0, trans_mag=0.5,
                 partial_p_keep=[0.7, 0.7], crossval=False, train_part=False):
    if cfg.DATASET_NAME == 'ModelNet40':
        transforms = get_transforms(partition=partition, num_points=num_points, noise_type=noise_type,
                                    rot_mag=rot_mag, trans_mag=trans_mag, partial_p_keep=partial_p_keep)
        transforms = torchvision.transforms.Compose(transforms)
        datasets = ModelNet40(partition, unseen, transforms, crossval=crossval, train_part=train_part)
    else:
        print('please input ModelNet40')

    return datasets


def collate_fn(data: list):
    """
    Create mini-batch data2d for training.
    :param data: data2d dict
    :return: mini-batch
    """

    def pad_tensor(inp):
        assert type(inp[0]) == torch.Tensor
        it = iter(inp)
        t = next(it)
        max_shape = list(t.shape)
        while True:
            try:
                t = next(it)
                for i in range(len(max_shape)):
                    max_shape[i] = int(max(max_shape[i], t.shape[i]))
            except StopIteration:
                break
        max_shape = np.array(max_shape)

        padded_ts = []
        for t in inp:
            pad_pattern = np.zeros(2 * len(max_shape), dtype=np.int64)
            pad_pattern[::-2] = max_shape - np.array(t.shape)
            pad_pattern = tuple(pad_pattern.tolist())
            padded_ts.append(F.pad(t, pad_pattern, 'constant', 0))

        return padded_ts

    def stack(inp):
        if type(inp[0]) == list:
            ret = []
            for vs in zip(*inp):
                ret.append(stack(vs))
        elif type(inp[0]) == dict:
            ret = {}
            for kvs in zip(*[x.items() for x in inp]):
                ks, vs = zip(*kvs)
                for k in ks:
                    assert k == ks[0], "Key value mismatch."
                ret[k] = stack(vs)
        elif type(inp[0]) == torch.Tensor:
            new_t = pad_tensor(inp)
            ret = torch.stack(new_t, 0)
        elif type(inp[0]) == np.ndarray:
            new_t = pad_tensor([torch.from_numpy(x) for x in inp])
            ret = torch.stack(new_t, 0)
        elif type(inp[0]) == str:
            ret = inp
        else:
            raise ValueError('Cannot handle type {}'.format(type(inp[0])))
        return ret

    ret = stack(data)

    return ret


def get_dataloader(dataset, phase, shuffle=False):
    if phase == 'test':
        batch_size = cfg.DATASET.TEST_BATCH_SIZE
    else:
        batch_size = cfg.DATASET.TRAIN_BATCH_SIZE
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                       shuffle=shuffle, num_workers=cfg.DATALOADER_NUM,
                                       collate_fn=collate_fn, pin_memory=False)
