import os
from torch.utils.data import Dataset, DataLoader
import torch
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from modules.transforms import get_basics
from modules.transforms_functional import tensor_to_img
from numbers import Number


class CellsDataset(Dataset):
    def __init__(self, path, ids, size=256, transforms=None, use_augs=True):
        self.path = path
        self.ids = ids
        if isinstance(size, Number):
            size = (size, size)
        self.size = size
        if transforms is None:
            transforms = get_basics(size)
        self.transforms = transforms
        self.use_augs = use_augs

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        i = self.ids[idx]
        img_path, mask_path = self.__get_paths(i)
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        mask = self.__get_mask(mask_path, img.shape[:2])
        transformed = self.transforms(image=img, mask=mask)
        return transformed['image'], transformed['mask']

    def __get_paths(self, i):
        if self.use_augs:
            augs_path = self.path / str(i) / 'augs'
            spl = i.split('_')
            if len(spl) == 1:
                img_path = self.path / str(i) / 'images' / f'{i}.png'
                mask_path = self.path / str(i) / 'masks'
            else:
                augs_path = self.path / spl[0] / 'augs'
                img_path = augs_path / spl[1] / 'images' / f'{spl[1]}.png'
                mask_path = augs_path / spl[1] / 'masks'
        else:
            img_path = self.path / str(i) / 'images' / f'{i}.png'
            mask_path = self.path / str(i) / 'masks'
        return img_path, mask_path

    def __get_mask(self, mask_path, size, erosion=True, label=False):
        mask = np.zeros(size, np.uint8)
        for k, mask_file in enumerate(next(os.walk(mask_path))[2]):
            mask_ = cv2.imread(
                str(mask_path / mask_file),
                cv2.IMREAD_UNCHANGED)
            if erosion:
                mask_ = cv2.erode(
                    mask_.astype(np.uint8),
                    np.ones((3, 3),
                            np.uint8),
                    iterations=1)
            if label:
                mask_ = (mask_/255*(k+1)).astype(np.uint8)
            mask = mask + mask_
        return np.expand_dims(mask, -1)

    def show(self, idx, show_mask=True, label=False, transformed=False):
        i = self.ids[idx]
        img_path, mask_path = self.__get_paths(i)
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        mask = self.__get_mask(
            mask_path, img.shape[: 2],
            erosion=False, label=label)
        if transformed:
            tfm = self.transforms(image=img, mask=mask)
            img = tensor_to_img(tfm['image'])
            mask = tensor_to_img(tfm['mask'])
        plt.figure(0, (15, 15))
        if show_mask:
            if label:
                cmap = ListedColormap(np.random.rand(256, 3))
                cmap.set_bad(color='black')
                mask = np.where(mask > 0, mask, np.nan)
            else:
                cmap = 'gray'
            plt.subplot(121)
            plt.axis('off')
            plt.imshow(img)
            plt.subplot(122)
            plt.axis('off')
            plt.imshow(mask.squeeze(), cmap=cmap)
        else:
            plt.axis('off')
            plt.imshow(img)

    def show_rand(self, show_mask=True, label=False, transformed=False):
        idx = random.randint(0, len(self.ids)-1)
        self.show(idx, show_mask=show_mask,
                  label=label, transformed=transformed)


class Testset(Dataset):
    def __init__(self, path, ids, size=1388, transforms=None):
        self.path = path
        self.ids = ids
        if isinstance(size, Number):
            size = (size, size)
        self.size = size
        if transforms is None:
            transforms = get_basics(size)
        self.transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        i = self.ids[idx]
        img_path = self.path / str(i) / 'images' / f'{i}.png'
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        transformed = self.transforms(image=img)
        return transformed['image']

    def show(self, idx):
        i = self.ids[idx]
        img_path = self.path / str(i) / 'images' / f'{i}.png'
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        plt.figure(0, (15, 15))
        plt.axis('off')
        plt.imshow(img)

    def show_rand(self):
        idx = random.randint(0, len(self.ids)-1)
        self.show(idx)


def load_train_data(path, size=256, bs=8, val_split=0.2,
                    transforms=None, use_augs=True, shuffle=True):
    ids = next(os.walk(path))[1]
    if use_augs:
        aug_ids = []
        for i in ids:
            augs_path = path / str(i) / 'augs'
            aug_ids += [f'{i}_{aug_i}' for aug_i in next(
                os.walk(augs_path))[1]]
        ids += aug_ids
    if shuffle:
        ids = random.sample(ids, len(ids))
    n_ids = len(ids)
    k = (1-val_split)*random.random()
    trainset = CellsDataset(path, ids[:int(k*n_ids)]+ids[
                            int((k+val_split)*n_ids):], size=size,
                            transforms=transforms, use_augs=use_augs)
    valset = CellsDataset(
        path, ids[int(k * n_ids): int((k + val_split) * n_ids)],
        size=size,
        transforms=transforms, use_augs=use_augs)

    def col(l): return torch.cat(l, dim=0)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=bs, shuffle=True, num_workers=0, collate_fn=col)
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=bs, shuffle=False, num_workers=0, collate_fn=col)
    return trainloader, valloader


def load_test_data(path, size=1388, bs=8, transforms=None):
    test_ids = next(os.walk(path))[1]
    testset = Testset(path, test_ids,
                      size=size, transforms=transforms)
    testloader = DataLoader(testset, batch_size=bs,
                            shuffle=False, num_workers=0)
    return testloader


def get_stats(path, channels=3, bs=8, num_workers=0):
    dataset = CellsDataset(
        path, next(os.walk(path))[1],
        train=False, crop=True, erosion=False, grayscale=True, aug=False)
    dl = DataLoader(dataset, batch_size=bs, num_workers=num_workers,
                    shuffle=False)
    mean = torch.zeros(channels)
    std = torch.zeros(channels)
    nb_samples = 0.
    for data, _ in dl:
        batch_samples = data.size(0)
        data = data.view(batch_samples, channels, -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    return mean, std
