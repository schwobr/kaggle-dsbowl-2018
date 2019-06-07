import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch
import cv2
import PIL
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from modules.files import getNextId
from modules.transforms import get_basics
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
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        mask = self.__get_mask(mask_path)
        transformed = transforms(image=img, mask=mask)
        return transformed['image'], transforms['mask']

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
        return str(img_path), str(mask_path)

    def __get_mask(self, mask_path, erosion=True, label=False):
        mask = np.zeros(self.size, np.uint8)
        for k, mask_file in enumerate(next(os.walk(mask_path))[2]):
            mask_ = cv2.imread(
                mask_path / mask_file,
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

    def show(self, idx, show_mask=True, label=False):
        i = self.ids[idx]
        img_path, mask_path = self.__get_paths(self.ids[i])
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        mask = self.__get_mask(mask_path, eorsion=False, label=label)
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
            plt.imshow(mask, cmap=cmap)
        else:
            plt.axis('off')
            plt.imshow(img)

    def show_rand(self, show_mask=True, label=False):
        idx = random.randint(0, len(self.ids)-1)
        self.show(idx, show_mask=show_mask, label=label)


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
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        transformed = transforms(image=img)
        return transformed['image']

    def show(self, idx):
        i = self.ids[idx]
        img_path = self.path / str(i) / 'images' / f'{i}.png'
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        plt.figure(0, (15, 15))
        plt.axis('off')
        plt.imshow(img)

    def show_rand(self):
        idx = random.randint(0, len(self.ids)-1)
        self.show(idx)


def load_train_data(path, height=256, width=256, bs=8, val_split=0.2,
                    transforms=transforms, use_augs=True, shuffle=True):
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
                            int((k+val_split)*n_ids):], size=(height, width),
                            transforms=transforms, use_augs=use_augs)
    valset = CellsDataset(
        path, ids[int(k * n_ids): int((k + val_split) * n_ids)],
        size=(height, width),
        transforms=transforms, use_augs=use_augs)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs,
                                              shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(valset, batch_size=bs,
                                            shuffle=False, num_workers=0)
    return trainloader, valloader


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


def augment_data(path, hue_range=0.05, brightness_range=0.2,
                 contrast_range=0.2, p_hue=0.8, p_brightness=0.8,
                 p_contrast=0.8, max_rot=180, max_scale=0.1,
                 max_shear=10, p_hflip=0.5):
    ids = next(os.walk(path))[1]

    for i in ids:
        tfms = [transforms.Grayscale()]
        mask_tfms = []

        if random.random() < p_hue:
            tfms.append(transforms.Lambda(lambda x: TF.adjust_hue(
                x, random.uniform(-hue_range, hue_range))))
        if random.random() < p_brightness:
            tfms.append(transforms.Lambda(lambda x: TF.adjust_brightness(
                x, random.uniform(1-brightness_range,
                                  1+brightness_range))))
        if random.random() < p_contrast:
            tfms.append(transforms.Lambda(lambda x: TF.adjust_contrast(
                x, random.uniform(1-contrast_range,
                                  1+contrast_range))))

        angle, scale, shear = get_affine((-max_rot, max_rot),
                                         (1-max_scale, 1+max_scale),
                                         (-max_shear, max_shear))
        affine = transforms.Lambda(lambda x: TF.affine(x, angle, (0, 0),
                                                       scale, shear))
        tfms.append(affine)
        mask_tfms.append(affine)

        if random.random() < p_hflip:
            hflip = transforms.Lambda(lambda x: TF.hflip(x))
            tfms.append(hflip)
            mask_tfms.append(hflip)

        tfms.append(transforms.Grayscale(3))
        tfms = transforms.Compose(tfms)
        mask_tfms = transforms.Compose(mask_tfms)

        augs_path = path / str(i) / 'augs'
        if not os.path.exists(augs_path):
            os.makedirs(augs_path)
        new_id = str(getNextId(augs_path))
        os.makedirs(augs_path / new_id)
        os.makedirs(augs_path / new_id / 'images')
        os.makedirs(augs_path / new_id / 'masks')

        img = cv2.imread(str(path / str(i) / 'images' / f'{i}.png'))
        img = PIL.Image.fromarray(img)
        img = tfms(img)
        img.save(augs_path / new_id / 'images' / f'{new_id}.png')

        mask_path = path / str(i) / 'masks'
        for k, mask_file in enumerate(next(os.walk(mask_path))[2]):
            mask = cv2.imread(str(mask_path / mask_file))
            mask = PIL.Image.fromarray(mask)
            mask = mask_tfms(mask)
            mask.save(augs_path / new_id / 'masks' /
                      f'{new_id}_{k}.png')


def get_affine(degrees, scale_ranges, shears):
    angle = random.uniform(degrees[0], degrees[1])

    if scale_ranges is not None:
        scale = random.uniform(scale_ranges[0], scale_ranges[1])
    else:
        scale = 1.0

    if shears is not None:
        shear = random.uniform(shears[0], shears[1])
    else:
        shear = 0.0

    return angle, scale, shear
