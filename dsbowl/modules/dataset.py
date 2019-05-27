from skimage.io import imread, imshow
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch
from fastai.vision.data import SegmentationItemList, SegmentationLabelList
from fastai.vision.image import (open_image,  ImageSegment)
from fastai.vision.transform import rand_pad
import cv2
import PIL
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class CellsDataset(Dataset):
    def __init__(self, path, ids, height=256, width=256, train=True,
                 erosion=True, normalize=None, crop=True, resize=False,
                 pad=False, grayscale=True, aug=True):
        self.path = path
        self.ids = ids
        self.height = height
        self.width = width
        self.c = 1
        self.erosion = erosion
        self.normalize = normalize
        self.resize = resize
        self.crop = crop
        self.grayscale = grayscale
        self.train = train
        self.aug = aug
        self.pad = pad

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        ids = np.array(self.ids[idx], ndmin=1)
        images = []
        masks = []
        sizes = []
        for i in ids:
            tfms = []
            mask_tfms = []

            if self.aug:
                augs_path = os.path.join(self.path, i, 'augs')
                spl = i.split('_')
                if len(spl) == 1:
                    img_path = os.path.join(
                        self.path, i, 'images', f'{i}.png')
                    mask_path = os.path.join(self.path, i, 'masks')
                else:
                    augs_path = os.path.join(self.path, spl[0], 'augs')
                    img_path = os.path.join(augs_path, spl[1],
                                            'images', f'{spl[1]}.png')
                    mask_path = os.path.join(augs_path, spl[1], 'masks')
            else:
                img_path = os.path.join(self.path, i, 'images', f'{i}.png')
                mask_path = os.path.join(self.path, i, 'masks')
            img = imread(img_path)
            sizes.append(torch.tensor(img.shape).unsqueeze(0))
            img = PIL.Image.fromarray(img)

            if self.grayscale:
                tfms.append(transforms.Grayscale(3))

            if self.crop:
                i, j, th, tw = self.__get_crop(img)
                crop = transforms.Lambda(lambda x: TF.crop(x, i, j, th, tw))
                tfms.append(crop)
                mask_tfms.append(crop)

            if self.resize:
                resize = transforms.Resize((self.height, self.width))
                tfms.append(resize)
                mask_tfms.append(resize)

            if self.pad:
                assert (self.train is False and self.width >= img.width
                        and self.height >= img.height)
                pad = transforms.Pad((0, 0, self.width-img.width,
                                      self.height-img.height))
                tfms.append(pad)

            tfms.append(transforms.ToTensor())
            mask_tfms.append(transforms.ToTensor())

            if self.normalize:
                tfms.append(transforms.Normalize(self.normalize[0],
                                                 self.normalize[1]))

            transform = transforms.Compose(tfms)
            mask_transform = transforms.Compose(mask_tfms)
            images.append(transform(img).unsqueeze(0))

            if self.train:
                masks.append(self.__get_mask(mask_path, mask_transform))

        if self.train:
            return torch.cat(images).squeeze(), torch.cat(masks)
        else:
            return torch.cat(images).squeeze(), torch.cat(sizes).squeeze()

    def show(self, idx):
        img = self.__getitem__(idx)
        img = np.asarray(TF.to_pil_image(img))
        imshow(img)
        plt.show()
        return img

    def __get_crop(self, img):
        w, h = img.size
        th, tw = self.height, self.width
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __get_mask(self, mask_path, transform):
        mask = torch.zeros((self.height, self.width))
        for mask_file in next(os.walk(mask_path))[2]:
            mask_ = imread(os.path.join(mask_path, mask_file))
            if self.erosion:
                mask_ = cv2.erode(
                    mask_.astype(np.uint8),
                    np.ones((3, 3),
                            np.uint8),
                    iterations=1)
            mask_ = PIL.Image.fromarray(mask_)
            mask_ = transform(mask_)
            mask = mask + mask_
        return mask


def load_train_data(path, height=256, width=256, bs=8, val_split=0.2,
                    erosion=True, normalize=None, crop=True,
                    grayscale=True, aug=True, shuffle=True):
    ids = next(os.walk(path))[1]
    if aug:
        aug_ids = []
        for i in ids:
            augs_path = os.path.join(path, i, 'augs')
            aug_ids += [f'{i}_{aug_i}' for aug_i in next(
                os.walk(augs_path))[1]]
        ids += aug_ids
    if shuffle:
        ids = random.sample(ids, len(ids))
    n_ids = len(ids)
    k = (1-val_split)*random.random()
    trainset = CellsDataset(path, ids[:int(k*n_ids)]+ids[
                            int((k+val_split)*n_ids):], height, width,
                            erosion=erosion, normalize=normalize, crop=crop,
                            grayscale=grayscale, aug=aug)
    valset = CellsDataset(path, ids[int(k*n_ids):int((k+val_split)*n_ids)],
                          height, width, erosion=erosion, normalize=normalize,
                          crop=crop, grayscale=grayscale, aug=aug)
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

        augs_path = os.path.join(path, i, 'augs')
        if not os.path.exists(augs_path):
            os.makedirs(augs_path)
        new_id = str(getNextId(augs_path))
        os.makedirs(os.path.join(augs_path, new_id))
        os.makedirs(os.path.join(augs_path, new_id, 'images'))
        os.makedirs(os.path.join(augs_path, new_id, 'masks'))

        img = imread(os.path.join(path, i, 'images', f'{i}.png'))
        img = PIL.Image.fromarray(img)
        img = tfms(img)
        img.save(os.path.join(augs_path, new_id, 'images', f'{new_id}.png'))

        mask_path = os.path.join(path, i, 'masks')
        for k, mask_file in enumerate(next(os.walk(mask_path))[2]):
            mask = imread(os.path.join(mask_path, mask_file))
            mask = PIL.Image.fromarray(mask)
            mask = mask_tfms(mask)
            mask.save(os.path.join(augs_path, new_id, 'masks',
                                   f'{new_id}_{k}.png'))


def getNextId(output_folder):
    highest_num = -1
    for d in os.listdir(output_folder):
        dir_name = os.path.splitext(d)[0]
        try:
            i = int(dir_name)
            if i > highest_num:
                highest_num = i
        except ValueError:
            'The dir name "%s" is not an integer. Skipping' % dir_name

    new_id = highest_num + 1
    return new_id


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


def load_data(path, size=256, bs=8, val_split=0.2,
              erosion=True, normalize=True, shuffle=True):
    train_list = (
        SegmentationItemList.
        from_folder(path, extensions=['.png']).
        filter_by_func(lambda fn: fn.parent == Path('/images')).
        split_by_rand_pct(valid_pct=val_split).
        label_from_func(
            lambda x: x.parents[1] / 'masks/', label_cls=MultiMasksList,
            classes=['nucl'], erosion=erosion).transform(
            (rand_pad(0, size), rand_pad(0, size)), tfm_y=True).databunch(
            bs=bs, num_workers=0, shuffle=shuffle))
    if normalize:
        train_list = train_list.normalize()
    return train_list


class MultiMasksList(SegmentationLabelList):
    def __init__(self, *args, erosion=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.erosion = erosion

    def open(self, fn):
        mask_files = next(os.walk(fn))[2]
        mask = open_image(os.path.join(fn, mask_files.pop(0))).px
        for mask_file in mask_files:
            mask += open_image(os.path.join(fn, mask_file),
                               convert_mode='L').px
        if self.erosion:
            mask = torch.tensor(
                cv2.erode(
                    mask.numpy().astype(np.uint8),
                    np.ones((3, 3),
                            np.uint8),
                    iterations=1)).float()
        return ImageSegment(mask)

    def analyze_pred(self, pred, thresh: float = 0.5):
        return (pred > thresh).float()

    def reconstruct(self, t): return ImageSegment(t)
