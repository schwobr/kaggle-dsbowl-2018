import cv2
import PIL
import random
import numpy as np
import os

from skimage.io import imread

from torchvision import transforms
import torchvision.transforms.functional as TF
import torch

from fastai.vision.data import SegmentationItemList, SegmentationLabelList
from fastai.vision.image import open_image, Image, image2np, pil2tensor
from fastai.vision.transform import rand_pad

import modules.transforms_functional as F
from modules.files import getNextId


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


def load_data(
        path, size=256, bs=8, val_split=0.2, use_augs=True, erosion=True,
        normalize=None, classes=['nucl'],
        testpath=None, max_size=1388, tfms=True):
    if use_augs:
        def filter_func(fn): return fn.parent.name == 'images'
    else:
        def filter_func(fn): return fn.parent.name == 'images' and \
            not fn.with_suffix('').name.isdigit()

    train_list = (
        CellImageList.
        from_folder(path, extensions=['.png']).
        filter_by_func(filter_func).
        split_by_rand_pct(valid_pct=val_split).
        label_from_func(
            lambda x: x.parents[1] / 'masks', label_cls=MultiMasksList,
            classes=classes, erosion=erosion))

    if tfms:
        train_list.transform(
            (rand_pad(0, size), rand_pad(0, size)), tfm_y=True)

    train_list = train_list.databunch(bs=bs, num_workers=0)

    if testpath:
        test_list = CellImageList.from_folder(
            testpath, extensions=['.png'],
            max_size=max_size)
        train_list.add_test(test_list)
        train_list.test_ds.x.max_size = max_size

    if normalize:
        train_list = train_list.normalize(
            [torch.tensor(normalize[0]),
             torch.tensor(normalize[1])])
    return train_list


class CellImageList(SegmentationItemList):
    def __init__(self, *args, max_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        print(max_size)
        self.max_size = max_size

    def open(self, fn):
        x = super().open(fn).data
        x = image2np(x)
        x = F.to_gray(x)
        if self.max_size:
            pad_width = (
                (0, self.max_size - x.shape[0]),
                (0, self.max_size - x.shape[1]),
                (0, 0))
            x = F.pad(x, pad_width)
        return Image(pil2tensor(x, np.float32))


class MultiMasksList(SegmentationLabelList):
    def __init__(self, *args, erosion=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.erosion = erosion

    def open(self, fn):
        mask_files = next(os.walk(fn))[2]
        mask = open_image(fn / mask_files.pop(0),
                          convert_mode='L').px
        for mask_file in mask_files:
            mask += open_image(fn / mask_file,
                               convert_mode='L').px
        if self.erosion:
            mask = pil2tensor(
                cv2.erode(
                    image2np(mask).astype(np.uint8),
                    np.ones((3, 3),
                            np.uint8),
                    iterations=1), np.float32)
        return Image(mask)

    def analyze_pred(self, pred, thresh: float = 0.5):
        return (pred > thresh).float()

    def reconstruct(self, t): return Image(t)


class TestImageList(CellImageList):
    def __init__(self, *args, max_size=1388, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_size = max_size
        self.label_cls = None

    def open(self, fn):
        x = super().open(fn).data
        x = image2np(x)
        pad_width = (
            (0, self.max_size - x.shape[0]),
            (0, self.max_size - x.shape[1]),
            (0, 0))
        x = F.pad(x, pad_width)
        return Image(pil2tensor(x, np.float32))
