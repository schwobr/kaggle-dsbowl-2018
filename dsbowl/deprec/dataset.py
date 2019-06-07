import os
import PIL
import cv2
import random
import matplotlib.pyplot as plt
from fastai.vision.data import SegmentationItemList, SegmentationLabelList
from fastai.vision.image import open_image, Image
from fastai.vision.transform import rand_pad
from skimage.io import imread, imshow
from pathlib import Path
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.util.data import Dataset
import numpy as np


class CellsDataset1(Dataset):
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
            try:
                img.shape[2]
            except IndexError:
                img = np.expand_dims(img, axis=2)
                img = np.concatenate((img, img, img), axis=2).astype(np.uint8)
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


def load_data(path, size=256, bs=8, val_split=0.2,
              erosion=True, normalize=None, testset=None):
    train_list = (
        SegmentationItemList.
        from_folder(path, extensions=['.png']).
        filter_by_func(lambda fn: Path(fn).parent.name == 'images').
        split_by_rand_pct(valid_pct=val_split).
        label_from_func(
            lambda x: x.parents[1] / 'masks/', label_cls=MultiMasksList,
            classes=['nucl'], erosion=erosion).transform(
            (rand_pad(0, size), rand_pad(0, size)), tfm_y=True))
    if testset:
        train_list.test = testset
    train_list = train_list.databunch(bs=bs, num_workers=0)
    if normalize:
        train_list = train_list.normalize(
            [torch.tensor(normalize[0]),
             torch.tensor(normalize[1])])
    return train_list


class MultiMasksList(SegmentationLabelList):
    def __init__(self, *args, erosion=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.erosion = erosion

    def open(self, fn):
        mask_files = next(os.walk(fn))[2]
        mask = open_image(os.path.join(fn, mask_files.pop(0)),
                          convert_mode='L').px
        for mask_file in mask_files:
            mask += open_image(os.path.join(fn, mask_file),
                               convert_mode='L').px
        if self.erosion:
            mask = torch.tensor(
                cv2.erode(
                    mask.numpy().squeeze().astype(np.uint8),
                    np.ones((3, 3),
                            np.uint8),
                    iterations=1)).unsqueeze(0)
        return Image(mask.float())

    def analyze_pred(self, pred, thresh: float = 0.5):
        return (pred > thresh).float()

    def reconstruct(self, t): return Image(t)
