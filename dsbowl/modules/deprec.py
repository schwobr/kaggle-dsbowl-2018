from skimage.io import imread, imshow
import os
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch
from torch.utils.data import Dataset
import cv2
import PIL
import random
import numpy as np
import matplotlib.pyplot as plt


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
