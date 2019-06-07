import os

import numpy as np
import pandas as pd
import cv2

from math import ceil

from tqdm.autonotebook import tqdm
from skimage.morphology import label

import torch
import torchvision.transforms.functional as TF

from modules.files import getNextFilePath
from modules.transforms_functional import tensor_to_img


def predict_all(model, dl):
    preds = []
    with torch.no_grad():
        for X_test, sizes_test in tqdm(dl):
            preds.append(model(X_test).cpu())
    preds = torch.cat(preds)
    return preds


def predict_TTA_all(model, dl, device, size=(512, 512), overlap=64,
                    rotations=(0, 90, 180, 270)):
    preds = []
    with torch.no_grad():
        for X_test, sizes_test in tqdm(dl):
            for tens, s in zip(X_test, sizes_test):
                crops, pos, overlaps = get_crops(tens.cpu()[:, :s[0], :s[1]],
                                                 size, overlap)
                pred = torch.zeros((1, s[0], s[1]))
                for crop, ((x_min, y_min), (x_max, y_max)) in zip(crops, pos):
                    img = TF.to_pil_image(crop)
                    res = predict_TTA(model, img, size, rotations, device)
                    pred[:, x_min:x_max, y_min:y_max] += res[
                         :, :x_max-x_min, :y_max-y_min]
                pred /= overlaps
                preds.append(pred)
    return torch.cat(preds)


def get_crops(img, size, overlap):
    n, h, w = img.size()
    n_h = max(1, ceil((h-size[0])/(size[0]-overlap)))
    n_w = max(1, ceil((w-size[1])/(size[1]-overlap)))
    crops = []
    pos = []
    overlaps = torch.zeros((h, w))
    for i in range(n_h):
        for j in range(n_w):
            crop = torch.zeros((n, size[0], size[1]))
            x_min = i*(size[0]-overlap)
            y_min = j*(size[1]-overlap)
            x_max = min(h, (i+1)*size[0]-i*overlap)
            y_max = min(w, (j+1)*size[1]-j*overlap)
            crop[:, :x_max-x_min, :y_max-y_min] = img[:, x_min:x_max,
                                                      y_min:y_max]
            crops.append(crop)
            overlaps[x_min:x_max, y_min:y_max] += 1
            pos.append(((x_min, y_min), (x_max, y_max)))
    return crops, pos, overlaps


def predict_TTA(model, img, size, rotations, device):
    flipped = TF.hflip(img)
    res = torch.zeros(size).unsqueeze(0)
    for angle in rotations:
        rot = TF.rotate(img, angle)
        rot = TF.to_tensor(rot).to(device)
        rot_flipped = TF.rotate(flipped, angle)
        rot_flipped = TF.to_tensor(rot_flipped).to(device)

        out_rot = model(rot).cpu()
        out_rot = torch.sigmoid(out_rot)
        out_rot = TF.rotate(TF.to_pil_image(out_rot), -angle)
        out_rot = TF.to_tensor(out_rot)

        out_flipped = model(rot_flipped).cpu()
        out_flipped = torch.sigmoid(out_flipped)
        out_flipped = TF.hflip(
            TF.rotate(TF.to_pil_image(out_flipped), -angle))
        out_flipped = TF.to_tensor(out_flipped)

        res += (out_rot+out_flipped)/2
    res /= len(rotations)
    return res


def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev+1):
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    mask = np.zeros_like(lab_img)
    for i in range(1, lab_img.max() + 1):
        img = ((lab_img == i).astype(np.uint8) * 255)
        img = cv2.dilate(img, np.ones((3, 3), np.uint8),
                         iterations=1).astype('float32')/255
        mask[img > cutoff] = i
    for i in range(1, mask.max() + 1):
        yield rle_encoding(mask == i)


def create_submission(preds, sizes, test_ids, folder, resize=False):
    preds = preds
    if resize:
        preds_test_upsampled = []
        for i, pred in enumerate(preds):
            pred = tensor_to_img(pred)
            preds_test_upsampled.append(cv2.resize(
                                        pred, (sizes[i, 0], sizes[i, 1])))
    else:
        preds_test_upsampled = [tensor_to_img(pred) for pred in preds]
    new_test_ids = []
    rles = []
    for n, id_ in enumerate(tqdm(test_ids)):
        rle = list(prob_to_rles(preds_test_upsampled[n]))
        rles.extend(rle)
        if rle == []:
            rles.append([])
        new_test_ids.extend([id_] * max(1, len(rle)))

    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(
        lambda x: ' '.join(str(y) for y in x))

    base_name = 'sub_dsbowl_pt'
    sub_file = os.path.join(
        folder, base_name+f'_{getNextFilePath(folder, base_name)}.csv')
    sub.to_csv(sub_file, index=False)
