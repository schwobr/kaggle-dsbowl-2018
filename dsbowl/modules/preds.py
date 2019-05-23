import os

import numpy as np
import pandas as pd
import cv2

from math import ceil

from tqdm import tqdm_notebook
from skimage.morphology import label

from fastai.basic_data import DatasetType

import torch
import torchvision.transforms.functional as TF


def getNextFilePath(output_folder):
    highest_num = 0
    for f in os.listdir(output_folder):
        if os.path.isfile(os.path.join(output_folder, f)):
            file_name = os.path.splitext(f)[0]
            try:
                split = file_name.split('.')
                split = split[0].split('_')
                file_num = int(split[-1])
                if file_num > highest_num:
                    highest_num = file_num
            except ValueError:
                'The file name "%s" is not an integer. Skipping' % file_name

    output_file = highest_num + 1
    return output_file


def predict_all(learner):
    preds = []
    sizes = []
    with torch.no_grad():
        for X_test, sizes_test in tqdm_notebook(learner.dl(DatasetType.Test)):
            sizes.append(sizes_test)
            preds.append(learner.model(X_test).cpu())
    sizes = torch.cat(sizes).numpy().squeeze()
    preds = torch.cat(preds)
    preds = preds.squeeze().numpy()
    return preds, sizes


def predict_TTA_all(learner, size=(512, 512), overlap=64,
                    rotations=(0, 90, 180, 270), device='cuda:0'):
    preds = []
    sizes = []
    with torch.no_grad():
        for X_test, sizes_test in tqdm_notebook(learner.dl(DatasetType.Test)):
            sizes.append(sizes_test)
            for tens, s in zip(X_test, sizes_test.squeeze()):
                crops, pos, overlaps = get_crops(tens.cpu()[:, :s[0], :s[1]],
                                                 size, overlap)
                pred = torch.zeros((s[0], s[1]))
                for crop, ((x_min, y_min), (x_max, y_max)) in zip(crops, pos):
                    img = TF.to_pil_image(crop)
                    res = predict_TTA(learner, img, size, rotations, device)
                    pred[x_min:x_max, y_min:y_max] += res[
                        :x_max-x_min, :y_max-y_min]/overlaps[
                        :x_max-x_min, :y_max-y_min]
                preds.append(pred.numpy())
    sizes = torch.cat(sizes).cpu().numpy().squeeze()
    return preds, sizes


def get_crops(img, size, overlap):
    n, h, w = img.size()
    n_h = ceil((h+overlap/2-1)/(size[0]-overlap))
    n_w = ceil((w+overlap/2-1)/(size[1]-overlap))
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


def predict_TTA(learner, img, size, rotations, device):
    flipped = TF.hflip(img)
    res = torch.zeros(size)
    for angle in rotations:
        rot = TF.rotate(img, angle)
        rot = TF.to_tensor(rot).unsqueeze(0).to(device)
        rot_flipped = TF.rotate(flipped, angle)
        rot_flipped = TF.to_tensor(rot_flipped).unsqueeze(0).to(device)

        out_rot = learner.model(rot).cpu().squeeze()
        out_rot = torch.sigmoid(out_rot).squeeze()
        out_rot = TF.rotate(TF.to_pil_image(out_rot), -angle)
        out_rot = TF.to_tensor(out_rot).squeeze()

        out_flipped = learner.model(rot_flipped).cpu().squeeze()
        out_flipped = torch.sigmoid(out_flipped).squeeze()
        out_flipped = TF.hflip(TF.rotate(TF.to_pil_image(out_flipped), -angle))
        out_flipped = TF.to_tensor(out_flipped).squeeze()

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
    if resize:
        preds_test_upsampled = []
        for i, pred in enumerate(preds):
            preds_test_upsampled.append(resize(
                                        pred, (sizes[i, 0], sizes[i, 1]),
                                        mode='constant', preserve_range=True))
    else:
        preds_test_upsampled = preds
    new_test_ids = []
    rles = []
    for n, id_ in enumerate(tqdm_notebook(test_ids)):
        rle = list(prob_to_rles(preds_test_upsampled[n]))
        rles.extend(rle)
        if rle == []:
            rles.append([])
        new_test_ids.extend([id_] * max(1, len(rle)))

    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(
        lambda x: ' '.join(str(y) for y in x))

    sub_file = os.path.join(folder,
                            f'sub_dsbowl_pt_{getNextFilePath(folder)}.csv')
    sub.to_csv(sub_file, index=False)
