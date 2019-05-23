import os
import random
import warnings

import numpy as np
import cv2

from skimage.morphology import label

from fastai.vision.data import ImageDataBunch
from fastai.vision.learner import unet_learner
import fastai.vision.models as mod
from fastai.callbacks import SaveModelCallback

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from modules.dataset import CellsDataset, load_train_data
from modules.preds import predict_TTA_all, create_submission

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
TRAIN_PATH = '../stage1_train/'
TEST_PATH = '../stage2_test_final/'
BATCH_SIZE = 4
MEAN = 0.5
STD = 0.5

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

# Get train and test IDs
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]


def mean_iou(y_pred, y_true, smooth=1e-6):
    scores = np.zeros(y_true.shape[0])
    y_true = y_true.squeeze(1)
    y_pred = torch.sigmoid(y_pred).squeeze(1)
    for i in range(y_true.shape[0]):
        labels_pred = label(y_pred.to('cpu').numpy()[i] > 0.5)
        labels_true = label(y_true.to('cpu').numpy()[i])
        score = 0
        cnt = 0
        n_masks_pred = np.max(labels_pred)
        n_masks_true = np.max(labels_true)
        inter_union = np.zeros((n_masks_pred, n_masks_true, 2), dtype=np.int)
        for k in range(y_true.shape[1]):
            for l in range(y_true.shape[2]):
                m = labels_pred[k, l]
                n = labels_true[k, l]
                if m != 0:
                    inter_union[m-1, :, 1] += 1
                if n != 0:
                    inter_union[:, n-1, 1] += 1
                if m != 0 and n != 0:
                    inter_union[m-1, n-1, 0] += 1
        ious = inter_union[:, :, 0]/(
               inter_union[:, :, 1]-inter_union[:, :, 0]+smooth)
        for t in np.arange(0.5, 1.0, 0.05):
            cnt += 1
            tp = 0
            fp = 0
            fn = 0
            fn_tests = np.ones(n_masks_true, dtype=np.bool)
            for m in range(n_masks_pred):
                fp_test = True
                for n in range(n_masks_true):
                    if ious[m, n] > t:
                        tp += 1
                        fp_test = False
                        fn_tests[n] = False
                if fp_test:
                    fp += 1
            fn = np.count_nonzero(fn_tests)
            try:
                score += tp/(tp+fp+fn)
            except ZeroDivisionError:
                pass
        score = score/cnt
        scores[i] = score
    return torch.tensor(scores).mean()


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


def to_rles(x):
    n = np.max(x)
    for i in range(1, n + 1):
        yield rle_encoding(x == i)


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


trainloader, valloader = load_train_data(
    TRAIN_PATH, normalize=((MEAN, MEAN, MEAN), (STD, STD, STD)))
testset = CellsDataset(
    TEST_PATH, test_ids, height=1388, width=1388, train=False, erosion=True,
    crop=False, resize=False, aug=False, pad=True,
    normalize=((MEAN, MEAN, MEAN),
               (STD, STD, STD)))
testloader = DataLoader(testset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=0)
db = ImageDataBunch(trainloader, valloader,
                    test_dl=testloader, device='cuda:0')

learner = unet_learner(db, mod.resnet34, pretrained=False, metrics=[
                       mean_iou], loss_func=nn.BCEWithLogitsLoss(), wd=0.1)
learner.fit_one_cycle(
    100, 2e-4,
    callbacks=[
        SaveModelCallback(
            learner, monitor='mean_iou', name='bestmodel4')])

preds, sizes = predict_TTA_all(learner)
create_submission(preds, sizes, test_ids)
