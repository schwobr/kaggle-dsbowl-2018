import os
import random
import warnings

import numpy as np

from fastai.vision.data import ImageDataBunch
from fastai.vision.learner import unet_learner
import fastai.vision.models as mod
from fastai.callbacks import SaveModelCallback

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dsbowl.modules.dataset import CellsDataset, load_train_data
from dsbowl.modules.preds import predict_TTA_all, create_submission
from dsbowl.modules.metrics import mean_iou
from dsbowl.modules.utils import getNextFilePath
import dsbowl.config as cfg


def run():
    # Models definition
    models = {
        'resnet34': mod.resnet34, 'resnet50': mod.resnet50,
        'resnet101': mod.resnet101}

    warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
    seed = 42
    random.seed = seed
    np.random.seed = seed

    test_ids = next(os.walk(cfg.TEST_PATH))[1]

    trainloader, valloader = load_train_data(
        cfg.TRAIN_PATH, height=cfg.TRAIN_HEIGHT, width=cfg.TRAIN_WIDTH,
        bs=cfg.BATCH_SIZE)
    testset = CellsDataset(
        cfg.TEST_PATH, test_ids, height=cfg.MAX_HEIGHT, width=cfg.MAX_WIDTH,
        train=False, erosion=True, crop=False, resize=False, aug=False,
        pad=True, normalize=(cfg.MEAN, cfg.STD))
    testloader = DataLoader(testset, batch_size=cfg.BATCH_SIZE,
                            shuffle=False, num_workers=0)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    db = ImageDataBunch(trainloader, valloader,
                        test_dl=testloader, device=device)

    learner = unet_learner(
        db, models[cfg.MODEL],
        pretrained=False, metrics=[mean_iou],
        loss_func=nn.BCEWithLogitsLoss(),
        wd=cfg.WD, model_dir=cfg.MODELS_PATH)

    save_name = f'{cfg.MODEL}_{cfg.EPOCHS}_'
    save_name += f'{cfg.LR}_{cfg.WD}_{getNextFilePath(cfg.MODELS_PATH)}'

    learner.fit_one_cycle(
        cfg.EPOCHS, cfg.LR,
        callbacks=[
            SaveModelCallback(
                learner, monitor='mean_iou', name=save_name)])

    preds, sizes = predict_TTA_all(
        learner, size=(cfg.TEST_HEIGHT, cfg.TEST_WIDTH),
        overlap=cfg.TEST_OVERLAP, device=device)
    create_submission(preds, sizes, test_ids, folder=cfg.SUB_PATH)
