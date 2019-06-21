import os
import random
import warnings

import numpy as np

from fastai.vision.learner import unet_learner
import fastai.vision.models as mod

import torch
import torch.nn as nn
from modules.dataset import CellsDataset, load_data
from modules.preds import predict_TTA_all, create_submission
from modules.metrics import mean_iou
import config as cfg


def run(model):
    models = {
        'resnet34': mod.resnet34, 'resnet50': mod.resnet50,
        'resnet101': mod.resnet101}

    warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
    seed = 42
    random.seed = seed
    np.random.seed = seed

    test_ids = next(os.walk(cfg.TEST_PATH))[1]

    testset = CellsDataset(
        cfg.TEST_PATH, test_ids, height=cfg.MAX_HEIGHT, width=cfg.MAX_WIDTH,
        train=False, erosion=True, crop=False, resize=False, aug=False,
        pad=True, normalize=(cfg.MEAN, cfg.STD))

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    db = load_data(
        cfg.TRAIN_PATH, size=cfg.TRAIN_WIDTH, bs=cfg.BATCH_SIZE,
        testset=testset, normalize=(cfg.MEAN, cfg.STD))

    learner = unet_learner(
        db, models[cfg.MODEL],
        pretrained=False, metrics=[mean_iou],
        loss_func=nn.BCEWithLogitsLoss(),
        wd=cfg.WD, model_dir=cfg.MODELS_PATH)

    learner.load(model)

    preds, sizes = predict_TTA_all(
        learner, size=(cfg.TEST_HEIGHT, cfg.TEST_WIDTH),
        overlap=cfg.TEST_OVERLAP, device=device)
    create_submission(preds, sizes, test_ids, folder=cfg.SUB_PATH)
