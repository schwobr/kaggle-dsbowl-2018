import os
import random
import warnings
import numpy as np

import fastai.vision.models as mod
from fastai.callbacks import SaveModelCallback

import torch.nn as nn

from modules.learner import UnetLearner
from modules.dataset import load_data
from modules.preds import create_submission
from modules.metrics import mean_iou
from modules.files import getNextFilePath, get_sizes
import config as cfg


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

    db = load_data(
        cfg.TRAIN_PATH, size=cfg.TRAIN_WIDTH, bs=cfg.BATCH_SIZE,
        classes=cfg.CLASSES, testpath=cfg.TEST_PATH, max_size=cfg.MAX_SIZE)

    if len(cfg.CLASSES) < 3:
        loss_func = nn.BCEWithLogitsLoss()
    else:
        loss_func = nn.CrossEntropyLoss()

    learner = UnetLearner(
        db, models[cfg.MODEL],
        pretrained=cfg.PRETRAINED, metrics=[mean_iou],
        loss_func=loss_func, wd=cfg.WD, model_dir=cfg.MODELS_PATH)

    save_name = f'{cfg.MODEL}_fastai_{cfg.EPOCHS}_{cfg.LR}_{cfg.WD}'
    save_name += f'_{getNextFilePath(cfg.MODELS_PATH, save_name)}'

    learner.fit_one_cycle(
        cfg.EPOCHS, slice(cfg.LR),
        callbacks=[
            SaveModelCallback(
                learner, monitor='mean_iou', name=save_name)])

    if cfg.UNFROZE_EPOCHS:
        learner.fit_one_cycle(
            cfg.UNFROZE_EPOCHS, slice(cfg.LR),
            callbacks=[
                SaveModelCallback(
                    learner, monitor='mean_iou', name=save_name)])

    sizes = get_sizes(cfg.TEST_CSV, test_ids)
    learner.load(save_name)

    preds = learner.predict_all(
        sizes, size=cfg.TEST_SIZE, overlap=cfg.TEST_OVERLAP,
        out_channels=len(cfg.CLASSES))

    create_submission(preds, sizes, test_ids, folder=cfg.SUB_PATH)
