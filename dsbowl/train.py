import random
import warnings

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from modules.dataset import load_train_data, load_test_data
from modules.preds import create_submission
from modules.metrics import mean_iou
from modules.utils import getNextFilePath
from modules.transforms import get_train_tfms
from modules.nets import Net, OneCycleScheduler
from modules.model_factory import get_model
import dsbowl.config as cfg


def run():
    warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
    seed = 42
    random.seed = seed
    np.random.seed = seed

    trainloader, valloader = load_train_data(
        cfg.TRAIN_PATH, size=cfg.TRAIN_SIZE, bs=cfg.BATCH_SIZE,
        transforms=get_train_tfms(cfg.TRAIN_SIZE))

    dls = {'train': trainloader, 'val': valloader}
    testloader = load_test_data(
        cfg.TEST_PATH, size=cfg.TEST_SIZE, bs=cfg.BATCH_SIZE)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    mod = get_model(cfg.MODEL, cfg.CLASSES, act=cfg.ACT)
    opt = optim.Adam(mod.parameters(), lr=cfg.LR, weight_decay=cfg.WD)
    net = Net(mod, opt, nn.BCELoss, [mean_iou], cfg.MODELS_PATH)

    save_name = f'{cfg.MODEL}_{cfg.EPOCHS}_'
    save_name += f'{cfg.LR}_{cfg.WD}_{getNextFilePath(cfg.MODELS_PATH)}'

    scheduler = OneCycleScheduler(cfg.LR, len(trainloader), bs=cfg.BATCH_SIZE)

    mod = net.fit(dls, cfg.EPOCHS, save_name, device, scheduler=scheduler)

    preds = net.predict(testloader, device)
    # TODO: implement a get_sizes function to read them directly from csv
    sizes = []
    create_submission(preds, sizes, testloader.dataset.ids,
                      folder=cfg.SUB_PATH)
