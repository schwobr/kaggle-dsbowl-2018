import random
import warnings

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from modules.dataset import load_test_data
from modules.preds import create_submission
from modules.nets import Net
from modules.files import get_sizes
from modules.metrics import mean_iou
from modules.transforms import get_test_tfms
from modules.model_factory import get_model
import config as cfg


def run(model):
    warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
    seed = 42
    random.seed = seed
    np.random.seed = seed

    testloader = load_test_data(
        cfg.TEST_PATH, size=cfg.MAX_SIZE, bs=cfg.BATCH_SIZE,
        transforms=get_test_tfms(cfg.MAX_SIZE))

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    mod = get_model(cfg.MODEL, cfg.CLASSES, act=cfg.ACT)
    opt = optim.Adam(mod.parameters(), lr=cfg.LR, weight_decay=cfg.WD)
    net = Net(mod, opt, nn.BCELoss(), [mean_iou], cfg.MODELS_PATH)

    net.load(model)

    ids = testloader.dataset.ids
    sizes = get_sizes(cfg.TEST_CSV, ids)

    preds = net.predict(
        testloader, device, sizes, TTA=True, size=cfg.TEST_SIZE,
        overlap=cfg.TEST_OVERLAP)

    create_submission(preds, sizes, ids, folder=cfg.SUB_PATH)
