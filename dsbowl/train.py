import random
import warnings

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from modules.dataset import load_train_data, load_test_data
from modules.preds import create_submission
from modules.metrics import mean_iou
from modules.transforms import get_train_tfms, get_test_tfms
from modules.nets import Net, OneCycleScheduler
from modules.model_factory import get_model
from modules.files import get_sizes, getNextFilePath
import config as cfg


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
        cfg.TEST_PATH, size=cfg.MAX_SIZE, bs=cfg.BATCH_SIZE,
        transforms=get_test_tfms(cfg.TRAIN_SIZE))

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    mod = get_model(cfg.MODEL, cfg.CLASSES, act=cfg.ACT)
    opt = optim.Adam(mod.parameters(), lr=cfg.LR, weight_decay=cfg.WD)
    net = Net(mod, opt, nn.BCELoss(), [mean_iou], cfg.MODELS_PATH)

    save_name = f'{cfg.MODEL}_{cfg.EPOCHS}_{cfg.LR}_{cfg.WD}'
    save_name += f'_{getNextFilePath(cfg.MODELS_PATH, save_name)}'

    writer = SummaryWriter(log_dir=cfg.LOG/save_name)
    writer.add_graph(
        mod, input_to_model=next(iter(trainloader))[0],
        operator_export_type='RAW')
    scheduler = OneCycleScheduler(cfg.LR, len(trainloader))

    mod = net.fit(dls, cfg.EPOCHS, save_name, device,
                  writer=writer, scheduler=scheduler)

    ids = testloader.dataset.ids
    sizes = get_sizes(cfg.TEST_CSV, ids)

    preds = net.predict(
        testloader, device, sizes, TTA=True, size=cfg.TEST_SIZE,
        overlap=cfg.TEST_OVERLAP)

    create_submission(preds, sizes, ids, folder=cfg.SUB_PATH)
    writer.close()
