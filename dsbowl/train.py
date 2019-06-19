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
        transforms=get_test_tfms(cfg.MAX_SIZE))

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    mod = get_model(cfg.MODEL, cfg.CLASSES, act=cfg.ACT,
                    pretrained=cfg.PRETRAINED, fastai=True)
    opt = optim.Adam(mod.parameters(), lr=cfg.LRS[-1], weight_decay=cfg.WD)
    net = Net(mod, opt, nn.BCELoss(), [mean_iou], cfg.MODELS_PATH)

    if cfg.GROUP_LIMITS is not None:
        param_groups = []
        assert len(cfg.GROUP_LIMITS)+1 == len(cfg.LRS),\
            "You must specify a learning rate for each group"
        groups = net.get_groups(cfg.GROUP_LIMITS)
        for lr, group in zip(cfg.LRS, groups):
            param_groups.append({'params': group, 'lr': lr})
        opt = optim.Adam(param_groups, weight_decay=cfg.WD)
        mod.opt = opt

    save_name = f'{cfg.MODEL}_fastai_{cfg.EPOCHS}_{cfg.LRS[-1]}_{cfg.WD}'
    save_name += f'_{getNextFilePath(cfg.MODELS_PATH, save_name)}'

    writer = SummaryWriter(log_dir=cfg.LOG/save_name)
    writer.add_graph(
        mod, input_to_model=next(iter(trainloader))[0],
        operator_export_type='RAW')

    if cfg.FREEZE_UNTIL is not None:
        net.freeze(cfg.FREEZE_UNTIL)

    scheduler = OneCycleScheduler(cfg.LRS, len(trainloader))

    mod = net.fit(dls, cfg.EPOCHS, save_name, device,
                  writer=writer, scheduler=scheduler)

    if cfg.UNFROZE_EPOCHS is not None:
        scheduler = OneCycleScheduler(cfg.LRS, len(trainloader))
        net.unfreeze()
        mod = net.fit(dls, cfg.UNFROZE_EPOCHS, save_name, device,
                      writer=writer, scheduler=scheduler, frozen=cfg.EPOCHS)

    ids = testloader.dataset.ids
    sizes = get_sizes(cfg.TEST_CSV, ids)

    preds = net.predict(
        testloader, device, sizes, TTA=True, size=cfg.TEST_SIZE,
        overlap=cfg.TEST_OVERLAP)

    create_submission(preds, sizes, ids, folder=cfg.SUB_PATH)
    writer.close()
