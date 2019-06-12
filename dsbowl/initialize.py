import config as cfg
from modules.files import create_csv


def run():
    if not cfg.TRAIN_CSV.is_file():
        create_csv(cfg.TRAIN_PATH, cfg.PROJECT_PATH)
    if not cfg.TEST_CSV.is_file():
        create_csv(cfg.TEST_PATH, cfg.PROJECT_PATH)
