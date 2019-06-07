from modules.files import create_csv
import config as cfg


def run():
    create_csv(cfg.TRAIN_PATH, cfg.PROJECT_PATH)
    create_csv(cfg.TEST_PATH, cfg.PROJECT_PATH)


if __name__ == '__main__':
    run()
