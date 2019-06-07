from pathlib import Path

# IMAGE SIZES
TRAIN_WIDTH = 256
TRAIN_HEIGHT = 256
MAX_WIDTH = 1388
MAX_HEIGHT = 1388
TEST_HEIGHT = 256
TEST_WIDTH = 256
TEST_OVERLAP = 64
IMG_CHANNELS = 3

# PATHS
PROJECT_PATH = Path(
    '/work/stages/schwob/data-science-bowl-2018/kaggle-dsbowl-2018/')
TRAIN_PATH = PROJECT_PATH/'data/stage1_train/'
TEST_PATH = PROJECT_PATH/'data/stage2_test_final/'
MODELS_PATH = PROJECT_PATH/'models/'
SUB_PATH = PROJECT_PATH/'submissions/'

# NORMALIZE
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)

# LEARNER CONFIG
BATCH_SIZE = 4
WD = 0.1
LR = 2e-4
EPOCHS = 100
MODEL = "resnet34"
