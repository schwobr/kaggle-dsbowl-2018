from pathlib import Path

# IMAGE SIZES
TRAIN_SIZE = 224
MAX_SIZE = 1388
TEST_SIZE = 224
TEST_OVERLAP = 64
IMG_CHANNELS = 3

# PATHS
PROJECT_PATH = Path(
    '/work/stages/schwob/data-science-bowl-2018/kaggle-dsbowl-2018/')
TRAIN_PATH = PROJECT_PATH/'data/stage1_train/'
TEST_PATH = PROJECT_PATH/'data/stage2_test_final/'
MODELS_PATH = PROJECT_PATH/'models/'
SUB_PATH = PROJECT_PATH/'submissions/'
TRAIN_CSV = PROJECT_PATH/'stage1_train.csv'
TEST_CSV = PROJECT_PATH/'stage2_test_final.csv'
LOG = Path('/work/stages/schwob/runs')

# NORMALIZE
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)

# LEARNER CONFIG
BATCH_SIZE = 8
WD = 0.1
LRS = [2e-4]
GROUP_LIMITS = None
FREEZE_UNTIL = 'encoder.layer4'
EPOCHS = 20
UNFROZE_EPOCHS = 20
PRETRAINED = True
MODEL = 'resnet101'
CLASSES = 1
ACT = 'sigmoid'
