from fastai.basic_train import Learner
from fastai.vision.learner import unet_learner


class UnetLearner(Learner):
    def __new__(self, *args, **kwargs):
        return unet_learner(*args, **kwargs)
