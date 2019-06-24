from fastai.basic_train import Learner
from fastai.vision.learner import unet_learner


class UnetLearner(Learner):
    def __init__(self, *args, **kwargs):
        learner = unet_learner(*args, **kwargs)
        for v in vars(learner):
            setattr(self, v, getattr(learner, v))
