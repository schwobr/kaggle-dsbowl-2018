from fastai.basic_train import Learner
from fastai.vision.learner import unet_learner


class UnetLearner(Learner):
    def __new__(cls, *args, **kwargs):
        learner = unet_learner(*args, **kwargs)
        new_learner = super(UnetLearner, cls).__new__(cls)
        for v in vars(learner):
            setattr(new_learner, v, getattr(learner, v))
        return new_learner
