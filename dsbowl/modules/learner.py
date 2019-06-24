from fastai.basic_train import Learner
from fastai.vision.learner import unet_learner

from modules.preds import predict_TTA_all, predict_all


class UnetLearner(Learner):
    def __init__(self, *args, **kwargs):
        learner = unet_learner(*args, **kwargs)
        for v in vars(learner):
            setattr(self, v, getattr(learner, v))

    def predict_all(self, sizes, TTA=True, **kwargs):
        if TTA:
            return predict_TTA_all(self, sizes, **kwargs)
        else:
            return predict_all(self)
