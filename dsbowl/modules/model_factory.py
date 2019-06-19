from modules.unets import Unet, UnetFastAI
from torchvision.models import (
    resnet18, resnet34, resnet50, resnet101, resnet152)


def get_model(model, n_classes, pretrained=False, act='sigmoid', fastai=True):
    if model == 'resnet18':
        encoder = resnet18(pretrained=pretrained)
    elif model == 'resnet34':
        encoder = resnet34(pretrained=pretrained)
    elif model == 'resnet50':
        encoder = resnet50(pretrained=pretrained)
    elif model == 'resnet101':
        encoder = resnet101(pretrained=pretrained)
    elif model == 'resnet152':
        encoder = resnet152(pretrained=pretrained)
    else:
        raise ValueError('Wrong model name used')
    if fastai:
        return UnetFastAI(encoder, n_classes, act=act)
    else:
        return Unet(encoder, n_classes, act=act)
