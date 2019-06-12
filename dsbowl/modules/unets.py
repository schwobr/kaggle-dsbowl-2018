import torch
import torch.nn as nn
from torchvision.models import resnet


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True, eps=1e-5, momentum=0.01, **kwargs):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, bias=bias, **kwargs)
        self.bn = nn.BatchNorm2d(
            out_channels, eps=eps, momentum=momentum, **kwargs)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvRelu(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, padding=0,
            bias=True, **kwargs):
        super(ConvRelu, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, bias=bias, **kwargs)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(
            self, in_channels, mid_channels, out_channels, kernel_size, stride=1,
            padding=0, bias=True, **kwargs):
        super(DecoderBlock, self).__init__()
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv1 = ConvRelu(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, bias=bias, **kwargs)
        self.conv2 = ConvRelu(
            mid_channels, out_channels, kernel_size, stride=stride,
            padding=padding, bias=bias, **kwargs)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.conv1(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv2(x)
        return x


class DecoderBlockBn(nn.Module):
    def __init__(
            self, in_channels, mid_channels, out_channels, kernel_size, stride=1,
            padding=0, bias=True, **kwargs):
        super(DecoderBlock, self).__init__()
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv1 = ConvBnRelu(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, bias=bias, **kwargs)
        self.conv2 = ConvBnRelu(
            mid_channels, out_channels, kernel_size, stride=stride,
            padding=padding, bias=bias, **kwargs)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.conv1(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv2(x)
        return x


class DoubleConv(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1,
            padding=0, bias=True, scale_factor=None, **kwargs):
        super(DoubleConv, self).__init__()
        self.conv1 = ConvRelu(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, bias=bias, **kwargs)
        self.conv2 = ConvRelu(
            out_channels, out_channels, kernel_size, stride=stride,
            padding=padding, bias=bias, **kwargs)
        self.up = nn.UpsamplingNearest2d(
            scale_factor=scale_factor) if scale_factor is not None else None

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        if self.up is not None:
            x = self.up(x)
        return x


class UpConv(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1,
            padding=0, bias=True, scale_factor=2, **kwargs):
        super(UpConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, bias=bias, **kwargs)
        self.up = nn.UpsamplingNearest2d(scale_factor=scale_factor)

    def forward(self, x, skip=None):
        x = self.conv(x)
        if skip is not None:
            x += self.up(skip)
        return x


class Unet(nn.Module):
    def __init__(self, encoder, n_classes, act='sigmoid'):
        super(Unet, self).__init__()
        if not isinstance(encoder, resnet.ResNet):
            raise ValueError('Encoder should be a resnet')
        self.encoder = encoder
        self.n_classes = n_classes
        self.outputs = []

        def hook(module, input, output):
            self.outputs.append(output)

        self.encoder.relu.register_forward_hook(hook)
        self.encoder.layer1.register_forward_hook(hook)
        self.encoder.layer2.register_forward_hook(hook)
        self.encoder.layer3.register_forward_hook(hook)
        self.encoder.layer4.register_forward_hook(hook)
            
        relu_param = list(encoder.conv1.parameters())[-1]
        layer1_param = list(encoder.layer1.parameters())[-1]
        layer2_param = list(encoder.layer2.parameters())[-1]
        layer3_param = list(encoder.layer3.parameters())[-1]
        layer4_param = list(encoder.layer4.parameters())[-1]
        self.upconvs = nn.ModuleList(
            [UpConv(relu_param.size(0), 256, 1),
             UpConv(layer1_param.size(0), 256, 1),
             UpConv(layer2_param.size(0), 256, 1),
             UpConv(layer3_param.size(0), 256, 1),
             UpConv(layer4_param.size(0), 256, 1)][:: -1])
        
        self.doubleconvs = nn.ModuleList([
            DoubleConv(
                256, 128, 3, padding=1, scale_factor=2 ** n)
            for n in range(3, 0, -1)])
        self.doubleconvs.append(DoubleConv(256, 128, 3, padding=1))

        self.aggregate = ConvBnRelu(512, 256, 3, padding=1)
        self.decode = DecoderBlock(256, 128+relu_param.size(0), 128, 3, padding=1)
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.doubleconv = DoubleConv(128, 64, 3, padding=1)
        self.activation = get_activation(act, 64, n_classes)

    def forward(self, x):
        x = self.encoder(x)
        ps = []
        p = None
        for k, out in enumerate(self.outputs[::-1]):
            upconv = self.upconvs[k]
            p = upconv(out, p)
            if k < 4:
                doubleconv = self.doubleconvs[k]
                ps.append(doubleconv(p))
        x = torch.cat(ps, dim=1)
        x = self.aggregate(x)
        x = self.decode(x, self.outputs[0])
        x = self.up(x)
        x = self.doubleconv(x)
        x = self.activation(x)
        self.outputs = []
        return x


def get_activation(act, in_channels, out_channels):
    conv = nn.Conv2d(in_channels, out_channels, 1)
    act = act.lower()
    if act == 'softmax':
        activation = nn.Softmax()
    elif act == 'sigmoid':
        activation = nn.Sigmoid()
    else:
        raise ValueError('Invalid activation function')
    return nn.Sequential(conv, activation)
