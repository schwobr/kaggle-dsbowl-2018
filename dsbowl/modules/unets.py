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
            self, in_channels, mid_channels, out_channels, kernel_size,
            stride=1, padding=0, bias=True, **kwargs):
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
            self, in_channels, mid_channels, out_channels, kernel_size,
            stride=1, padding=0, bias=True, **kwargs):
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


class Decoder(nn.Module):
    def __init__(self, sizes):
        super(Decoder, self).__init__()
        self.upconvs = nn.ModuleList(
            [UpConv(size, 256, 1) for size in sizes[:-1]])

        self.doubleconvs = nn.ModuleList([
            DoubleConv(
                256, 128, 3, padding=1, scale_factor=2 ** n)
            for n in range(3, 0, -1)])
        self.doubleconvs.append(DoubleConv(256, 128, 3, padding=1))

        self.aggregate = ConvBnRelu(512, 256, 3, padding=1)
        self.decode = DecoderBlock(256, 128 + sizes[-1], 128, 3, padding=1)
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.doubleconv = DoubleConv(128, 64, 3, padding=1)

    def forward(self, x, outputs):
        ps = []
        p = None
        out0 = outputs.pop()
        for k, out in enumerate(outputs):
            upconv = self.upconvs[k]
            p = upconv(out, p)
            doubleconv = self.doubleconvs[k]
            ps.append(doubleconv(p))
        x = torch.cat(ps, dim=1)
        x = self.aggregate(x)
        x = self.decode(x, out0)
        x = self.up(x)
        x = self.doubleconv(x)
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

        layers = ['conv1', 'relu']+[f'layer{k+1}' for k in range(4)]
        sizes = []
        for layer in layers:
            for name, module in self.encoder.named_children():
                if name == layer:
                    if name != 'conv1':
                        module.register_forward_hook(hook)
                    if name != 'relu':
                        sizes.append(list(module.parameters())[-1].size(0))
                    break
        self.decoder = Decoder(sizes[::-1])
        self.final_conv = nn.Sequential(
            ConvBnRelu(67, 67, 3, padding=1),
            ConvBnRelu(67, 67, 3, padding=1))
        self.activation = get_activation(act, 67, n_classes)

    def forward(self, x):
        y = self.encoder(x)
        y = self.decoder(y, self.outputs[::-1])
        x = torch.cat([x, y], dim=1)
        y = self.final_conv(x)
        x += y
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
