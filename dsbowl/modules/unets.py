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


class DoubleConvFastAI(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size, mid_channels=None,
            scale_factor=2, stride=1, padding=0, bias=True, **kwargs):
        super(DoubleConvFastAI, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels
        self.relu = nn.ReLU()
        self.conv1 = ConvRelu(
            in_channels, mid_channels, kernel_size, stride=stride,
            padding=padding, bias=bias, **kwargs)
        self.conv2 = ConvRelu(
            mid_channels, in_channels, kernel_size, stride=stride,
            padding=padding, bias=bias, **kwargs)
        self.up = PixelShuffleICNR(
            in_channels, out_channels, scale_factor=scale_factor, **kwargs)

    def forward(self, x):
        x = self.relu(x)
        x = self.conv1(x)
        x = self.conv2(x)
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


class PixelShuffleICNR(nn.Module):
    def __init__(
            self, in_channels, out_channels, bias=True, scale_factor=2, **
            kwargs):
        super(PixelShuffleICNR, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels*scale_factor**2, 1, bias=bias, **kwargs)
        icnr(self.conv.weight)
        self.shuf = nn.PixelShuffle(scale_factor)
        self.pad = nn.ReflectionPad2d((1, 0, 1, 0))
        self.blur = nn.AvgPool2d(2, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.shuf(x)
        x = self.pad(x)
        x = self.blur(x)
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

    def forward(self, outputs):
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


class DecoderFastAI(nn.Module):
    def __init__(self, sizes, **kwargs):
        super(DecoderFastAI, self).__init__()
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(size, **kwargs) for size in sizes] +
            [nn.Identity()])
        doubleconvs = []
        cur_channels = 0
        for size in sizes[:-1]:
            doubleconvs.append(
                DoubleConvFastAI(
                    size + cur_channels,
                    (size + cur_channels) // 2, 3, padding=1))
            cur_channels = (size+cur_channels)//2
        doubleconvs.append(
            DoubleConv(
                sizes[-1] + cur_channels, sizes[-1] +
                cur_channels, 3, padding=1))
        self.doubleconvs = nn.ModuleList(doubleconvs)

    def forward(self, x, outputs):
        p = None
        for bn, doubleconv, out in zip(
                self.bns, self.doubleconvs, outputs + [x]):
            out = bn(out)
            if p is not None:
                out = torch.cat([out, p], dim=1)
            p = doubleconv(p)
        return out + p


class UnetFastAI(nn.Module):
    def __init__(self, encoder, n_classes, act='sigmoid'):
        super(UnetFastAI, self).__init__()
        if not isinstance(encoder, resnet.ResNet):
            raise ValueError('Encoder should be a resnet')
        self.encoder = encoder
        self.n_classes = n_classes
        self.outputs = []

        def hook(module, input, output):
            self.outputs.append(output.detach())

        layers = ['conv1', 'relu']+[f'layer{k+1}' for k in range(4)]
        sizes = [3]
        for layer in layers:
            for name, module in self.encoder.named_children():
                if name == layer:
                    if name != 'conv1':
                        module.register_forward_hook(hook)
                    if name != 'relu':
                        sizes.append(list(module.parameters())[-1].size(0))
                    break
        self.decoder = DecoderFastAI(sizes[::-1])
        n_channels = list(self.decoder.parameters())[-1].size(0)
        self.activation = get_activation(act, n_channels, n_classes)

    def forward(self, x):
        self.encoder(x)
        x = self.decoder(x, self.outputs[::-1])
        x = self.activation(x)
        self.outputs = []
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
            self.outputs.append(output.detach())

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
        x = x+y
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


def icnr(x, scale=2, init=nn.init.kaiming_normal_):
    ni, nf, h, w = x.shape
    ni2 = int(ni/(scale**2))
    k = init(torch.zeros([ni2, nf, h, w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    k = k.contiguous().view([nf, ni, h, w]).transpose(0, 1)
    x.data.copy_(k)
