import math
import warnings

import torch
import torch.nn as nn

# from .modules import IBN


__all__ = ['ResNet_IBN', 'resnet50_ibn_a', 'resnet101_ibn_a', 'resnet152_ibn_a',
           'resnet50_ibn_b', 'resnet101_ibn_b', 'resnet152_ibn_b']


# model_urls = {
#     'resnet50_ibn_a': './logs/pretrained/resnet50_ibn_a.pth',
#     'resnet101_ibn_a': './logs/pretrained/resnet101_ibn_a.pth',
#     'resnet50_ibn_b': './logs/pretrained/resnet50_ibn_b.pth',
#     'resnet101_ibn_b': './logs/pretrained/resnet101_ibn_b.pth',
# }
model_urls = {
    'resnet50_ibn_a': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_a-d9d0bb7b.pth',
    'resnet101_ibn_a': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet101_ibn_a-59ea0ac6.pth',
    'resnet50_ibn_b': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_b-9ca61e85.pth',
    'resnet101_ibn_b': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet101_ibn_b-c55f6dba.pth',
}


class IBN(nn.Module):
    r"""Instance-Batch Normalization layer from
    `"Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"
    <https://arxiv.org/pdf/1807.09441.pdf>`
    Args:
        planes (int): Number of channels for the input tensor
        ratio (float): Ratio of instance normalization in the IBN layer
    """
    def __init__(self, planes, ratio=0.5):
        super(IBN, self).__init__()
        self.half = int(planes * ratio)
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = nn.BatchNorm2d(planes - self.half)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, int(channel/reduction), bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(int(channel/reduction), channel, bias=False),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
        

class Bottleneck_IBN(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, ibn=None, stride=1, downsample=None):
        super(Bottleneck_IBN, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if ibn == 'a':
            self.bn1 = IBN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.IN = nn.InstanceNorm2d(planes * 4, affine=True) if ibn == 'b' else None
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.IN is not None:
            out = self.IN(out)
        out = self.relu(out)

        return out


class ResNet_IBN(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 ibn_cfg=('a', 'a', 'a', None),
                 num_classes=1000):
        self.inplanes = 64
        super(ResNet_IBN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        if ibn_cfg[0] == 'b':
            self.bn1 = nn.InstanceNorm2d(64, affine=True)
        else:
            self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], ibn=ibn_cfg[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, ibn=ibn_cfg[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, ibn=ibn_cfg[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, ibn=ibn_cfg[3])
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, ibn=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes,
                            None if ibn == 'b' else ibn,
                            stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                None if (ibn == 'b' and i < blocks-1) else ibn))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet50_ibn_a(pretrained=False, **kwargs):
    """Constructs a ResNet-50-IBN-a model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(block=Bottleneck_IBN,
                       layers=[3, 4, 6, 3],
                       ibn_cfg=('a', 'a', 'a', None),
                       **kwargs)
    if pretrained:
        # state_dict = torch.load(model_urls['resnet50_ibn_a'], map_location=torch.device('cpu'))['state_dict']
        # state_dict = remove_module_key(state_dict)
        # model.load_state_dict(state_dict)
        model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['resnet50_ibn_a']))
    return model


def resnet101_ibn_a(pretrained=False, **kwargs):
    """Constructs a ResNet-101-IBN-a model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(block=Bottleneck_IBN,
                       layers=[3, 4, 23, 3],
                       ibn_cfg=('a', 'a', 'a', None),
                       **kwargs)
    if pretrained:
        # state_dict = torch.load(model_urls['resnet101_ibn_a'], map_location=torch.device('cpu'))['state_dict']
        # state_dict = remove_module_key(state_dict)
        # model.load_state_dict(state_dict)
        model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['resnet101_ibn_a']))
    return model


def resnet152_ibn_a(pretrained=False, **kwargs):
    """Constructs a ResNet-152-IBN-a model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(block=Bottleneck_IBN,
                       layers=[3, 8, 36, 3],
                       ibn_cfg=('a', 'a', 'a', None),
                       **kwargs)
    if pretrained:
        warnings.warn("Pretrained model not available for ResNet-152-IBN-a!")
    return model


def resnet50_ibn_b(pretrained=False, **kwargs):
    """Constructs a ResNet-50-IBN-b model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(block=Bottleneck_IBN,
                       layers=[3, 4, 6, 3],
                       ibn_cfg=('b', 'b', None, None),
                       **kwargs)
    if pretrained:
        # state_dict = torch.load(model_urls['resnet50_ibn_b'], map_location=torch.device('cpu'))['state_dict']
        # state_dict = remove_module_key(state_dict)
        # model.load_state_dict(state_dict)
        model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['resnet50_ibn_b']))
    return model


def resnet101_ibn_b(pretrained=False, **kwargs):
    """Constructs a ResNet-101-IBN-b model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(block=Bottleneck_IBN,
                       layers=[3, 4, 23, 3],
                       ibn_cfg=('b', 'b', None, None),
                       **kwargs)
    if pretrained:
        # state_dict = torch.load(model_urls['resnet101_ibn_b'], map_location=torch.device('cpu'))['state_dict']
        # state_dict = remove_module_key(state_dict)
        # model.load_state_dict(state_dict)
        model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['resnet101_ibn_b']))
    return model


def resnet152_ibn_b(pretrained=False, **kwargs):
    """Constructs a ResNet-152-IBN-b model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(block=Bottleneck_IBN,
                       layers=[3, 8, 36, 3],
                       ibn_cfg=('b', 'b', None, None),
                       **kwargs)
    if pretrained:
        warnings.warn("Pretrained model not available for ResNet-152-IBN-b!")
    return model
