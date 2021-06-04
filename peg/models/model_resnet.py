import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, ReLU, Dropout, MaxPool2d, Sequential, Module
from torch.nn import functional as F


# # Support: ['ResNet_50', 'ResNet_101', 'ResNet_152']


# def conv3x3(in_planes, out_planes, stride = 1):
#     """3x3 convolution with padding"""

#     return Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride,
#                      padding = 1, bias = False)


# def conv1x1(in_planes, out_planes, stride = 1):
#     """1x1 convolution"""

#     return Conv2d(in_planes, out_planes, kernel_size = 1, stride = stride, bias = False)


# class BasicBlock(Module):
#     expansion = 1

#     def __init__(self, inplanes, planes, stride = 1, downsample = None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = BatchNorm2d(planes)
#         self.relu = ReLU(inplace = True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = BatchNorm2d(planes)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         out = self.relu(out)

#         return out


# class Bottleneck(Module):
#     expansion = 4

#     def __init__(self, inplanes, planes, stride = 1, downsample = None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = conv1x1(inplanes, planes)
#         self.bn1 = BatchNorm2d(planes)
#         self.conv2 = conv3x3(planes, planes, stride)
#         self.bn2 = BatchNorm2d(planes)
#         self.conv3 = conv1x1(planes, planes * self.expansion)
#         self.bn3 = BatchNorm2d(planes * self.expansion)
#         self.relu = ReLU(inplace = True)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         out = self.relu(out)

#         return out


# class ResNet(Module):

#     def __init__(self, input_size, block, layers, zero_init_residual = True):
#         super(ResNet, self).__init__()
#         assert input_size[0] in [112, 224], "input_size should be [112, 112] or [224, 224]"
#         self.inplanes = 64
#         self.conv1 = Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
#         self.bn1 = BatchNorm2d(64)
#         self.relu = ReLU(inplace = True)
#         self.maxpool = MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride = 2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride = 2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride = 2)

#         self.bn_o1 = BatchNorm2d(2048)
#         self.dropout = Dropout()
#         if input_size[0] == 112:
#             self.fc = Linear(2048 * 4 * 4, 512)
#         else:
#             self.fc = Linear(2048 * 8 * 8, 512)
#         self.bn_o2 = BatchNorm1d(512)

#         for m in self.modules():
#             if isinstance(m, Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
#             elif isinstance(m, BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#         # Zero-initialize the last BN in each residual branch,
#         # so that the residual branch starts with zeros, and each residual block behaves like an identity.
#         # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
#         if zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, Bottleneck):
#                     nn.init.constant_(m.bn3.weight, 0)
#                 elif isinstance(m, BasicBlock):
#                     nn.init.constant_(m.bn2.weight, 0)

#     def _make_layer(self, block, planes, blocks, stride = 1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 BatchNorm2d(planes * block.expansion),
#             )

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes))

#         return Sequential(*layers)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.bn_o1(x)
#         x = self.dropout(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         x = self.bn_o2(x)
#         if self.training is False:
#             return F.normalize(x)
#         return F.normalize(x), F.normalize(x)


# def ResNet_50(input_size, **kwargs):
#     """Constructs a ResNet-50 model.
#     """
#     model = ResNet(input_size, Bottleneck, [3, 4, 6, 3], **kwargs)

#     return model


# def ResNet_101(input_size, **kwargs):
#     """Constructs a ResNet-101 model.
#     """
#     model = ResNet(input_size, Bottleneck, [3, 4, 23, 3], **kwargs)

#     return model


# def ResNet_152(input_size, **kwargs):
#     """Constructs a ResNet-152 model.
#     """
#     model = ResNet(input_size, Bottleneck, [3, 8, 36, 3], **kwargs)

#     return model


"""
@author: Jun Wang    
@date: 20201019   
@contact: jun21wangustc@gmail.com 
"""

# based on:  
# https://github.com/TreB1eN/InsightFace_Pytorch/blob/master/model.py

from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch.nn.functional as F
import torch
from collections import namedtuple


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0 ,bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0 ,bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride ,bias=False), BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1 ,bias=False), PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1 ,bias=False), BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride ,bias=False), 
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3,3), (1,1),1 ,bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3,3), stride, 1 ,bias=False),
            BatchNorm2d(depth),
            SEModule(depth,16)
            )
    def forward(self,x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''
    
def get_block(in_channel, depth, num_units, stride = 2):
  return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units-1)]

def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units = 3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    return blocks

#class Backbone(Module):
from ..loss import ArcFace
class Resnet(Module):
    def __init__(self, num_layers, drop_ratio, mode='ir', feat_dim=512, out_h=7, out_w=7):
        super(Resnet, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1 ,bias=False), 
                                      BatchNorm2d(64), 
                                      PReLU(64))
        self.output_layer = Sequential(BatchNorm2d(512), 
                                       Dropout(drop_ratio),
                                       Flatten(),
                                       Linear(512 * out_h * out_w, feat_dim), # for eye
                                       BatchNorm1d(feat_dim))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)
        # self.head = ArcFace(feat_dim, 10575)
        self.head = Linear(512, 10575)

    def forward(self,x, labels=None):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        if self.training is False:
            return l2_norm(x)

        # x = l2_norm(x)
        prob = self.head(x)
        # prob = self.head(x, labels)
        return x, prob
