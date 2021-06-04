from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
from .pooling import GeneralizedMeanPooling

from .resnext_ibn_a import resnext101_ibn_a


__all__ = ['ResNextIBN', 'resnext_ibn101_a']


class ResNextIBN(nn.Module):
    __factory = {
        '101a': resnext101_ibn_a,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False, rmds=True, 
                 num_features=0, norm=False, dropout=0, num_classes=0):
        super(ResNextIBN, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        resnext = ResNextIBN.__factory[depth](pretrained=pretrained)

        # resnext = ResNetIBN.__factory[depth](pretrained=pretrained)
        if rmds:
            resnext.layer4[0].conv2.stride = (1,1)
            resnext.layer4[0].downsample[0].stride = (1,1)
        self.base = nn.Sequential(
            resnext.conv1, resnext.bn1, resnext.relu, resnext.maxpool1,
            resnext.layer1, resnext.layer2, resnext.layer3, resnext.layer4)
        self.gap = nn.AdaptiveAvgPool2d(1)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = resnext.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
                self.feat_bn = nn.BatchNorm1d(self.num_features)
            self.feat_bn.bias.requires_grad_(False)
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
                init.normal_(self.classifier.weight, std=0.001)
        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

        if not pretrained:
            self.reset_params()

    def forward(self, x, feature_withbn=False):
        x = self.base(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        if self.cut_at_pooling:
            return x

        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(x))
        else:
            bn_x = self.feat_bn(x)

        if self.training is False:
            bn_x = F.normalize(bn_x)
            return bn_x

        if self.norm:
            bn_x = F.normalize(bn_x)
        elif self.has_embedding:
            bn_x = F.relu(bn_x)

        if self.dropout > 0:
            bn_x = self.drop(bn_x)

        if self.num_classes > 0:
            prob = self.classifier(bn_x)
        else:
            return x, bn_x

        if feature_withbn:
            return bn_x, prob
        return x, prob

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        resnext = ResNetIBN.__factory[self.depth](pretrained=self.pretrained)
        self.base[0].load_state_dict(resnext.conv1.state_dict())
        self.base[1].load_state_dict(resnext.bn1.state_dict())
        self.base[2].load_state_dict(resnext.relu.state_dict())
        self.base[3].load_state_dict(resnext.maxpool.state_dict())
        self.base[4].load_state_dict(resnext.layer1.state_dict())
        self.base[5].load_state_dict(resnext.layer2.state_dict())
        self.base[6].load_state_dict(resnext.layer3.state_dict())
        self.base[7].load_state_dict(resnext.layer4.state_dict())


def resnext_ibn101_a(**kwargs):
    return ResNextIBN('101a', **kwargs)



