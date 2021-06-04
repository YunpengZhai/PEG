from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
from .pooling import GeneralizedMeanPooling

from .se_resnet_ibn_a import se_resnet101_ibn_a, se_resnet50_ibn_a


__all__ = ['SeResNetIBN', 'se_resnet_ibn50_a', 'se_resnet_ibn101_a']


class SeResNetIBN(nn.Module):
    __factory = {
        '50a': se_resnet50_ibn_a,
        '101a': se_resnet101_ibn_a,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False, rmds=True, 
                 num_features=0, norm=False, dropout=0, num_classes=0):
        super(SeResNetIBN, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        seresnet = SeResNetIBN.__factory[depth](pretrained=pretrained)

        # seresnet = ResNetIBN.__factory[depth](pretrained=pretrained)
        if rmds:
            seresnet.layer4[0].conv2.stride = (1,1)
            seresnet.layer4[0].downsample[0].stride = (1,1)
        self.base = nn.Sequential(
            seresnet.conv1, seresnet.bn1, seresnet.relu, seresnet.maxpool,
            seresnet.layer1, seresnet.layer2, seresnet.layer3, seresnet.layer4)
        self.gap = nn.AdaptiveAvgPool2d(1)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = seresnet.fc.in_features

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

        seresnet = ResNetIBN.__factory[self.depth](pretrained=self.pretrained)
        self.base[0].load_state_dict(seresnet.conv1.state_dict())
        self.base[1].load_state_dict(seresnet.bn1.state_dict())
        self.base[2].load_state_dict(seresnet.relu.state_dict())
        self.base[3].load_state_dict(seresnet.maxpool.state_dict())
        self.base[4].load_state_dict(seresnet.layer1.state_dict())
        self.base[5].load_state_dict(seresnet.layer2.state_dict())
        self.base[6].load_state_dict(seresnet.layer3.state_dict())
        self.base[7].load_state_dict(seresnet.layer4.state_dict())


def se_resnet_ibn50_a(**kwargs):
    return SeResNetIBN('50a', **kwargs)

def se_resnet_ibn101_a(**kwargs):
    return SeResNetIBN('101a', **kwargs)



