from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
from .pooling import GeneralizedMeanPooling


__all__ = ['DenseNet', 'densenet121', 'densenet161', 'densenet169', 'densenet201']


class DenseNet(nn.Module):
    __factory = {
        121: torchvision.models.densenet121,
        161: torchvision.models.densenet161,
        169: torchvision.models.densenet169,
        201: torchvision.models.densenet201,
    }
    def __init__(self, depth, pretrained=True, cut_at_pooling=False, rmds=True,
                 num_features=0, norm=False, dropout=0, num_classes=0, pool="gem", bnneck=True):
        super(DenseNet, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        self.bnneck = bnneck
        # Construct base (pretrained) resnet
        densenet_model = DenseNet.__factory[depth](pretrained=True)

        self.base = nn.Sequential(densenet_model.features)
        self.gap = nn.AdaptiveAvgPool2d(1)
        if pool == "gem":
            self.pooling = GeneralizedMeanPooling()
        else:
            self.pooling = nn.AdaptiveAvgPool2d(1)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = densenet_model.classifier.in_features
            
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
        if not self.bnneck:
            self.feat_bn = nn.Sequential()

        # if not pretrained:
        #     self.reset_params()

    def forward(self, x, feature_withbn=False):

        x = self.base(x)
        x = F.relu(x,inplace=True)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)

        if self.cut_at_pooling:
            return x

        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(x))
        else:
            bn_x = self.feat_bn(x)
        # if self.has_embedding:
        #     bn_x = self.feat(x)
        # else:
        #     bn_x = x

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



def densenet121(**kwargs):
    return DenseNet(121, **kwargs)

def densenet161(**kwargs):
    return DenseNet(161, **kwargs)

def densenet169(**kwargs):
    return DenseNet(169, **kwargs)

def densenet201(**kwargs):
    return DenseNet(201, **kwargs)

