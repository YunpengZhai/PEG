from __future__ import absolute_import

from .resnet import *
from .resnest import *
from .resnet_ibn import *
from .resnext import *
from .resnext_ibn import *
from .se_resnet_ibn import *
from .densenet_ibn import *
from .densenet import *
from .inceptionv3 import *
from .osnet import *


__factory = {
    'resnest269':resnest269,
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnet_ibn50a': resnet_ibn50a,
    'resnet_ibn101a': resnet_ibn101a,
    'resnet_ibn50b': resnet_ibn50b,
    'resnet_ibn101b': resnet_ibn101b,
    'resnext50': resnext50,
    'resnext_ibn101_a': resnext_ibn101_a,
    # 'se_resnet_ibn50_a': se_resnet_ibn50_a,
    'se_resnet_ibn101_a': se_resnet_ibn101_a,
    'densenet121': densenet121,
    'densenet161': densenet161,
    'densenet169': densenet169,
    'densenet201': densenet201,
    'densenet_ibn121a': densenet_ibn121a,
    'densenet_ibn169a': densenet_ibn169a,
    'inceptionv3': inceptionv3,
    'osnet':osnet_ibn,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a model instance.

    Parameters
    ----------
    name : str
        Model name. Can be one of 'inception', 'resnet18', 'resnet34',
        'resnet50', 'resnet101', and 'resnet152'.
    pretrained : bool, optional
        Only applied for 'resnet*' models. If True, will use ImageNet pretrained
        model. Default: True
    cut_at_pooling : bool, optional
        If True, will cut the model before the last global pooling layer and
        ignore the remaining kwargs. Default: False
    num_features : int, optional
        If positive, will append a Linear layer after the global pooling layer,
        with this number of output units, followed by a BatchNorm layer.
        Otherwise these layers will not be appended. Default: 256 for
        'inception', 0 for 'resnet*'
    norm : bool, optional
        If True, will normalize the feature to be unit L2-norm for each sample.
        Otherwise will append a ReLU layer after the above Linear layer if
        num_features > 0. Default: False
    dropout : float, optional
        If positive, will append a Dropout layer with this dropout rate.
        Default: 0
    num_classes : int, optional
        If positive, will append a Linear layer at the end as the classifier
        with this number of output units. Default: 0
    """
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](*args, **kwargs)
