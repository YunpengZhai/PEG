import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
# from models.BN_Inception import Embedding


from collections import OrderedDict
import torch
import torch.nn as nn
import os

class Embedding(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=None, normalized=True):
        super(Embedding, self).__init__()
        self.bn = nn.BatchNorm2d(in_dim, eps=1e-5)
        self.linear = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.dropout = dropout
        self.normalized = normalized

    def forward(self, x):
        if self.dropout is not None:
            x = nn.Dropout(p=self.dropout)(x, inplace=True)
        x = self.linear(x)
        if self.normalized:
            norm = x.norm(dim=1, p=2, keepdim=True)
            x = x.div(norm.expand_as(x))
        return x


class InceptionModule(nn.Module):
    def __init__(self, inplane, outplane_a1x1, outplane_b3x3_reduce, outplane_b3x3, outplane_c5x5_reduce, outplane_c5x5, outplane_pool_proj):
        super(InceptionModule, self).__init__()
        self.a = nn.Sequential(OrderedDict([
            ('1x1', nn.Conv2d(inplane, outplane_a1x1, (1, 1), (1, 1), (0, 0))),
            ('1x1_relu', nn.ReLU(True))
        ]))

        self.b = nn.Sequential(OrderedDict([
            ('3x3_reduce', nn.Conv2d(inplane, outplane_b3x3_reduce, (1, 1), (1, 1), (0, 0))),
            ('3x3_relu1', nn.ReLU(True)),
            ('3x3', nn.Conv2d(outplane_b3x3_reduce, outplane_b3x3, (3, 3), (1, 1), (1, 1))),
            ('3x3_relu2', nn.ReLU(True))
        ]))

        self.c = nn.Sequential(OrderedDict([
            ('5x5_reduce', nn.Conv2d(inplane, outplane_c5x5_reduce, (1, 1), (1, 1), (0, 0))),
            ('5x5_relu1', nn.ReLU(True)),
            ('5x5', nn.Conv2d(outplane_c5x5_reduce, outplane_c5x5, (5, 5), (1, 1), (2, 2))),
            ('5x5_relu2', nn.ReLU(True))
        ]))

        self.d = nn.Sequential(OrderedDict([
            ('pool_pool', nn.MaxPool2d((3, 3), (1, 1), (1, 1))),
            ('pool_proj', nn.Conv2d(inplane, outplane_pool_proj, (1, 1), (1, 1), (0, 0))),
            ('pool_relu', nn.ReLU(True))
        ]))
        for m in self.a.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
        for m in self.b.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
        for m in self.c.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
        for m in self.d.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)


    def forward(self, input):
        return torch.cat([self.a(input), self.b(input), self.c(input), self.d(input)], 1)
      

class Model(nn.Module):
    def __init__(self, dim=512, self_supervision_rot=0):
        super(Model, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Sequential(OrderedDict([
                ('7x7_s2', nn.Conv2d(3, 64, (7, 7), (2, 2), (3, 3))),
                ('relu1', nn.ReLU(True)),
                ('pool1', nn.MaxPool2d((3, 3), (2, 2), ceil_mode = True)),
                ('lrn1', nn.CrossMapLRN2d(5, 0.0001, 0.75, 1))
            ]))),

            ('conv2', nn.Sequential(OrderedDict([
                ('3x3_reduce', nn.Conv2d(64, 64, (1, 1), (1, 1), (0, 0))),
                ('relu1', nn.ReLU(True)),
                ('3x3', nn.Conv2d(64, 192, (3, 3), (1, 1), (1, 1))),
                ('relu2', nn.ReLU(True)),
                ('lrn2', nn.CrossMapLRN2d(5, 0.0001, 0.75, 1)),
                ('pool2', nn.MaxPool2d((3, 3), (2, 2), ceil_mode = True))
            ]))),

            ('inception_3a', InceptionModule(192, 64, 96, 128, 16, 32, 32)),
            ('inception_3b', InceptionModule(256, 128, 128, 192, 32, 96, 64)),

            ('pool3', nn.MaxPool2d((3, 3), (2, 2), ceil_mode = True)),

            ('inception_4a', InceptionModule(480, 192, 96, 208, 16, 48, 64)),
            ('inception_4b', InceptionModule(512, 160, 112, 224, 24, 64, 64)),
            ('inception_4c', InceptionModule(512, 128, 128, 256, 24, 64, 64)),
            ('inception_4d', InceptionModule(512, 112, 144, 288, 32, 64, 64)),
            ('inception_4e', InceptionModule(528, 256, 160, 320, 32, 128, 128)),

            ('pool4', nn.MaxPool2d((3, 3), (2, 2), ceil_mode = True)),

            ('inception_5a', InceptionModule(832, 256, 160, 320, 32, 128, 128)),
            ('inception_5b', InceptionModule(832, 384, 192, 384, 48, 128, 128)),
            ('pool5', nn.AvgPool2d((7, 7), (1, 1), ceil_mode = True))])
        )
        self.cut_at_pooling = False
        self.dim = dim
        self.self_supervision_rot = self_supervision_rot
        normalized = (self.dim % 64 == 0)
        self.classifier = Embedding(1024, self.dim, normalized=normalized)
        if self.self_supervision_rot:
            self.classifier_rot = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 4),
            nn.ReLU(True),
            )
            print(self.classifier_rot)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #print(m)
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                #print(m)
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                #print(m)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, rot=False, org_feature=False):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        if self.cut_at_pooling:
            return x
        if (not self.self_supervision_rot) or (not rot):
            y = self.classifier(x)
            return y 
        else:
            z = self.classifier_rot(x)
            return z
        
dic = {"3x3":"b", "5x5":"c", "1x1":"a", "poo":"d"}    
def inception_v1_ml(dim=512, pretrained=True, self_supervision_rot=0, model_path=None):
    model = Model(dim, self_supervision_rot)
    if model_path is None:
        model_path = "/home/zyp/code/eccv/UDML_SS/pretrained/inception.pth"
    if pretrained:
        print("loaded++++++++++++++++++++++++++++++++++")
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path)
        print(pretrained_dict.keys())
        for key in list(pretrained_dict.keys()):
            l = key.split(".")
            if "inception" in l[0]:
                l.insert(1, dic[l[1][:3]])
                newkey = ".".join(l)
            else:
                newkey = key
            newkey = "features." + newkey
            tmp = pretrained_dict[key]
            pretrained_dict[newkey] = tmp
            del pretrained_dict[key]
        pretrained_dict = {k: torch.from_numpy(v).cuda() for k, v in pretrained_dict.items() if k in model_dict}
        print(len(pretrained_dict))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("finished")
    return model



