from __future__ import print_function, absolute_import
import time
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import os.path as osp
from .evaluation_metrics import cmc, mean_ap
from .feature_extraction import extract_cnn_feature
from .utils.meters import AverageMeter
import torch.nn.functional as F
from .utils.rerank import re_ranking


def extract_features(model, data_loader, print_freq=10, metric=None):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    features = OrderedDict()
    labels = OrderedDict()
    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, pids, _) in enumerate(data_loader):

            data_time.update(time.time() - end)

            outputs = extract_cnn_feature(model, imgs)
            for fname, output, pid in zip(fnames, outputs, pids):
                features[fname] = output
                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    return features, labels

def pairwise_distance(features, query=None, gallery=None, metric=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        if metric is not None:
            x = metric.transform(x)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist_m
    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    if metric is not None:
        x = metric.transform(x)
        y = metric.transform(y)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m, x.numpy(), y.numpy()

def evaluate_batch(query_features, gallery_features, distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), cmc_flag=False):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    # print('Mean AP: {:4.1%}'.format(mAP))

    if (not cmc_flag):
        return mAP

    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True)
                }
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    # print('CMC Scores:')
    # for k in cmc_topk:
    #     print('  top-{:<4}{:12.1%}'
    #           .format(k,
    #                   cmc_scores['market1501'][k-1]))
    return cmc_scores['market1501'], mAP

def evaluate_minibatch(features, query, gallery, metric=None, cmc_flag=False, cmc_topk=(1, 5, 10), rerank=False, pre_features=None, minibatch=None):
    q_len = len(query)
    cmc_scores = np.zeros(100)
    mAP = 0
    if minibatch==None:
        minibatch = int(2e8//len(gallery))
    print('Using minibatch {} for evaluation.'.format(minibatch))
    for i in range(q_len//minibatch+1):
        start = minibatch*i
        end = min(minibatch*i+minibatch, q_len)
        query_i = query[start:end]
        distmat_i, query_i_features, gallery_i_features = pairwise_distance(features, query_i, gallery, metric=metric)
        cmc_scores_i, map_i = evaluate_batch(query_i_features, gallery_i_features, distmat_i, query=query_i, gallery=gallery, cmc_flag=cmc_flag, cmc_topk=cmc_topk)
        cmc_scores += cmc_scores_i*len(query_i)
        mAP += map_i*len(query_i)
    cmc_scores /= q_len
    mAP /= q_len
    print('Mean AP: {:4.1%}'.format(mAP))
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'
              .format(k,
                      cmc_scores[k-1]))
    return cmc_scores[0], mAP


def evaluate_all(query_features, gallery_features, distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), cmc_flag=False):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    if (not cmc_flag):
        return mAP

    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True)
                }
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores:')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'
              .format(k,
                      cmc_scores['market1501'][k-1]))
    return cmc_scores['market1501'][0], mAP
    # return cmc_scores['market1501'][0], mAP


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, data_loader, query, gallery, metric=None, cmc_flag=False, cmc_topk=(1, 5, 10), rerank=False, pre_features=None, minibatch=None):
        if (pre_features is None):
            features, _ = extract_features(self.model, data_loader)
        else:
            features = pre_features
        # import pdb;pdb.set_trace()

        if minibatch or len(query)*len(gallery)>3e8:
            print('Run minibatch evaluating.')
            results = evaluate_minibatch(features, query, gallery, metric=metric, cmc_flag=cmc_flag, cmc_topk=cmc_topk, minibatch=minibatch)
        else:
            distmat, query_features, gallery_features = pairwise_distance(features, query, gallery, metric=metric)
            results = evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag, cmc_topk=cmc_topk)
        if (not rerank):
            return results

        print('Applying person re-ranking ...')
        distmat_qq = pairwise_distance(features, query, query, metric=metric)
        distmat_gg = pairwise_distance(features, gallery, gallery, metric=metric)
        distmat = re_ranking(distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy())
        return evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)

class JointEvaluator(object):
    def __init__(self, models_list, weights=[1,1,1]):
        super(JointEvaluator, self).__init__()

        class EnsembleModel(nn.Module):
            def __init__(self, models_list, weights):
                super(EnsembleModel, self).__init__()
                self.models = nn.ModuleList(models_list)
                self.weights = weights
            def forward(self, x):
                f = [F.normalize(self.models[i](x),dim=1,p=2)*self.weights[i] for i in range(len(self.models))]
                f = torch.cat(f, dim=1)
                # f = torch.cat([feature.unsqueeze(0) for feature in f],dim=0).sum(dim=0)

                return f

        self.model = EnsembleModel(models_list, weights)


    def evaluate(self, data_loader, query, gallery, metric=None, cmc_flag=False, rerank=False, pre_features=None):
        if (pre_features is None):
            features, _ = extract_features(self.model, data_loader)
        else:
            features = pre_features
        distmat, query_features, gallery_features = pairwise_distance(features, query, gallery, metric=metric)
        results = evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)
        if (not rerank):
            return results

        print('Applying person re-ranking ...')
        distmat_qq, _, _ = pairwise_distance(features, query, query, metric=metric)
        distmat_gg, _, _ = pairwise_distance(features, gallery, gallery, metric=metric)
        distmat = re_ranking(distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy())
        return evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)

# from .evaluation_metrics.verification import *
# class EvaluatorFace(object):
#     def __init__(self, model, ver_root, ver_datasets, image_size=[112,112]):
#         super(EvaluatorFace, self).__init__()
#         self.model = model
#         self.ver_list = []
#         self.ver_name_list = []
#         for name in ver_datasets.split(','):
#             path = osp.join(ver_root, name + ".bin")
#             if osp.exists(path):
#                 print('loading.. ', name)
#                 data_set = load_bin(path, image_size)
#                 self.ver_list.append(data_set)
#                 self.ver_name_list.append(name)

#     def evaluate(self, data_loader, query, gallery, metric=None, cmc_flag=False, cmc_topk=(1, 5, 10), rerank=False, pre_features=None, minibatch=None):
#         if (pre_features is None):
#             features, _ = extract_features(self.model, data_loader)
#         else:
#             features = pre_features
#         # import pdb;pdb.set_trace()

#         if minibatch or len(query)*len(gallery)>3e8:
#             print('Run minibatch evaluating.')
#             results = evaluate_minibatch(features, query, gallery, metric=metric, cmc_flag=cmc_flag, cmc_topk=cmc_topk, minibatch=minibatch)
#         else:
#             distmat, query_features, gallery_features = pairwise_distance(features, query, gallery, metric=metric)
#             results = evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag, cmc_topk=cmc_topk)
#         if (not rerank):
#             return results

#         print('Applying person re-ranking ...')
#         distmat_qq = pairwise_distance(features, query, query, metric=metric)
#         distmat_gg = pairwise_distance(features, gallery, gallery, metric=metric)
#         distmat = re_ranking(distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy())
#         return evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)