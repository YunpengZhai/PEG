from __future__ import print_function, absolute_import
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import random
from .evaluation_metrics import accuracy
from .loss import TripletLoss, CrossEntropyLabelSmooth, SoftTripletLoss, SoftEntropy
from .utils.meters import AverageMeter


class PreTrainer(object):
    def __init__(self, model, num_classes, margin=0.0):
        super(PreTrainer, self).__init__()
        self.model = model
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        # self.criterion_triple = None
        self.criterion_triple = SoftTripletLoss(margin=margin).cuda()

    def train(self, epoch, data_loader_source, data_loader_target, optimizer, train_iters=200, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        precisions = AverageMeter()

        end = time.time()

        for i in range(train_iters):
            source_inputs = data_loader_source.next()
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            s_inputs, targets = self._parse_data(source_inputs)
            t_inputs, _ = self._parse_data(target_inputs)
            s_features, s_cls_out = self.model(s_inputs)
            # target samples: only forward
            # t_features, _ = self.model(t_inputs)

            # backward main #
            loss_ce, loss_tr, prec1 = self._forward(s_features, s_cls_out, targets)
            loss = loss_ce + loss_tr
            
            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tr.item())
            precisions.update(prec1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if ((i + 1) % print_freq == 0):
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_tr {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tr.val, losses_tr.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets

    def _forward(self, s_features, s_outputs, targets):
        loss_ce = self.criterion_ce(s_outputs, targets)
        if isinstance(self.criterion_triple, SoftTripletLoss):
            loss_tr = self.criterion_triple(s_features, s_features, targets)
        elif isinstance(self.criterion_triple, TripletLoss):
            loss_tr, _ = self.criterion_triple(s_features, targets)
        else:
            loss_tr = torch.zeros(1).cuda()
        prec, = accuracy(s_outputs.data, targets.data)
        prec = prec[0]

        return loss_ce, loss_tr, prec


class BaseTrainer(object): # for cross reference evaluation
    def __init__(self, model, num_classes, margin=0.0):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_triple = SoftTripletLoss(margin=margin).cuda()

    def train(self, epoch, data_loader_source, optimizer, ceonly=False ,train_iters=200, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        precisions = AverageMeter()

        end = time.time()

        for i in range(train_iters):
            source_inputs = data_loader_source.next()
            data_time.update(time.time() - end)

            s_inputs, targets = self._parse_data(source_inputs)
            s_features, s_cls_out = self.model(s_inputs)
            
            # backward main #
            loss_ce, loss_tr, prec1 = self._forward(s_features, s_cls_out, targets)
            
            # loss = loss_tr
            loss = loss_ce + loss_tr


            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tr.item())
            precisions.update(prec1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if ((i + 1) % print_freq == 0):
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_tr {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tr.val, losses_tr.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets

    def _forward(self, s_features, s_outputs, targets):
        loss_ce = self.criterion_ce(s_outputs, targets)
        if isinstance(self.criterion_triple, SoftTripletLoss):
            loss_tr = self.criterion_triple(s_features, s_features, targets)
        elif isinstance(self.criterion_triple, TripletLoss):
            loss_tr, _ = self.criterion_triple(s_features, targets)

        prec, = accuracy(s_outputs.data, targets.data)
        prec = prec[0]

        return loss_ce, loss_tr, prec



class PopulationTrainer(object):
    def __init__(self, population, num_cluster=500, alpha=0.999, group_max=3):
        super(PopulationTrainer, self).__init__()
        self.population = population
        self.num_cluster = num_cluster
        self.model_num = population.num_agents
        self.alpha = alpha
        self.group_max = group_max
        
        # self.criterion_ce = CrossEntropyLabelSmooth(num_cluster).cuda()
        self.criterion_ce_soft = SoftEntropy().cuda()
        self.criterion_tri = SoftTripletLoss(margin=0.0).cuda()
        self.criterion_tri_soft = SoftTripletLoss(margin=None).cuda()

        self.criterion_ce = [CrossEntropyLabelSmooth(num_cluster, epsilon=agent.hyperparam['epsilon']).cuda() for agent in population.agents]

    def train(self, epoch, data_loader_target,
            optimizer, print_freq=1, train_iters=200):
        for agent in self.population.agents:
            agent.model.train()
            agent.model_ema.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_ce = AverageMeter()
        losses_tri = AverageMeter()
        losses_ce_soft = AverageMeter()
        losses_tri_soft = AverageMeter()
        precision = AverageMeter()
        count = np.zeros((self.population.num_agents))

        end = time.time()
        for iter_idx in range(train_iters):
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, targets = self._parse_data(target_inputs)

            model_list = []
            model_ema_list = []
            self.group = min(len(self.population.family), self.group_max)
            family_inds = np.random.choice(range(len(self.population.family)), self.group, replace=False)
            ag_inds=[]
            for ind in family_inds:
                ag_idx = np.random.choice(self.population.family[ind])
                ag_inds.append(ag_idx)
                model_list.append(self.population.agents[ag_idx].model)
                model_ema_list.append(self.population.agents[ag_idx].model_ema)
            count[ag_inds]+=1

            f_out_t = []
            p_out_t = []
            f_out_t_ema = []
            p_out_t_ema = []
            for i in range(self.group):
                f_out_t_i, p_out_t_i = model_list[i](inputs[i])
                f_out_t.append(f_out_t_i)
                p_out_t.append(p_out_t_i)

                f_out_t_ema_i, p_out_t_ema_i = model_ema_list[i](inputs[i])
                f_out_t_ema.append(f_out_t_ema_i)
                p_out_t_ema.append(p_out_t_ema_i)
            
            sum_loss_ce = sum_loss_tri = sum_loss_ce_soft = sum_loss_tri_soft = 0
            loss = 0
            for i in range(self.group):#student
                loss_ce = self.criterion_ce[ag_inds[i]](p_out_t[i], targets)
                loss_tri = self.criterion_tri(f_out_t[i], f_out_t[i], targets)
                sum_loss_ce += loss_ce
                sum_loss_tri += loss_tri

                loss_ce_soft = loss_tri_soft = 0
                for j in range(self.group): # teacher
                    if i != j:
                        loss_ce_soft += (1.0/(self.group-1))*self.criterion_ce_soft(p_out_t[i], p_out_t_ema[j])
                        loss_tri_soft +=  (1.0/(self.group-1))*self.criterion_tri_soft(f_out_t[i], f_out_t_ema[j], targets)
                sum_loss_ce_soft+=loss_ce_soft
                sum_loss_tri_soft+=loss_tri_soft

                ce_soft_weight = min(self.population.agents[i].hyperparam['soft_ce_weight'], 1)
                tri_soft_weight = min(self.population.agents[i].hyperparam['soft_tri_weight'], 1)


                loss += loss_ce*(1-ce_soft_weight) + \
                         loss_tri*(1-tri_soft_weight) + \
                         loss_ce_soft*ce_soft_weight + loss_tri_soft*tri_soft_weight

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            for i in range(self.group):
                self._update_ema_variables(model_list[i], model_ema_list[i], self.alpha, epoch*len(data_loader_target)+iter_idx)

            prec_s = [accuracy(p_out_t[mi].data, targets.data)[0] for mi in range(self.group)]
            for mi in range(self.group):
                precision.update(prec_s[mi].item())

            losses_ce.update(sum_loss_ce.item())
            losses_tri.update(sum_loss_tri.item())
            losses_ce_soft.update(sum_loss_ce_soft.item())
            losses_tri_soft.update(sum_loss_tri_soft.item())
        
            # print log #
            batch_time.update(time.time() - end)
            end = time.time()
            

            if (iter_idx + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f}\t'
                      'Loss_tri {:.3f}\t'
                      'Loss_ce_soft {:.3f}\t'
                      'Loss_tri_soft {:.3f}\t'
                      'Prec {:.2%}\t'
                      'Training count {}'
                      .format(epoch, iter_idx + 1, len(data_loader_target),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce.avg, losses_tri.avg,
                              losses_ce_soft.avg, losses_tri_soft.avg,
                              precision.avg, count))

    def train_ops(self, epoch, data_loader_target,
            optimizer, print_freq=1, train_iters=200):
        for agent in self.population.agents:
            agent.model.train()
            agent.model_ema.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_ce = AverageMeter()
        losses_tri = AverageMeter()
        losses_ce_soft = AverageMeter()
        losses_tri_soft = AverageMeter()
        precision = AverageMeter()
        count = np.zeros((self.population.num_agents))

        end = time.time()
        for iter_idx in range(train_iters):
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, targets = self._parse_data(target_inputs)

            model_list = []
            model_ema_list = []
            optims = []
            self.group = min(len(self.population.family), self.group_max)
            family_inds = np.random.choice(range(len(self.population.family)), self.group, replace=False)
            ag_inds=[]
            for ind in family_inds:
                ag_idx = np.random.choice(self.population.family[ind])
                ag_inds.append(ag_idx)
                model_list.append(self.population.agents[ag_idx].model)
                model_ema_list.append(self.population.agents[ag_idx].model_ema)
                optims.append(self.population.agents[ag_idx].optim)
            count[ag_inds]+=1

            f_out_t = []
            p_out_t = []
            f_out_t_ema = []
            p_out_t_ema = []
            for i in range(self.group):
                f_out_t_i, p_out_t_i = model_list[i](inputs[i])
                f_out_t.append(f_out_t_i)
                p_out_t.append(p_out_t_i)

                f_out_t_ema_i, p_out_t_ema_i = model_ema_list[i](inputs[i])
                f_out_t_ema.append(f_out_t_ema_i)
                p_out_t_ema.append(p_out_t_ema_i)
            
            sum_loss_ce = sum_loss_tri = sum_loss_ce_soft = sum_loss_tri_soft = 0
            loss = 0
            for i in range(self.group):#student
                loss_ce = self.criterion_ce[ag_inds[i]](p_out_t[i], targets)
                loss_tri = self.criterion_tri(f_out_t[i], f_out_t[i], targets)
                sum_loss_ce += loss_ce
                sum_loss_tri += loss_tri

                loss_ce_soft = loss_tri_soft = 0
                for j in range(self.group): # teacher
                    if i != j:
                        loss_ce_soft += (1.0/(self.group-1))*self.criterion_ce_soft(p_out_t[i], p_out_t_ema[j])
                        loss_tri_soft +=  (1.0/(self.group-1))*self.criterion_tri_soft(f_out_t[i], f_out_t_ema[j], targets)
                sum_loss_ce_soft+=loss_ce_soft
                sum_loss_tri_soft+=loss_tri_soft

                ce_soft_weight = self.population.agents[i].hyperparam['soft_ce_weight']
                tri_soft_weight = self.population.agents[i].hyperparam['soft_tri_weight']

                loss = loss_ce*(1-ce_soft_weight) + \
                         loss_tri*(1-tri_soft_weight) + \
                         loss_ce_soft*ce_soft_weight + loss_tri_soft*tri_soft_weight
                optims[i].zero_grad()
                loss.backward()
                optims[i].step()
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            
            for i in range(self.group):
                self._update_ema_variables(model_list[i], model_ema_list[i], self.alpha, epoch*len(data_loader_target)+iter_idx)

            prec_s = [accuracy(p_out_t[mi].data, targets.data)[0] for mi in range(self.group)]
            for mi in range(self.group):
                precision.update(prec_s[mi].item())

            losses_ce.update(sum_loss_ce.item())
            losses_tri.update(sum_loss_tri.item())
            losses_ce_soft.update(sum_loss_ce_soft.item())
            losses_tri_soft.update(sum_loss_tri_soft.item())
        
            # print log #
            batch_time.update(time.time() - end)
            end = time.time()
            

            if (iter_idx + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f}\t'
                      'Loss_tri {:.3f}\t'
                      'Loss_ce_soft {:.3f}\t'
                      'Loss_tri_soft {:.3f}\t'
                      'Prec {:.2%}\t'
                      'Training count {}'
                      .format(epoch, iter_idx + 1, len(data_loader_target),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce.avg, losses_tri.avg,
                              losses_ce_soft.avg, losses_tri_soft.avg,
                              precision.avg, count))

    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def _parse_data(self, inputs):
        pids = inputs[-1]
        inputs_list = inputs[:-1]
        inputs_list = [img.cuda() for img in inputs_list]
        targets = pids.cuda()
        return inputs_list, targets


