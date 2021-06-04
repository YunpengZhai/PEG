from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import os
import copy
import sys
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import normalize

import torch
from torch import nn
import torch.nn.functional as F
from torch.backends import cudnn
from torch.utils.data import DataLoader

from peg import models
from peg.trainers import BaseTrainer
from peg.evaluators import Evaluator, extract_features
from peg.utils.data import IterLoader
from peg.utils.data import transforms as T
from peg.utils.data.sampler import RandomMultipleGallerySampler
from peg.utils.data.preprocessor import Preprocessor
from peg.utils.logging import Logger
from peg.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from peg.utils.scatter import J_scatter, eval_cluster
from peg.utils.lr_scheduler import WarmupMultiStepLR

def get_train_loader(dataset, height, width, batch_size, workers,
                    num_instances, iters):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             normalizer,
             T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
         ])

    train_set = dataset.train
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
                DataLoader(Preprocessor(train_set, root=dataset.images_dir,
                                        transform=train_transformer, mutual=1),
                            batch_size=batch_size, num_workers=workers, sampler=sampler,
                            shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader

def print_slots(slots, agents):
    names = []
    for i in slots:
        if i == -1:
            names.append("None")
        else:
            names.append(agents[i].arch)
    print("Using archtectures: " + ",".join(names))


def cross_scatter(model_emas, dataset_target, cluster_loader, args):
    model = models.create("osnet", num_features=args.features, dropout=args.dropout, num_classes=args.num_clusters)
    model.cuda()
    # model = nn.DataParallel(model)
    evaluator = Evaluator(model)

    cf = []
    for i in range(len(model_emas)):
        dict_f, _ = extract_features(model_emas[i], cluster_loader, print_freq=100)
        cf_i = torch.stack(list(dict_f.values()))
        cf_i = F.normalize(cf_i, dim=-1).numpy()
        cf.append(cf_i)

    cf = np.concatenate(cf, axis=-1)
    moving_avg_features = cf

    km = MiniBatchKMeans(n_clusters=args.num_clusters, max_iter=100, batch_size=100, init_size=1500).fit(moving_avg_features) # previous experiments
    # km = MiniBatchKMeans(n_clusters=args.num_clusters, init='random' if args.num_clusters>1000 else 'k-means++', max_iter=100, batch_size=2000, init_size=3*args.num_clusters).fit(moving_avg_features)
    # km = MiniBatchKMeans(n_clusters=args.num_clusters, max_iter=100, batch_size=100, init_size=3*args.num_clusters).fit(moving_avg_features)
    target_label = km.labels_

    # import pdb;pdb.set_trace()

    # change pseudo labels
    for i in range(len(dataset_target.train)):
        dataset_target.train[i] = list(dataset_target.train[i])
        dataset_target.train[i][1] = int(target_label[i])
        dataset_target.train[i] = tuple(dataset_target.train[i])

    train_loader_target = get_train_loader(dataset_target, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, args.cross_iters)

    crs_lr = args.lr
    if hasattr(args, 'crs_lr'):
        crs_lr = args.crs_lr
    # Optimizer
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": crs_lr, "weight_decay": args.weight_decay}]
    optimizer = torch.optim.Adam(params)
    

    # Trainer
    trainer = BaseTrainer(model, args.num_clusters, margin=args.margin)

    rpnum = 1
    scatters=[]
    for repeat in range(rpnum):
        for epoch in range(1):
            train_loader_target.new_epoch()
            trainer.train(epoch, train_loader_target, optimizer,
                        train_iters=len(train_loader_target), print_freq=200)
        #####evaluate scatter
            dict_f, _ = extract_features(model, cluster_loader, print_freq=200)
            cf = torch.stack(list(dict_f.values()))
            cf = F.normalize(cf, dim=-1).numpy()
            scatter = J_scatter(cf, target_label)
            print(scatter)
        scatters.append(scatter)
    avg_scatter = sum(scatters)/len(scatters)
    # mid_scatter=(sum(scatters)-max(scatters)-min(scatters))/(len(scatters)-2)
    # return scatter
    # return mid_scatter
    return avg_scatter


class Agent():
    def __init__(self, name, arch, hyperparam, model,model_ema=None, args=None):
        self.name=name
        self.arch=arch
        self.hyperparam=hyperparam
        self.model=model
        self.model_ema=model_ema
        self.evaluator_ema=Evaluator(self.model_ema)
        self.optim = None
        params = []
        for key, value in self.model.named_parameters():
            if not value.requires_grad:
                continue
            params += [{"params": [value], "lr": self.hyperparam['lr'], "weight_decay": args.weight_decay}]
        self.optim = torch.optim.Adam(params)

    
    def display(self):
        print("Arch: {}".format(self.arch))
        print("Name: {}".format(self.name))
        print(self.hyperparam)

    def transgenation(self, r):
        minf=1-r
        maxf=1+r
        for k in self.hyperparam:
            factor = random.uniform(minf,maxf)
            self.hyperparam[k]=factor*self.hyperparam[k]

    def reproduce(self,num=2, r=0.2):
        ret=[]
        for i in range(num):
            agent=copy.deepcopy(self)
            agent.name += str(i+1)
            agent.transgenation(r)
            ret.append(agent)
        self.name +='0'
        return ret

class Population():
    def __init__(self, agents, args):
        self.agents = agents
        self.num_agents = len(self.agents)
        self.args = args
        self.r = args.mutation_r
        self.family = [[i] for i in range(self.num_agents)]


    def update_family(self):
        family_dict={}
        for i in range(self.num_agents):
            fname = self.agents[i].name[:-1]
            if fname not in family_dict:
                family_dict[fname]=[]
            family_dict[fname].append(i)
        family=list(family_dict.values())
        self.family=family
        

    def display(self):
        print("Display Population:")
        for agent in self.agents:
            agent.display()

    def reproduce(self,num=2):
        for i in range(self.num_agents):
            # if i==1:
            #     newagents = self.agents[i].reproduce(num=0)
            # else:
            newagents = self.agents[i].reproduce(num=num, r=self.r)
            self.agents.extend(newagents)
        self.num_agents = len(self.agents)
        self.update_family()
        

    def best_response(self, dataset_target, cluster_loader, outcome_func=cross_scatter, init=False):
        #init slots
        candidates = [-1] + [i for i in range(self.num_agents)] #-1 : none
        slots = np.random.choice(candidates, size=self.args.slots, replace=False)


        if init:
            print("Skip best response using {}".format(init))
            slots = np.array(init)
            # slots = np.array([2,3])
            self.agents = [self.agents[i] for i in slots if i != -1]
            self.num_agents = len(self.agents)
            self.family = [[i] for i in range(self.num_agents)]
            return
        print("initial slots: {} for archs".format(slots))
        print_slots(slots, self.agents)
        
        print("init model list with ema model")
        model_list = [self.agents[i].model_ema for i in slots if i != -1]
        score = outcome_func(model_list, dataset_target, cluster_loader, self.args)
        print(score)
        # import pdb;pdb.set_trace()
        # return

        nochange = 0
        slot_i = 0
        while nochange<self.args.slots-1:
            # if slot_i==1:
            #     import pdb;pdb.set_trace()
            print("searching for slot {}".format(slot_i))
            this = slots[slot_i]
            temp_slots = copy.deepcopy(slots)
            max_score = score
            for cdt in candidates:
                if cdt!=-1 and cdt in slots:
                    continue;
                temp_slots[slot_i] = cdt
                if temp_slots.sum() ==  - len(temp_slots): # all is -1
                    continue;
                # if (temp_slots!=-1).sum()<=1:
                #     continue;
                print_slots(temp_slots, self.agents)
                print("evaluate slots {}".format(temp_slots))
                model_list = [self.agents[i].model_ema for i in temp_slots if i != -1]
                temp_score = outcome_func(model_list, dataset_target, cluster_loader, self.args)
                print(temp_score)
                if temp_score > max_score:
                    max_score = temp_score
                    max_slots = copy.deepcopy(temp_slots)
            if max_score>score:
                print("max score is {}, with slots {}".format(max_score, max_slots))
                score = max_score
                slots = max_slots
                nochange = 0
            else:
                print("no change")
                nochange += 1
            slot_i = (slot_i+1)%self.args.slots
        print("best slots: {}".format(slots))
        self.agents = [self.agents[i] for i in slots if i != -1]
        self.num_agents = len(self.agents)
        self.update_family()
        return

    def best_individual(self, dataset_target, cluster_loader, outcome_func=cross_scatter):
        scores=[]
        for i in range(self.num_agents):
            model_list = [self.agents[i].model_ema]
            score = outcome_func(model_list, dataset_target, cluster_loader, self.args)
            scores.append(score)
            print(score)
        scores = np.array(scores)
        inds = np.argsort(scores)[::-1]
        slots = inds[:self.args.slots]
        print("best slots: {}".format(slots))
        self.agents = [self.agents[i] for i in slots if i != -1]
        self.num_agents = len(self.agents)
        self.update_family()
        return

    def best_network(self, dataset_target, cluster_loader, outcome_func=cross_scatter):
        # select the best one network for test.
        scores=[]
        for i in range(self.num_agents):
            print("testing network {}".format(i))
            model_list = [self.agents[i].model_ema]
            score = outcome_func(model_list, dataset_target, cluster_loader, self.args)
            scores.append(score)
            print(score)
        scores = np.array(scores)
        inds = np.argsort(scores)[::-1]
        self.agents = [self.agents[inds[0]] ]
        self.num_agents = len(self.agents)
        self.update_family()
        return