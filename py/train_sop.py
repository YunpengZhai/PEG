from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import os
import glob
import copy
import sys
sys.path.insert(0, os.getcwd())
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import torch
from torch import nn
import torch.nn.functional as F
from torch.backends import cudnn
from torch.utils.data import DataLoader

from mebnet.dbscan import dbscan
from mebnet.core.utils.compute_dist import compute_jaccard_distance

from mebnet import datasets
from mebnet import models
from mebnet.population import *
from mebnet.reranking import re_ranking
from mebnet.trainers import PopulationTrainer, BaseTrainer
from mebnet.evaluators import Evaluator, extract_features
from mebnet.utils.data import IterLoader
from mebnet.utils.data import transforms as T
from mebnet.utils.data.sampler import RandomMultipleGallerySampler
from mebnet.utils.data.preprocessor import Preprocessor
from mebnet.utils.logging import Logger
from mebnet.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from mebnet.utils.scatter import J_scatter, eval_cluster
from mebnet.utils.lr_scheduler import WarmupMultiStepLR


start_epoch = best_mAP = 0

def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset

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
                                        transform=train_transformer, mutual=3),
                            batch_size=batch_size, num_workers=workers, sampler=sampler,
                            shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader

def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    
    # test_transformer = T.Compose([
    #          T.Resize((height, width), interpolation=3),
    #          T.ToTensor(),
    #          normalizer
    #      ])
    test_transformer = T.Compose([
             T.Resize((256, 256), interpolation=3),
             T.CenterCrop(height),
             T.ToTensor(),
             normalizer
         ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader

def create_population(archs, args):
    if not args.resume_dir:
        arch_num = len(archs)
        agents=[]
        for i in range(arch_num):
            arch = archs[i]
            model = models.create(arch, num_features=args.features, dropout=args.dropout, num_classes=args.num_clusters, rmds=args.downsampling, pool=args.pooling).cuda()
            model_ema = models.create(arch, num_features=args.features, dropout=args.dropout, num_classes=args.num_clusters, rmds=args.downsampling, pool=args.pooling).cuda()

            model = nn.DataParallel(model)
            model_ema = nn.DataParallel(model_ema)
            
            model_ema.module.classifier.weight.data.copy_(model.module.classifier.weight.data)

            for param in model_ema.parameters():
                param.detach_()

            hyper_param = {'soft_ce_weight':args.soft_ce_weight,
                           'soft_tri_weight':args.soft_tri_weight,
                           'lr':args.lr,
                           'epsilon': args.epsilon,
            }
            agent = Agent(str(i), arch, hyper_param, model, model_ema)
            agents.append(agent)
    else:
        print("create population from checkpoint in {}".format(args.resume_dir))
        cpt_list = glob.glob(args.resume_dir+"/model*_checkpoint.pth.tar")
        cpt_list = np.sort(cpt_list)
        agent_num = len(cpt_list)
        agents=[]
        for i in range(agent_num):
            agent_cpt = load_checkpoint(cpt_list[i])
            arch = agent_cpt['arch']
            model = models.create(arch, num_features=args.features, dropout=args.dropout, num_classes=args.num_clusters, rmds=args.downsampling, pool=args.pooling).cuda()
            model_ema = models.create(arch, num_features=args.features, dropout=args.dropout, num_classes=args.num_clusters, rmds=args.downsampling, pool=args.pooling).cuda()

            model = nn.DataParallel(model)
            model_ema = nn.DataParallel(model_ema)
            
            copy_state_dict(agent_cpt['state_dict'], model)
            copy_state_dict(agent_cpt['state_dict'], model_ema)
            model_ema.module.classifier.weight.data.copy_(model.module.classifier.weight.data)

            for param in model_ema.parameters():
                param.detach_()

            hyper_param = agent_cpt['hyperparam']
            agent = Agent(str(i), arch, hyper_param, model, model_ema)
            agents.append(agent)

    return Population(agents, args)


def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        print("set seed all cuda")
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    args.height=args.width=227
    if args.dataset_target in ['cub', 'car']:
        cmc_topk = (1,2,4,8)
    else:
        cmc_topk = (1,10,100)
    iters = args.iters if (args.iters>0) else None
    dataset_target = get_data(args.dataset_target, args.data_dir)
    test_loader_target = get_test_loader(dataset_target, args.height, args.width, args.batch_size, args.workers)
    
    cluster_loader = get_test_loader(dataset_target, args.height, args.width, args.batch_size, args.workers, testset=dataset_target.train)

    # Create model
    population = create_population(args.arch_list, args)
    population.display()
    

    for gn in range(args.genaration):
        # best_response
        population.best_response(dataset_target, cluster_loader)
        population.display()
        

        # mutual learning
        for epoch in range(args.epochs):
            cf = []
            f_lens = []
            for i in range(population.num_agents):
                dict_f, _ = extract_features(population.agents[i].model_ema, cluster_loader, print_freq=100)
                cf_i = torch.stack(list(dict_f.values()))
                cf_i = F.normalize(cf_i, dim=-1).numpy()
                cf.append(cf_i) #
                f_lens.append(cf_i.shape[-1])

            cf = np.concatenate(cf, axis=-1)
            cf = torch.from_numpy(cf)
            cf = F.normalize(cf, p=2, dim=-1).numpy()

#########
            moving_avg_features = cf

            print('\n Clustering into {} classes by {} \n '.format(args.num_clusters, args.cluster))
            if args.cluster == 'kmeans':
                km = MiniBatchKMeans(n_clusters=args.num_clusters, init='random' if args.num_clusters>1000 else 'k-means++', 
                    max_iter=100, batch_size=2000, init_size=3*args.num_clusters).fit(moving_avg_features)
                target_label = km.labels_
                target_centers = torch.from_numpy(km.cluster_centers_)
                num_classes = args.num_clusters
            elif args.cluster == 'dbscan':
                moving_avg_features = torch.from_numpy(moving_avg_features)
                dist = compute_jaccard_distance(moving_avg_features, k1=30, k2=6, search_option=0, fp16=False, verbose=True)
                target_label, target_centers, num_classes= dbscan(moving_avg_features, dist, args.eps, args)
            

            true_label=np.array([data[1] for data in dataset_target.train])
            from sklearn.metrics.cluster import adjusted_rand_score

            print(adjusted_rand_score(true_label, target_label))


            target_centers = F.normalize(target_centers, p=2, dim=1)
            # print(J_scatter(cf, km.labels_))
            dim_start = 0
            for i in range(population.num_agents):
                weights = F.normalize(target_centers[:,dim_start:dim_start+f_lens[i]], p=2, dim=-1)
                population.agents[i].model.module.classifier = nn.Linear(population.agents[i].model.module.num_features, num_classes, bias=False).cuda()
                population.agents[i].model.module.classifier.weight.data.copy_(weights.float().cuda())
                population.agents[i].model_ema.module.classifier = copy.deepcopy(population.agents[i].model.module.classifier)
                dim_start += f_lens[i]

            dataset_wooutliers = copy.deepcopy(dataset_target)
            dataset_wooutliers.train=[]
            for i in range(len(dataset_target.train)):
                dataset_target.train[i] = list(dataset_target.train[i])
                dataset_target.train[i][1] = int(target_label[i])
                dataset_target.train[i] = tuple(dataset_target.train[i])
                if int(target_label[i])==-1:
                    continue;
                else:
                    dataset_wooutliers.train.append(dataset_target.train[i])
            print("training data num:{}, classes:{}".format(len(dataset_wooutliers.train), num_classes))
            
            train_loader_target = get_train_loader(dataset_wooutliers, args.height, args.width,
                                                args.batch_size, args.workers, args.num_instances, iters)

            

            # Optimizer
            for i in range(population.num_agents):
                params = []
                for key, value in population.agents[i].model.named_parameters():
                    if not value.requires_grad:
                        continue
                    params += [{"params": [value], "lr": population.agents[i].hyperparam['lr'], "weight_decay": args.weight_decay}]
                population.agents[i].optim = torch.optim.Adam(params)
            optimizer=None
            # Trainer
            trainer = PopulationTrainer(population, num_cluster=args.num_clusters, alpha=args.alpha)

            train_loader_target.new_epoch()
            trainer.train_ops(epoch, train_loader_target, optimizer,
                        print_freq=args.print_freq, train_iters=len(train_loader_target))

            def save_agent(agent, is_best, best_mAP, mid):
                fdir=osp.join(args.logs_dir, "{}-G".format(gn))
                save_checkpoint({
                'arch':agent.arch,
                'hyperparam':agent.hyperparam,
                'state_dict': agent.model_ema.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
                }, is_best, fpath=osp.join(fdir, 'model'+str(mid)+'_checkpoint.pth.tar'), )
            def save_population(population):
                fdir=osp.join(args.logs_dir, "{}-G".format(gn))
                save_checkpoint(population, False, fpath=osp.join(fdir, 'best_population.pth.tar'))
            
            if ((epoch+1)%args.eval_step==0 or (epoch==args.epochs-1)):
                mAP = []
                for i in range(population.num_agents):
                    mAP_i = population.agents[i].evaluator_ema.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True, cmc_topk=cmc_topk)[0]
                    mAP.append(mAP_i)
                is_best = max(mAP) > best_mAP
                best_mAP = max(mAP + [best_mAP])
                for i in range(population.num_agents):
                    save_agent(population.agents[i], (is_best and (mAP[i]==best_mAP)), best_mAP, i)
                    if (is_best and (mAP[i]==best_mAP)):
                        best_model_ind = i
                    # if is_best:
                    #     save_population(population)
                log_text='\n * Finished epoch {:3d} genaration {:3d} \n'.format(epoch, gn)
                for ai in range(len(mAP)):
                    log_text = log_text+"model no.{} mAP: {:5.1%} \n".format(ai, mAP[ai])
                log_text = log_text+"best: {:5.1%}{} \n".format(best_mAP, ' *' if is_best else '')
                print(log_text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MEB-Net Training")
    # data
    parser.add_argument('-dt', '--dataset-target', type=str, default='market1501',
                        choices=datasets.names())
    # parser.add_argument('-ds', '--dataset-source', type=str, default='dukemtmc',
    #                     choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--num-clusters', type=int, default=500)
    parser.add_argument('--slots', type=int, default=3)
    parser.add_argument('--height', type=int, default=227,
                        help="input height")
    parser.add_argument('--width', type=int, default=227,
                        help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    parser.add_argument('--downsampling', type=bool, default=False,
                        help="remove the last down sampling")
    parser.add_argument('--cluster', type=str, default='kmeans')
    parser.add_argument('--pooling', type=str, default='gap')
    parser.add_argument('--eps', type=float, default=0.6)
    # model
    # parser.add_argument('-a', '--arch', type=str, default='resnet50',
    #                     choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    # optimizer
    parser.add_argument('--crs-lr', type=float, default=0.00035)
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--alpha', type=float, default=0.999)
    parser.add_argument('--moving-avg-momentum', type=float, default=0)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int, default=[40, 70], help='milestones for the learning rate decay')
    parser.add_argument('--soft-ce-weight', type=float, default=0.5)
    parser.add_argument('--soft-tri-weight', type=float, default=0.8)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--margin', type=float, default=0.0, help='margin for the triplet loss with batch hard')    
    parser.add_argument('--genaration', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--iters', type=int, default=2400)
    parser.add_argument('--cross-iters', type=int, default=1000)
    parser.add_argument('--mutation-r', type=float, default=0.5)
    # training configs
    parser.add_argument('--arch-list', type=list, default=['densenet121', 'densenet169', 'densenet_ibn121a', 'densenet_ibn169a', 'inceptionv3', 'resnet50', 'resnet_ibn50a', 'resnet_ibn50b', ])
    parser.add_argument('--seed', type=int, default=9)
    parser.add_argument('--print-freq', type=int, default=1)
    parser.add_argument('--eval-step', type=int, default=1)
    parser.add_argument('--resume', type=int, default=0)
    parser.add_argument('--resume-dir', type=str, default="")
    parser.add_argument('--scatter', type=int, default=1)
    parser.add_argument('--debug', type=int, default=0)
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(os.getcwd(),'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(os.getcwd(), 'logs'))
    parser.add_argument('--init-dir', type=str, metavar='PATH',
                        default=osp.join(os.getcwd(), 'logs'))
    main()
