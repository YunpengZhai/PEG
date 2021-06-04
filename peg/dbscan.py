import collections

import numpy as np
import torch
from sklearn.cluster import DBSCAN

def dbscan(features, dist, eps, args):
    # assert isinstance(dist, np.ndarray)

    # clustering
    min_samples = 4 #args.min_samples
    use_outliers = False #args.use_outliers

    cluster = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed", n_jobs=-1,)
    labels = cluster.fit_predict(dist)
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # cluster labels -> pseudo labels
    # compute cluster centers
    centers = collections.defaultdict(list)
    outliers = 0
    for i, label in enumerate(labels):
        if label == -1:
            if not use_outliers:
                continue
            labels[i] = num_clusters + outliers
            outliers += 1

        centers[labels[i]].append(features[i])

    centers = [
        torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
    ]
    centers = torch.stack(centers, dim=0)
    # labels = to_torch(labels).long()
    num_clusters += outliers

    return labels, centers, num_clusters