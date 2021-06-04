import numpy as np
import torch

def J_scatter_debug(features, labels):
    label_list = list(set(labels))
    feature_centers =[]
    counts = []
    variance_w = 0
    vws = []
    for i in range(len(label_list)):
        if label_list[i]==-1:
            continue
        i_features = features[labels==label_list[i]]
        i_count = i_features.shape[0]
        counts.append(i_count)
        i_center = np.mean(i_features, axis=0, keepdims=True)
        feature_centers.append(i_center)
        i_variance = np.sum(np.power(i_features - i_center, 2),axis=1)
        i_variance = np.sum(i_variance, axis=0)
        variance_w += i_variance
        vws.append(i_variance)
        # np.dot(i_feature - i_center, i_feature - i_center)
    feature_centers = np.concatenate(feature_centers, axis=0)
    centers_center = np.mean(features, axis=0, keepdims=True)
    # centers_center = np.mean(feature_centers, axis=0, keepdims=True)
    counts = np.array(counts)
    variance_b = np.sum(np.power(feature_centers - centers_center, 2),axis=1)
    import pdb;pdb.set_trace()
    variance_b = np.sum(counts*variance_b, axis=0)
    return variance_b,variance_w,feature_centers,centers_center

def J_scatter(features, labels):
    label_list = list(set(labels))
    feature_centers =[]
    counts = []
    variance_w = 0
    for i in range(len(label_list)):
        if label_list[i]==-1:
            continue
        i_features = features[labels==label_list[i]]
        i_count = i_features.shape[0]
        counts.append(i_count)
        i_center = np.mean(i_features, axis=0, keepdims=True)
        feature_centers.append(i_center)
        i_variance = np.sum(np.power(i_features - i_center, 2),axis=1)
        i_variance = np.sum(i_variance, axis=0)
        variance_w += i_variance
        # np.dot(i_feature - i_center, i_feature - i_center)
    feature_centers = np.concatenate(feature_centers, axis=0)
    centers_center = np.mean(features, axis=0, keepdims=True)
    # print("center of center")
    # centers_center = np.mean(feature_centers, axis=0, keepdims=True)
    counts = np.array(counts)
    variance_b = np.sum(np.power(feature_centers - centers_center, 2),axis=1)
    variance_b = np.sum(counts*variance_b, axis=0)
    return variance_b/variance_w

def J_scatter_weighted(features, labels):
    label_list = list(set(labels))
    feature_centers =[]
    counts = []
    variance_w = 0
    for i in range(len(label_list)):
        if label_list[i]==-1:
            continue
        i_features = features[labels==label_list[i]]
        i_count = i_features.shape[0]
        counts.append(i_count)
        i_center = np.mean(i_features, axis=0, keepdims=True)
        feature_centers.append(i_center)

        i_variance = np.sum(np.power(i_features - i_center, 2),axis=1) #n,1
        i_variance += 1e-2

        weights = 1-i_variance/np.max(i_variance)
        weights += 1e-2
        weights = weights/weights.sum()

        i_variance = np.sum(weights*i_variance, axis=0)
        variance_w += i_count*i_variance
        # variance_w += i_count*i_variance
        # np.dot(i_feature - i_center, i_feature - i_center)
    feature_centers = np.concatenate(feature_centers, axis=0)
    centers_center = np.mean(features, axis=0, keepdims=True)
    counts = np.array(counts)
    # import pdb;pdb.set_trace()
    counts_all = counts.sum()
    counts = counts/counts.sum()

    variance_b = np.sum(np.power(feature_centers - centers_center, 2),axis=1)
    variance_b = np.sum(counts*variance_b, axis=0)

    # import pdb;pdb.set_trace()

    return variance_b/(variance_w/counts_all)


def eval_cluster(fake_labels, true_labels):
    fake_labels = torch.from_numpy(fake_labels)
    true_labels = torch.from_numpy(true_labels)

    n = fake_labels.size(0)
    # import pdb;pdb.set_trace()
    valid = ((fake_labels.expand(n, n)!=-1)*(fake_labels.expand(n, n).t()!=-1)).float() *(1-torch.eye(n))
    valid_p = float(valid.sum())/valid.numel()

    flabel_m = fake_labels.expand(n, n).eq(fake_labels.expand(n, n).t()).float()
    flabel_m = flabel_m *valid
    tlabel_m = true_labels.expand(n, n).eq(true_labels.expand(n, n).t()).float()
    # import pdb;pdb.set_trace()
    a = (flabel_m *tlabel_m).sum().float()
    b = (flabel_m *(1-tlabel_m)).sum().float()
    c = ((1-flabel_m) * tlabel_m).sum().float()
    jc = a/(a+b+c)
    fmi = (a/(a+b)*a/(a+c))**0.5

    print("P:{},R:{},jc:{},fmi:{}".format(a/(a+b), a/(a+c),jc,fmi))
    return a/(a+b), a/(a+c), jc, fmi #p r
