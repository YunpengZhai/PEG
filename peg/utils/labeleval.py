
import torch
import numpy as np
def label_eval(fake_labels, true_labels):
    n = fake_labels.size(0)

    flabel_m = fake_labels.expand(n, n).eq(fake_labels.expand(n, n).t()).bool()
    tlabel_m = true_labels.expand(n, n).eq(true_labels.expand(n, n).t()).bool()
    a = (flabel_m *tlabel_m ).sum().float()
    b = (flabel_m *(~tlabel_m)).sum().float()
    c = ((~flabel_m) * tlabel_m ).sum().float()
    jc = a/(a+b+c)
    fmi = (a/(a+b)*a/(a+c))**0.5
    print(jc, fmi)
    return jc, fmi


    n = fake_labels.size(0)
    # import pdb;pdb.set_trace()
    valid = ((fake_labels.expand(n, n)!=-1)*(fake_labels.expand(n, n).t()!=-1)).float() *(1-torch.eye(n))
    # valid_p = float(valid.sum())/valid.numel()

    flabel_m = fake_labels.expand(n, n).eq(fake_labels.expand(n, n).t()).bool()
    tlabel_m = true_labels.expand(n, n).eq(true_labels.expand(n, n).t()).bool()
    # import pdb;pdb.set_trace()
    a = (flabel_m *tlabel_m *valid).sum().float()#a = (flabel_m *tlabel_m).sum().float()
    b = (flabel_m *(1-tlabel_m)*valid).sum().float()
    c = ((1-flabel_m) * tlabel_m *valid).sum().float()
    jc = a/(a+b+c)
    fmi = (a/(a+b)*a/(a+c))**0.5
    print(jc, fmi)
    return jc, fmi

    # P = (flabel_m *tlabel_m *valid).sum().float()/(flabel_m*valid).sum().float()
    # R = (flabel_m *tlabel_m *valid).sum().float()/(tlabel_m).sum().float()
    # F1 = 2*P*R/(P+R)
    # return F1, valid_p

    
def assign_noise(X, labels):
    n = labels.shape[0]
    while(-1 in labels):
        valid = (labels!=-1)
        distance = X[valid==False][:,valid]
        n_index = np.argmin(np.min(distance, axis=1))
        l_index = np.argmin(distance[n_index])
        n_index = np.where(valid==False)[0][n_index]
        l_index = np.where(valid)[0][l_index]
        labels[n_index] = labels[l_index]
    return labels