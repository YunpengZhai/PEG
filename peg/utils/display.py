from time import time
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
 
from sklearn.manifold import TSNE


def plot_embedding(data, label, title, fpath):
    num_ids = len(set(label))

    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
 
    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / num_ids),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.savefig(fpath)

    return fig
    

def display_2d(data, label, fpath, nclass=20):
    randomlabels = np.random.choice(list(set(label)), size=nclass, replace=False)
    index = [lb in randomlabels for lb in label]
    data = data[index]
    label = label[index]
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    
    idlist = list(set(label))
    num_ids = len(idlist)

    x_min, x_max = np.min(result, 0), np.max(result, 0)
    result = (result - x_min) / (x_max - x_min)
 
    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(num_ids):
        pid = idlist[i]
        datapid = result[label==pid]
        ax.scatter(datapid[:,0], datapid[:,1], c=np.array([plt.cm.tab20(i / num_ids)]), marker="o")
    plt.xticks([])
    plt.yticks([])
    # plt.title(datasetname)
    plt.tight_layout()

    plt.savefig(fpath,dpi=600)
    # plt.show()
    import pdb
    pdb.set_trace()

def display_linear_unsupervised(data, label, fpath, nclass=10):
    randomlabels = np.random.choice(list(set(label)), size=nclass, replace=False)
    index = [lb in randomlabels for lb in label]
    data = data[index]
    label = label[index]
    tsne = TSNE(n_components=1, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    
    idlist = list(set(label))
    num_ids = len(idlist)

    x_min, x_max = np.min(result, 0), np.max(result, 0)
    result = (result - x_min) / (x_max - x_min)

    splits=200
    delta = 1.0/splits

    contri=[]
    for i in range(splits):
        contri.append((result<=(i+1)*delta).sum())
    for i in range(splits):
        if i != splits-1:
            contri[-1-i]-=contri[-1-i-1]
    
    fig = plt.figure(figsize=(15,5))
    ax = plt.subplot(111)
    s = np.array(range(splits))
    plt.bar(s, contri, width=1, color='salmon', edgecolor='k', lw=1) 

    plt.savefig(fpath)
    import pdb
    pdb.set_trace()

#     randomlabels = np.array([425,314,33,46])
#     from sklearn.decomposition import PCA
#     pca = PCA(n_components=1)
#     result = pca.fit_transform(data)
def display_linear(data, label, fpath, nclass=8):
    num_label=len(list(set(label)))
    counts = np.array([(label==i).sum() for i in range(num_label)])
    count_idx = np.argsort(counts)[::-1]

    randomlabels = np.random.choice(count_idx[:50], size=nclass, replace=False)
    # 104 136 76 429 434 630 30 414
    # randomlabels = count_idx[:30]

    # randomlabels = np.random.choice(list(set(label)), size=nclass, replace=False)

    # randomlabels = np.array([0,33,46,63,425,411,314,315,72,630,597,429])
    index = [lb in randomlabels for lb in label]
    data = data[index]
    label = label[index]
    tsne = TSNE(n_components=1, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    
    # randomlabels = np.random.choice(randomlabels, size=nclass, replace=False)
    # index = [lb in randomlabels for lb in label]
    # result = result[index]
    # label = label[index]
    idlist = list(set(label))
    num_ids = len(idlist)

    x_min, x_max = np.min(result, 0), np.max(result, 0)
    result = (result - x_min) / (x_max - x_min)

    splits=80
    delta = 1.0/splits

    fig = plt.figure(figsize=(16,4))
    ax = plt.subplot(111)
    s = np.array(range(splits))

    means={}
    # idlist = np.array([0,33,46,63,425,411,314,315,115,72,167])
    # num_ids = len(idlist)
    for ind in range(num_ids):
        datapid=result[label==idlist[ind]]
        
        means[idlist[ind]]=datapid.mean()/delta


        contri=[]
        for i in range(splits):
            contri.append((datapid<=(i+1)*delta).sum())
        for i in range(splits):
            if i != splits-1:
                contri[-1-i]-=contri[-1-i-1]
        

        color = plt.cm.tab20(ind / num_ids)
        # import pdb;pdb.set_trace()
        # color = (color[:3])+(0.5,)
        # plt.bar(s, contri, width=1, color=color, edgecolor=color, lw=1) 
        plt.bar(s, contri, width=1, color=color, edgecolor='k', lw=1) 
    
    # print(randomlabels)
    print(means)

    bwith=2
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
        
    plt.tick_params(labelsize=10)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    # plt.xticks([])
    # plt.yticks([])
    plt.tight_layout()

    plt.savefig(fpath,dpi=600)
    import pdb
    pdb.set_trace()

