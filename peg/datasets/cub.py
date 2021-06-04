from __future__ import print_function, absolute_import
import os.path as osp
import tarfile

import glob
import re
import urllib
import zipfile

from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json


def _pluck_cub(list_file, clsindex, subdir='images'):
    with open(list_file, 'r') as f:
        lines = f.readlines()
    ret = []
    pids = []
    for line in lines:
        line = line.strip()
        elements = line.split(' ')
        imageid = int(elements[0])
        fname = elements[1]
        pid = int(fname.split('.')[0])
        if pid not in clsindex:
            continue
        pid -= 1
        if pid not in pids:
            pids.append(pid)
        ret.append((osp.join(subdir,fname), pid, imageid))
    return ret, pids

class Dataset_CUB(object):
    def __init__(self, root):
        self.root = root
        self.train, self.val, self.trainval = [], [], []
        self.query, self.gallery = [], []
        self.num_train_ids, self.num_val_ids, self.num_trainval_ids = 0, 0, 0

    @property
    def images_dir(self):
        return osp.join(self.root, 'CUB_200_2011')

    def load(self, verbose=True):
        exdir = osp.join(self.root, 'CUB_200_2011')
        self.train, train_pids = _pluck_cub(osp.join(exdir, 'images.txt'), [i+1 for i in range(100)], subdir='images')
        self.query, query_pids = _pluck_cub(osp.join(exdir, 'images.txt'), [i+1 for i in range(100,200)], subdir='images')
        self.gallery, gallery_pids = _pluck_cub(osp.join(exdir, 'images.txt'), [i+1 for i in range(100,200)], subdir='images')
        # self.train, train_pids = _pluck_cub(osp.join(exdir, 'images.txt'), [i+1 for i in range(100)], subdir='images_crop')
        # self.query, query_pids = _pluck_cub(osp.join(exdir, 'images.txt'), [i+1 for i in range(100,200)], subdir='images_crop')
        # self.gallery, gallery_pids = _pluck_cub(osp.join(exdir, 'images.txt'), [i+1 for i in range(100,200)], subdir='images_crop')
        self.num_train_pids = len(list(set(train_pids)))

        if verbose:
            print(self.__class__.__name__, "dataset loaded")
            print("  subset   | # ids | # images")
            print("  ---------------------------")
            print("  train    | {:5d} | {:8d}"
                  .format(self.num_train_pids, len(self.train)))
            print("  query    | {:5d} | {:8d}"
                  .format(len(query_pids), len(self.query)))
            print("  gallery  | {:5d} | {:8d}"
                  .format(len(gallery_pids), len(self.gallery)))

class CUB(Dataset_CUB):

    def __init__(self, root, split_id=0, download=True):
        super(CUB, self).__init__(root)

        if download:
            self.download()

        self.load()

    def download(self):

        import re
        import hashlib
        import shutil
        from glob import glob
        from zipfile import ZipFile

        raw_dir = osp.join(self.root)
        mkdir_if_missing(raw_dir)

        # Download the raw zip file
        fpath = osp.join(raw_dir, 'CUB_200_2011')
        if osp.isdir(fpath):
            print("Using downloaded file: " + fpath)
        else:
            raise RuntimeError("Please download the dataset manually to {}".format(fpath))
