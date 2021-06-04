from __future__ import print_function, absolute_import
import os.path as osp
import tarfile

import glob
import re
import urllib
import zipfile

from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json


def _pluck_sop(list_file):
    with open(list_file, 'r') as f:
        lines = f.readlines()
    ret = []
    pids = []
    for i in range(1, len(lines)):
        line = lines[i]
        line = line.strip()
        elements = line.split(' ')
        imgid = int(elements[0])
        pid = int(elements[1])-1
        path = elements[3]
        if pid not in pids:
            pids.append(pid)
        ret.append((osp.join(path), pid, imgid))
    return ret, pids

class Dataset_SOP(object):
    def __init__(self, root):
        self.root = root
        self.train, self.val, self.trainval = [], [], []
        self.query, self.gallery = [], []
        self.num_train_ids, self.num_val_ids, self.num_trainval_ids = 0, 0, 0

    @property
    def images_dir(self):
        return osp.join(self.root, 'Stanford_Online_Products')

    def load(self, verbose=True):
        exdir = osp.join(self.root, 'Stanford_Online_Products')
        self.train, train_pids = _pluck_sop(osp.join(exdir, 'Ebay_train.txt'))
        self.query, query_pids = _pluck_sop(osp.join(exdir, 'Ebay_test.txt'))
        self.gallery, gallery_pids = _pluck_sop(osp.join(exdir, 'Ebay_test.txt'))
        self.num_train_pids = len(train_pids)

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

class SOP(Dataset_SOP):

    def __init__(self, root, split_id=0, download=True):
        super(SOP, self).__init__(root)

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
        fpath = osp.join(raw_dir, 'Stanford_Online_Products')
        if osp.isdir(fpath):
            print("Using downloaded file: " + fpath)
        else:
            raise RuntimeError("Please download the dataset manually to {}".format(fpath))
