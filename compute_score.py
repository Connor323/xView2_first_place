import os

from os import path, makedirs, listdir
import sys
import numpy as np
np.random.seed(1)
import random
random.seed(1)

import torch
from torch import nn
from torch.backends import cudnn

from torch.autograd import Variable

import pandas as pd
from tqdm import tqdm
import timeit
import cv2
from sklearn.model_selection import train_test_split

from zoo.models import Res34_Unet_Loc

from utils import *

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

res_folder = 'submission'
all_files = np.array(get_files())
val_idxs = train_test_split(np.arange(len(all_files)).astype(int), test_size=0.1, random_state=0)[1]

class ValData:
    def __init__(self, image_idxs):
        super().__init__()
        self.image_idxs = image_idxs

    def __len__(self):
        return len(self.image_idxs)

    def __getitem__(self, idx):
        _idx = self.image_idxs[idx]

        fn = all_files[_idx]
        msk_loc_pred = cv2.imread(path.join(res_folder, '{0}.png'.format(fn.split('/')[-1].replace('_pre', '_localization').replace('.jpg', '_prediction'))), cv2.IMREAD_UNCHANGED) 
        msk_cls_pred = cv2.imread(path.join(res_folder, '{0}.png'.format(fn.split('/')[-1].replace('_pre', '_damage').replace('.jpg', '_prediction'))), cv2.IMREAD_UNCHANGED) 
        msk_loc = cv2.imread(fn.replace('/images/', '/masks/').replace("jpg", "png"), cv2.IMREAD_UNCHANGED)
        msk_cls = cv2.imread(fn.replace('/images/', '/masks/').replace('_pre_disaster', '_change').replace("jpg", "png"), cv2.IMREAD_UNCHANGED)

        msk_cls[msk_cls == 5] = 1
        msk_loc[msk_loc != 0] = 1
        msk_loc_pred[msk_loc_pred != 0] = 1

        sample = {'msk_loc_pred': msk_loc_pred, 'msk_cls_pred': msk_cls_pred, 'msk_cls': msk_cls, 'msk_loc': msk_loc}
        return sample


def validate(data_loader):
    dices0 = []

    tp = np.zeros((5,))
    fp = np.zeros((5,))
    fn = np.zeros((5,))

    _thr = 0.3

    with torch.no_grad():
        for i, sample in enumerate(tqdm(data_loader)):
            msk_cls = sample["msk_cls"]
            msk_loc = sample["msk_loc"]
            msk_cls_pred = sample["msk_cls_pred"]
            msk_loc_pred = sample["msk_loc_pred"]
            
            tp[4] += np.logical_and(msk_loc > 0, msk_loc_pred > 0).sum()
            fn[4] += np.logical_and(msk_loc < 1, msk_loc_pred > 0).sum()
            fp[4] += np.logical_and(msk_loc > 0, msk_loc_pred < 1).sum()


            targ = msk_cls[msk_loc > 0]
            msk_cls_pred = msk_cls_pred * (msk_loc_pred > _thr)
            msk_cls_pred = msk_cls_pred[msk_loc > 0]
            for c in range(4):
                tp[c] += np.logical_and(msk_cls_pred == c+1, targ == c+1).sum()
                fn[c] += np.logical_and(msk_cls_pred != c+1, targ == c+1).sum()
                fp[c] += np.logical_and(msk_cls_pred == c+1, targ != c+1).sum()

    d0 = 2 * tp[4] / (2 * tp[4] + fp[4] + fn[4])

    f1_sc = np.zeros((4,))
    for c in range(4):
        f1_sc[c] = 2 * tp[c] / (2 * tp[c] + fp[c] + fn[c])

    f1 = 4 / np.sum(1.0 / (f1_sc + 1e-6))

    sc = 0.3 * d0 + 0.7 * f1
    print("Val Score: {}, Dice: {}, F1: {}, F1_0: {}, F1_1: {}, F1_2: {}, F1_3: {}".format(sc, d0, f1, f1_sc[0], f1_sc[1], f1_sc[2], f1_sc[3]))

if __name__ == "__main__":
    val_train = ValData(val_idxs)
    best_score = validate(val_train)