import os
import logging
import time
import glob
import cv2
import numpy as np
import torch

class DIMEvaluationDataset(torch.utils.data.Dataset):
    def __init__(self, datapath, input_shape):
        super(DIMEvaluationDataset, self).__init__()
        _line = [i.strip().split(' ') for i in open(os.path.join(datapath, 'val_set.txt'), 'r')]
        self.imgpath = [os.path.join(datapath, i[0]) for i in _line]
        self.tripath = [os.path.join(datapath, i[-1]) for i in _line]
        assert len(self.imgpath) == len(self.tripath)        
        self.length = len(self.imgpath)
        self.img_shape = [None] * self.length
        self.input_shape = input_shape
        for i in range(self.length):
            _i = cv2.imread(self.imgpath[i])
            _t = cv2.imread(self.tripath[i], cv2.IMREAD_GRAYSCALE)
            assert _t is not None, "Cannot find file: "+self.tripath[i]
            assert _i is not None, "Cannot find file: "+self.imgpath[i]
            assert _i.shape[:2] == _t.shape[:2]
            self.img_shape[i] = _t.shape[:2]
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = cv2.imread(self.imgpath[idx], cv2.IMREAD_COLOR)
        tri = cv2.imread(self.tripath[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, dsize=(self.input_shape[1], self.input_shape[0]))
        tri = cv2.resize(tri, dsize=(self.input_shape[1], self.input_shape[0]), interpolation=cv2.INTER_NEAREST)
        small_tri = cv2.resize(tri, dsize=(self.input_shape[1]//8, self.input_shape[0]//8), interpolation=cv2.INTER_NEAREST)
        
        img = np.float32(img) / 127.5 - 1.                  # [0, 255] to [-1, 1]
        tri = np.float32(tri // 127)[..., np.newaxis]       # [0, 255] to {0, 1, 2}
        small_tri = np.float32(small_tri // 127)[..., np.newaxis]
        
        img = torch.from_numpy(img.transpose([2, 0, 1])).float()
        tri = torch.from_numpy(tri.transpose([2, 0, 1])).float()
        small_tri = torch.from_numpy(small_tri.transpose([2, 0, 1])).float()
        idx = torch.from_numpy(np.array(idx)).int()
        return img, tri, small_tri, idx