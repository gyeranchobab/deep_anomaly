import torch
import numpy as np
from torch.utils.data import Dataset
from glob import glob 
from scipy import io
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.datasets import make_moons
import h5py

class RotorDataset(Dataset):
    def __init__(self, split, p, snr=None, seed=None, shift=None, lb=None, ub=None):
        if seed is not None:
            np.random.seed(seed)
        self.split=split
        self.p=p
        self.snr=snr
        self.shift=shift
        self.seed=seed
        
        fs = glob('./datasets/broken_rotor_bar_data/experimental/*')
        fs.sort()
        self.DATA = []
        self.label = []
        
        self.lb=None
        self.ub=None
        
        self.s = 10
        
        self.n_torque=8
        
        key2y = {'rs':0,'r1b':1,'r2b':2,'r3b':3,'r4b':4}
        test_kkeys = None#[6,7]
        for i,fn in enumerate(fs):
            DATA = []
            dataset = h5py.File(fn,'r')
            for key in dataset:
                if key != '#refs#':
                    data = dataset[key]
                    ykey=key
                    for j, kkey in enumerate(data):
                        if test_kkeys is not None and j in test_kkeys and self.split =='train':
                            continue
                        elif test_kkeys is not None and j not in test_kkeys and self.split == 'test':
                            continue
                        data_torque = data[kkey]
                        multivariate_data = []
                        for kkkey in data_torque:
                            data_torque_feature = data_torque[kkkey]
                            raw = np.array(data[h5py.h5r.get_name(data_torque_feature[0,0],data.id)])

                            downsampled = raw[0,np.linspace((5/18)*raw.shape[1], (15/18)*raw.shape[1]-1, 10*12000).astype(int)]
                            multivariate_data.append(downsampled)
                        multivariate_data = np.stack(multivariate_data,axis=1)
                        
                        if self.split == 'train':
                            multivariate_data = multivariate_data[:12000*8]
                            self.s = 8
                        elif self.split == 'test':
                            multivariate_data = multivariate_data[12000*8:]
                            self.s = 2
                        
                        DATA.append(multivariate_data)
                        minlog= multivariate_data.min(axis=0)
                        maxlog = multivariate_data.max(axis=0)
                        if ykey=='rs':
                            if self.lb is None:
                                self.lb =minlog
                            else:
                                self.lb = np.minimum(minlog, self.lb)
                            if self.ub is None:
                                self.ub = maxlog
                            else:
                                self.ub = np.maximum(maxlog, self.ub)
            self.DATA.append(DATA)
            self.label.append(key2y[ykey])
        self.label = np.array(self.label)
        self.label = np.repeat(self.label, len(self)//5)

        if lb is not None:
            self.lb=lb
        if ub is not None:
            self.ub=ub
        
        SEED=0
        if seed:
            SEED=seed
        np.random.seed(SEED)
        #always_mask=[6,7]
        if self.split in ['train','val']:
            if self.p == 0:
                n_l = 5
            else:
                n_l = int(self.p*len(self))
            while True:
                self.mask = np.random.choice(len(self), len(self), replace=False) < n_l
                if set([0,1,2,3,4]).issubset(set(self.label[self.mask])):
                    break
        elif self.split in ['test']:
            self.mask= np.array([True]*len(self))
        print('%s dataset size:%d, labeled:%d, unlabeled:%d'%(self.split, len(self), self.mask.sum(), (~self.mask).sum()))
        print('labels : [0:%d, 1:%d, 2:%d, 3:%d, 4:%d]'%tuple([(self.mask & (self.label==i)).sum() for i in range(5)]))
    def __len__(self):
        return 5*self.n_torque*(((12000*self.s)-2048)//256)
    def _find_idx(self, index):
        k = index % (((12000*self.s)-2048)//256)
        index = index // (((12000*self.s)-2048)//256)
        j = index % self.n_torque
        index = index // self.n_torque
        i = index % 5
        return i,j,k*256
    def __getitem__(self, index):
        
        i,j,k = self._find_idx(index)
        X = self.DATA[i][j][k:k+2048]
        Y = self.label[index]
        M = self.mask[index]
        
        if self.shift is not None:
            X_shift = self.DATA[i][j][k+self.shift:k+2048+self.shift]
            X_shift = X_shift.T
            
        X= X.T
        if self.snr is not None:
            X = self._add_noise(X, self.snr)                
        if self.shift is not None:
            return X_shift, X,Y,M
        return X,Y,M
    def _add_noise(self,x, snr):
        snr1 = 10**(snr/10.0)
        xpower = np.sum(x**2, axis=-1) / x.shape[-1]#len(x)
        npower = xpower/snr1
        noise = np.random.normal(0, np.sqrt(npower),x.shape)
        noise_data = x+noise
        return noise_data
    
class TwoMoonDataset(Dataset):
    def __init__(self, N, noise=None, ratio=1.0, n_label=None,seed=0):
        self.X, self.y, self.M = twomoon(N,noise,ratio,n_label, seed)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, index):
        return self.X[index],self.y[index],self.M[index]
        
def twomoon(N, noise=0.1, ratio=1.0, n_label=None,seed=0):
    data,label = make_moons(N,shuffle=False,noise=noise,random_state=seed)
    data = (data - data.mean(0,keepdims=True))/data.std(0,keepdims=True)
    l0_idx = (label==0)
    l1_idx = (label==1)
    
    np.random.seed(seed)
    m = np.array([False]*label.shape[0])
    l1 = np.array([False]*l0_idx.sum())
    l2 = np.array([False]*l1_idx.sum())
    
    
    if n_label is not None:
        if type(n_label) is tuple:
            l1[:n_label[0]]=True
            l2[:n_label[1]]=True
            #print(l1[n_label[0]:])
        elif type(n_label) in [int,float]:
            if n_label <1:
                l1[:int(n1*n_label)]=True
                l2[:int(n2*n_label)]=True
            else:
                l1[:n_label]=True
                l2[:n_label]=True
    np.random.shuffle(l1)
    np.random.shuffle(l2)
    m[l0_idx] = l1
    m[l1_idx] = l2
    return data, label, m
