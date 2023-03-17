import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
import scipy
import random

data_path = 'data_raw/'
output_path = 'data/Skeletons'
max_frame = 10000
nb_keep_max = 100000 #number of sample to keep for each class

random.seed(2022)
for file in os.listdir(data_path+'/male'):
    datas_m = loadmat(data_path+'/male/'+file)
    datas_f = loadmat(data_path+'/female/'+file)
    nb_test = 0
    nb_keep = 0
    for s in datas_m:
        if s!='__header__' and s!='__version__' and s!='__globals__' :
            data_m= datas_m[s]
            data_f = datas_f[s]
            nb_sub = (data_m.shape[0]//max_frame)+1
            for k in range(nb_sub):
                if k==nb_sub-1:
                    max_f = data_m.shape[0]%max_frame
                else:
                    max_f = max_frame
                Skeleton_A = np.zeros([45,max_f])
                Skeleton_B = np.zeros([45,max_f])
                for i in range (max_f):
                    skel_m = data_m[k*max_frame+i,:,:]
                    skel_f = data_f[k*max_frame+i,:,:]
                    skel_m_2 = np.concatenate([skel_m[:,0],skel_m[:,1]])
                    skel_m_2 = np.concatenate ([skel_m_2,skel_m[:,2]])
                    skel_f_2 = np.concatenate([skel_f[:,0],skel_f[:,1]])
                    skel_f_2 = np.concatenate ([skel_f_2,skel_f[:,2]])
                    Skeleton_A[:,i] = np.transpose(skel_m_2)
                    Skeleton_B[:,i] = np.transpose(skel_f_2)
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                if not os.path.exists(output_path+'/'+file[0:-4]):
                    os.makedirs(output_path+'/'+file[0:-4])
                if not os.path.exists(output_path+'/'+file[0:-4]+'/'+s):
                    os.makedirs(output_path+'/'+file[0:-4]+'/'+s)
                scipy.io.savemat(output_path+'/'+file[0:-4]+'/'+s+'/skeleton_A.mat', dict([('skeletons',Skeleton_A )]))
                scipy.io.savemat(output_path+'/'+file[0:-4]+'/'+s+'/skeleton_B.mat', dict([('skeletons',Skeleton_B )]))