#coding=utf-8
import scipy.io as sio
#import matplotlib.pyplot as plt
import os,sys
import numpy as np
import glob
import scipy

def load_data(rel_file_path = './traindata/',
              start_point=[80,80,80],
              sample_size=[80,80,80],
              patch_size=[10,10,10],
              stride=[1,1,1]):
    train_data = []
    test_data = []
    files_name = glob.glob(rel_file_path + '/*.mat')
    index = 1
    for file_name in files_name:
        data = sio.loadmat(file_name)
        if index == 1:
            data_data = data['C']                # 256*256*256
        else:
            data_data = data['C2']
        index = index + 1
        '''extract patches for training'''
        for i in range(start_point[0], start_point[0]+sample_size[0]-patch_size[0]+1, stride[0]):
            for j in range(start_point[1], start_point[1]+sample_size[1]-patch_size[1]+1, stride[1]):
                for k in range(start_point[2],start_point[2]+sample_size[2]-patch_size[2]+1, stride[2]):
                    train_temp = data_data[i:i+patch_size[0], j:j+patch_size[1], k:k+patch_size[2]]
                    train_data.append(train_temp)
                    #scipy.misc.imsave(rel_file_path + '/x_y_part/%d_%d_%d.jpg' %(i,j,k), train_temp[0,:,:])

        test_data.append(data_data)

    train_data = np.array(train_data)
    test_data = np.array(test_data)

    return train_data, test_data

    
    
    
    
    
    
    
    
    
    
    





