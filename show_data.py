# -*- coding: utf-8 -*-
import scipy.io as sio
from PIL import Image
import scipy
import glob
import matplotlib.pyplot as pl
file_path = './traindata'
files_name = glob.glob(file_path + '/*.mat')

for file_name in files_name:
    data = sio.loadmat(file_name)
    data_data = data['data']


    '''保存时间剖面x-y的图片''
    for i in range(1,256,1):
        temp = data_data[0:256,0:256,i]
        
        scipy.misc.imsave(file_path + '/x_y/%d.jpg' % i,temp)
    '''
    '''保存x-t剖面的图片''
    for i in range(1,256,1):
        temp = data_data[i,0:256,0:256]
        scipy.misc.imsave(file_path + '/x_t/%d.jpg' % i,temp)
    '''
    '''保存y-t时间剖面的图片''
    for i in range(1,256,1):
        temp = data_data[0:256,i,0:256]
        scipy.misc.imsave(file_path + '/y_t/%d.jpg' % i,temp)
    '''
    '''保存小区块''
    for i in range(100, 201, 1):
        temp = data_data[0:256, 0:256, i]
        for j in range(0,256 - 20 + 1,20):
            for k in range(0,256 - 20 + 1,20):
              temp_temp = temp[j : j + 20, k : k + 20]
              scipy.misc.imsave(file_path + '/x_y_part/%d_%d_%d.jpg' %(i,j,k), temp_temp)
    '''
