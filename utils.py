#coding=utf-8
import scipy.io as sio
#import matplotlib.pyplot as plt
import os,sys
import numpy as np
import glob
import scipy
import random
def load_data(rel_file_path = './traindata/',
              start_point=[80,80,80],
              end_point=[80,80,80],
              patch_size=[10,10,10],
              stride=[1,1,1],
              traindata_save='./traindata_save'):
    train_data = []
    train_data_noise = []
    test_data = []
    files_name = glob.glob(rel_file_path + '/*.mat')
    index = 1
    print "[*] start loading data"
    for file_name in files_name:
        data = sio.loadmat(file_name)
        if index == 1:
            data_data = data['C']                # 256*256*256
        else:
            data_data = data['C2']
        index = index + 1
        '''extract patches for training'''
        data_data = np.transpose(data_data,[1,2,0])
        for i in range(start_point[0], end_point[0]-patch_size[0]+1, stride[0]):
            for j in range(start_point[1], end_point[1]-patch_size[1]+1, stride[1]):
                for k in range(start_point[2],end_point[2]-patch_size[2]+1, stride[2]):
                    train_temp = data_data[i:i+patch_size[0], j:j+patch_size[1], k:k+patch_size[2]]

                    # normalize
                    #train_temp_mean = np.mean(train_temp)
                    #train_temp_std = np.std(train_temp)
                    #train_temp = (train_temp - train_temp_mean) / train_temp_std

                    # first rotate
                    aug_type1 = random.randint(1,6)    # rotate to different phases
                    aug_type2 = random.randint(1,8)     # different rotation and flip
                    if aug_type1 == 2:
                        train_temp = np.rot90(train_temp, k=1, axes=(0,1))
                    elif aug_type1 == 3:
                        train_temp = np.rot90(train_temp, k=2, axes=(0,1))
                    elif aug_type1 == 4:
                        train_temp = np.rot90(train_temp, k=3, axes=(0,1))
                    elif aug_type1 == 5:
                        train_temp = np.rot90(train_temp, k=1, axes=(0,2))
                    elif aug_type1 == 6:
                        train_temp = np.rot90(train_temp, k=3, axes=(0,2))

                    if aug_type2 == 2:
                        train_temp = np.rot90(train_temp, k=1, axes=(1,2))
                    elif aug_type2 == 3:
                        train_temp = np.rot90(train_temp, k=2, axes=(1,2))
                    elif aug_type2 == 4:
                        train_temp = np.rot90(train_temp, k=3, axes=(1,2))
                    else:
                        train_temp = np.flip(train_temp, axis=1)
                        if aug_type2 == 6:
                            train_temp = np.rot90(train_temp, k=1, axes=(1,2))
                        elif aug_type2 == 7:
                            train_temp = np.rot90(train_temp, k=2, axes=(1,2))
                        elif aug_type2 == 8:
                            train_temp = np.rot90(train_temp, k=3, axes=(1,2))

                    # second intensity aug
                    '''
                    intensity_aug = random.randint(1, 5)
                    if intensity_aug == 2:
                        train_temp = train_temp * np.sqrt(np.sqrt(np.abs(train_temp) + 1e-12))
                        train_temp = train_temp / np.max(train_temp)
                    elif intensity_aug == 3:
                        train_temp = train_temp / np.sqrt(np.sqrt(np.abs(train_temp) + 1e-12))
                        train_temp = train_temp / np.max(train_temp)
                    '''

                    train_temp = (train_temp - np.mean(train_temp)) / np.std(train_temp)
                    train_data.append(train_temp)

                    #scipy.misc.imsave(traindata_save + '/%d_%d_%d_labeldata.jpg' % (i, j, k), train_temp[0, :, :])

                    # third add noise
                    '''
                    noise_aug = random.randint(1,2)
                    if noise_aug == 1:
                        ref_value = np.max(np.abs(train_temp))
                        noise_temp = train_temp + np.random.normal(0, random.randint(1, 10) * 1e-2 * ref_value, train_temp.shape)
                    elif noise_aug == 2:
                        ref_value = np.mean(np.abs(train_temp))
                        noise_temp = train_temp + np.random.normal(0, random.randint(1, 15) * 1e-1 * ref_value,
                                                                   train_temp.shape)
                    '''


                    ref = np.max(train_temp)
                    noise_level = random.randint(1,30) * 1e-2
                    noise_temp = np.random.normal(0, noise_level * ref, train_temp.shape) + train_temp

                    train_data_noise.append(noise_temp)

                    #print '/%d_%d_%d_noisedata.jpg:' %(i,j,k), std_train_temp
                    #scipy.misc.imsave(traindata_save + '/%d_%d_%d_noisedata.jpg' %(i,j,k), noise_temp[0,:,:])

        test_data.append(data_data[:,:,end_point[2]:])

    train_data = np.array(train_data)
    test_data = np.array(test_data)
    train_data_noise = np.array(train_data_noise)
    print "[*] load data down"
    return train_data,train_data_noise, test_data

    
    
    
    
    
    
    
    
    
    
    





