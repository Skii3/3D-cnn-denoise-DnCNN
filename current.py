#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 10:53:04 2017

@author: edogawachia
"""
import time
from utils import cal_psnr #, load_data
from frcnnmodel import FRCNN_model
from numpy import random as nr
import numpy as np
import scipy.io as sio
from myfilterlab import mean_filter
import os
#from keras.callbacks import ModelCheckpoint

##-------------------global settings---------------------------#
#n_sample = 200
#n_test = 40
#n_imgrow = 300
#n_imgcol = 300
#shuffle_button = 3
#in_button = 3 # input mode: 1. [noisy] 2. [meanfiltered][residue] 3. [meanfiltered][residue1][residue2]
#label_button = 2 # label mode: 1.[real] 2. [real][noise]
#thres_button = 0 # hard threshold in preprocessing? 1. yes  0. no
#window_len = 7 # filter size in generating input data of the Neural Network
##-------------------global settings---------------------------#

# load data
#data,label = load_data(n_samples = 200,sample_size = [300,300])

def experiment( data, label, n_sample=200, n_test=40, n_imgrow=300, n_imgcol=300, shuffle_button=3,\
               in_button=3, window_len=7):

    if (shuffle_button == 1):
        # 1.shuffle the whole dataset
        order = nr.permutation(n_sample)
        print("[*] shuffle the whole dataset")
    elif (shuffle_button == 2):
        # 2.do not shuffle
        order = range(n_sample)
        print("[*] do not shuffle")
    elif (shuffle_button == 3):
        # 3. shuffle the training and validation only
        order = np.concatenate((nr.permutation(n_sample-n_test),range(n_sample-n_test,n_sample)),axis = 0)
        print("[*] shuffle the training and validation only")
    else:
        print("[*] shuffle button not confirmed")
    
    shuffledata = data[order,:,:]
    shufflelabel = label[order,:,:]
    # split input data and test data
    in_data = shuffledata[0:(n_sample-n_test)]
    in_label = shufflelabel[0:(n_sample-n_test)]
    t_data = shuffledata[(n_sample-n_test):n_sample,:]
    t_label = shufflelabel[(n_sample-n_test):n_sample,:]
    
    train_data = np.zeros([len(in_data),n_imgrow,n_imgcol,in_button])
    train_label = np.zeros([len(in_label),n_imgrow,n_imgcol,in_button])
    
    t0 = time.time()
    
    for i in range(len(in_data)):
        # generate input data
        if (in_button == 1):
            train_data[i] = in_data[i].reshape([n_imgrow,n_imgcol, 1]) # using the noisy image as input
            train_label[i] = (in_data[i] - in_label[i]).reshape([n_imgrow,n_imgcol, 1]) # using the clean image as output (label)
        elif (in_button == 2):
            in_hardthr = in_data[i]
            train_data_ch1 = mean_filter(in_hardthr,kernelsize=window_len) # channel 1 is for filtered
            train_data_ch2 = in_hardthr-train_data_ch1 # channel 2 is for residue
            train_data[i] = np.stack([train_data_ch1,train_data_ch2],axis=2)

            lab_hardthr = in_data[i] - in_label[i] # channel 1 is for original
            train_label_ch1 = mean_filter(lab_hardthr,kernelsize=window_len) # channel 1 is for filtered
            train_label_ch2 = lab_hardthr - train_label_ch1 # channel 2 is for residue
            train_label[i] = np.stack([train_label_ch1,train_label_ch2],axis=2)            
            
        elif (in_button == 3):
            in_hardthr = in_data[i]
            train_data_ch1 = mean_filter(mean_filter(in_hardthr),kernelsize=window_len) # channel 1 is for filtered twice
            train_data_ch2 = mean_filter(in_hardthr,kernelsize=window_len) - train_data_ch1 # residue 1 (once - twice filtered)
            train_data_ch3 = in_hardthr - mean_filter(in_hardthr,kernelsize=window_len) # residue 2 (original - once filtered)
            train_data[i] = np.stack([train_data_ch1,train_data_ch2,train_data_ch3],axis=2)
            
            lab_hardthr = in_data[i] - in_label[i]
            train_label_ch1 = mean_filter(mean_filter(lab_hardthr),kernelsize=window_len) # channel 1 is for filtered twice
            train_label_ch2 = mean_filter(lab_hardthr,kernelsize=window_len) - train_label_ch1 # residue 1 (once - twice filtered)
            train_label_ch3 = lab_hardthr - mean_filter(lab_hardthr,kernelsize=window_len) # residue 2 (original - once filtered)
            train_label[i] = np.stack([train_label_ch1,train_label_ch2,train_label_ch3],axis=2)

    print("[*] train data ready")
    t1 = time.time()
    print("Total time running: %s seconds" % ( str(t1-t0)) )
    
    t0 = time.time()
    
    test_data = np.zeros([len(t_data),n_imgrow,n_imgcol,in_button])
    test_label = np.zeros([len(t_label),n_imgrow,n_imgcol,in_button])
    for i in range(len(t_data)):
        # generate input data
        if (in_button == 1):
            test_data[i] = t_data[i].reshape([n_imgrow,n_imgcol, 1]) # using the noisy image as input
            test_label[i] = (t_data[i]- t_label[i]).reshape([n_imgrow, n_imgcol, 1]) # using the clean image as output (label)
        elif (in_button == 2):
            t_hardthr = t_data[i]
            test_data_ch1 = mean_filter(t_hardthr,kernelsize=window_len) # channel 1 is for filtered
            test_data_ch2 = t_hardthr-test_data_ch1 # channel 2 is for residue
            test_data[i] = np.stack([test_data_ch1,test_data_ch2],axis=2)
            
            tlab_hardthr = t_data[i]- t_label[i] # channel 1 is for original
            test_label_ch1 = mean_filter(tlab_hardthr,kernelsize=window_len) # channel 1 is for filtered
            test_label_ch2 = tlab_hardthr - test_label_ch1 # channel 2 is for residue
            test_label[i] = np.stack([test_label_ch1,test_label_ch2],axis=2)
            
        elif (in_button == 3):
            t_hardthr = t_data[i]
            test_data_ch1 = mean_filter(mean_filter(t_hardthr),kernelsize=window_len) # channel 1 is for filtered twice
            test_data_ch2 = mean_filter(t_hardthr,kernelsize=window_len) - test_data_ch1 # residue 1 (once - twice filtered)
            test_data_ch3 = t_hardthr - mean_filter(t_hardthr,kernelsize=window_len) # residue 2 (original - once filtered)
            test_data[i] = np.stack([test_data_ch1,test_data_ch2,test_data_ch3],axis=2)
            
            tlab_hardthr = t_data[i] - t_label[i]
            test_label_ch1 = mean_filter(mean_filter(tlab_hardthr),kernelsize=window_len) # channel 1 is for filtered twice
            test_label_ch2 = mean_filter(tlab_hardthr,kernelsize=window_len) - test_label_ch1 # residue 1 (once - twice filtered)
            test_label_ch3 = tlab_hardthr - mean_filter(tlab_hardthr,kernelsize=window_len) # residue 2 (original - once filtered)
            test_label[i] = np.stack([test_label_ch1,test_label_ch2,test_label_ch3],axis=2)            


    print("[*] test data ready")
    t1 = time.time()
    print("Total time running: %s seconds" % ( str(t1-t0)) )
    

    CNNclass = FRCNN_model(image_size=[n_imgrow,n_imgcol], in_channel=in_button)
    
    model = CNNclass.build_model()
    
    model,hist = CNNclass.train_model(model,train_data,train_label)
    denoised = CNNclass.test_model(model,test_data)
    
    output = open('log.txt', 'w+')
    output.write(hist.history['loss'])
    output.close
    
    # calculate the PSNR of this experiment
    ori_psnr = np.zeros([n_test,1])
    dnd_psnr = np.zeros([n_test,1])
    
#    if os.path.exists('./tobedown'):
#        os.removedirs('./tobedown')
    
    for i in range(n_test):
        noisy_img = t_data[i]
        denoised_img = (denoised[i,:,:,:].sum(axis=2)).reshape([n_imgrow,n_imgcol])
        real_img = (t_data[i]-t_label[i]).reshape([n_imgrow,n_imgcol])
        ori_psnr[i] = cal_psnr(real_img,noisy_img)
        dnd_psnr[i] = cal_psnr(real_img,denoised_img)
        '''
        print("the {0:d}th test image : ".format(i))
        print("---> original PSNR is {0:.4f} dB".format(cal_psnr(real_img,noisy_img)))
        print("---> denoised PSNR is {0:.4f} dB".format(cal_psnr(real_img,denoised_img)))
        print("the different frequency PSNR of {0:d}th test image : ".format(i))
        if (in_button == 1):
            print("---- no frequency segmentation ----")
        elif (in_button == 2):
            print("----> original PSNR of smooth is {0:.4f} dB".format(cal_psnr(test_data[i,:,:,0],test_label[i,:,:,0])))
            print("----> original PSNR of residue is {0:.4f} dB".format(cal_psnr(test_data[i,:,:,1],test_label[i,:,:,1])))
            print("----> denoised PSNR of smooth is {0:.4f} dB".format(cal_psnr(denoised[i,:,:,0],test_label[i,:,:,0])))
            print("----> denoised PSNR of residue is {0:.4f} dB".format(cal_psnr(denoised[i,:,:,1],test_label[i,:,:,1])))
        elif (in_button == 3):
            print("----> original PSNR of smooth is {0:.4f} dB".format(cal_psnr(test_data[i,:,:,0],test_label[i,:,:,0])))
            print("----> original PSNR of residue1 is {0:.4f} dB".format(cal_psnr(test_data[i,:,:,1],test_label[i,:,:,1])))
            print("----> original PSNR of residue2 is {0:.4f} dB".format(cal_psnr(test_data[i,:,:,2],test_label[i,:,:,2])))
            print("----> denoised PSNR of smooth is {0:.4f} dB".format(cal_psnr(denoised[i,:,:,0],test_label[i,:,:,0])))
            print("----> denoised PSNR of residue1 is {0:.4f} dB".format(cal_psnr(denoised[i,:,:,1],test_label[i,:,:,1])))
            print("----> denoised PSNR of residue2 is {0:.4f} dB".format(cal_psnr(denoised[i,:,:,2],test_label[i,:,:,2])))       
        '''
    print("---> original PSNR is %.4f dB" % np.mean(ori_psnr))
    print("---> denoised PSNR is %.4f dB" % np.mean(dnd_psnr))

    output = open('log.txt', 'w+')
    output.write("---> original PSNR is %.4f dB\n" % np.mean(ori_psnr))
    output.write("---> denoised PSNR is %.4f dB\n" % np.mean(dnd_psnr))
    output.close
    
    # save experiment results
    savingpath = './tobedown/tobedown_in' + str(in_button) + '_winlen' + str(window_len)
    if not os.path.exists(savingpath):
        os.makedirs(savingpath)
    postfix = str(in_button) + '_' + str(window_len)
    sio.savemat(os.path.join(savingpath,'denoised'+ postfix+ '.mat'), \
                {'denoised'+ postfix: denoised.sum(axis=3).reshape([n_test,n_imgrow,n_imgcol])})  
    sio.savemat(os.path.join(savingpath,'denoised_ch1'+ postfix+ '.mat'), \
                {'denoised1'+ postfix: denoised[:,:,:,0].reshape([n_test,n_imgrow,n_imgcol])})  
    if (in_button > 1):
        sio.savemat(os.path.join(savingpath,'denoised_ch2'+ postfix+ '.mat'), \
                {'denoised2'+ postfix: denoised[:,:,:,1].reshape([n_test,n_imgrow,n_imgcol])})  
    if (in_button > 2):
        sio.savemat(os.path.join(savingpath,'denoised_ch3'+ postfix+ '.mat'), \
                {'denoised3'+ postfix: denoised[:,:,:,2].reshape([n_test,n_imgrow,n_imgcol])})  
    
    sio.savemat(os.path.join(savingpath,'noisy'+ postfix+ '.mat'), \
                {'noisy'+ postfix: t_data})  
    sio.savemat(os.path.join(savingpath,'real'+ postfix+ '.mat'), \
                {'real'+ postfix: (t_data-t_label).reshape([n_test,n_imgrow,n_imgcol])})  
    sio.savemat(os.path.join(savingpath,'ori_psnr'+ postfix+ '.mat'), \
                {'ori_psnr'+ postfix: ori_psnr})  
    sio.savemat(os.path.join(savingpath,'dnd_psnr'+ postfix+ '.mat'), \
                {'dnd_psnr'+ postfix: dnd_psnr})
    
    return ori_psnr, dnd_psnr
    
    