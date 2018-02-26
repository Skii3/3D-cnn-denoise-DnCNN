#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 15:00:01 2017

@author: edogawachia
"""

from utils import load_MNIST, load_data
from current import experiment
import numpy as np
import os
import scipy.io as sio

#------------------ global settings ------------------#
EXP_TIMES = 1
N_SAMPLES = 200
N_TEST = 40
SAMPLE_SIZE = [300,300]
REL_FILE_PATH = './traindata/'
SAVEPSNR = './savepsnr'
#------------------ global settings ------------------#

data,label = load_data(rel_file_path = REL_FILE_PATH, n_samples = N_SAMPLES, sample_size = SAMPLE_SIZE)
#data,label = load_MNIST()

if not os.path.exists(SAVEPSNR):
    os.makedirs(SAVEPSNR)

for in_para in [3]:#[1,2,3]:
    for window_para in [3,5,7]:
        aver_ori_psnr = np.zeros([N_TEST,1])
        aver_dnd_psnr = np.zeros([N_TEST,1])
        postfix = str(in_para) + 'ch' + str(window_para) + 'win'
        for exp_id in range(EXP_TIMES):
            print('[*] parameter settings: ')
            print('[*] input channel = ' + str(in_para) + ', windowsize = ' + str(window_para))
            print('[*] start experiment --------------->>>')
            ori_psnr, dnd_psnr = experiment( data, label, n_sample=N_SAMPLES, n_test=N_TEST, \
                                            n_imgrow=SAMPLE_SIZE[0], n_imgcol=SAMPLE_SIZE[1], shuffle_button=3,\
                                            in_button=in_para, window_len=window_para)
            aver_ori_psnr = aver_ori_psnr + ori_psnr/EXP_TIMES
            aver_dnd_psnr = aver_dnd_psnr + dnd_psnr/EXP_TIMES
        sio.savemat(os.path.join(SAVEPSNR,'ori_psnr'+ postfix+ '.mat'), \
                    {'ori_psnr'+ postfix: aver_ori_psnr})
        sio.savemat(os.path.join(SAVEPSNR,'dnd_psnr'+ postfix+ '.mat'), \
                    {'dnd_psnr'+ postfix: aver_dnd_psnr})


'''
 parameter setting:
     1. in_para: control import channels[1,2,3]
     2. window_para:control scan window size?
     3. 
'''

















