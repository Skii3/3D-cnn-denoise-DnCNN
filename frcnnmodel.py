#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 10:39:19 2017

@author: edogawachia
"""

from keras.models import Sequential
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
#from keras import optimizers
from keras.callbacks import ModelCheckpoint
class FRCNN_model(object):
    # construct function 
    def __init__(self,batch_size=10,epoch=2,image_size=[300,300],nb_filter=16,kernel_size=3,in_channel=2):
        self.batch_size = batch_size
        self.epoch = epoch
        self.image_row = image_size[0]
        self.image_col = image_size[1]
        self.nb_filter = nb_filter
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        
        self.build_model()
        
    # build FRCNN denoising model
    def build_model(self):
        model = Sequential()
        # first layer
        model.add(Conv2D(self.nb_filter,self.kernel_size,strides=1,
                         padding='same',data_format='channels_last',
                         input_shape = (self.image_row,self.image_col,self.in_channel)))
        model.add(Activation('relu'))
        # layer 2 to 16
        # layer 2
        model.add(Conv2D(self.nb_filter,self.kernel_size,strides=1,padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        # layer 3
        model.add(Conv2D(self.nb_filter,self.kernel_size,strides=1,padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        # layer 4
        model.add(Conv2D(self.nb_filter,self.kernel_size,strides=1,padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        # layer 5
        model.add(Conv2D(self.nb_filter,self.kernel_size,strides=1,padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))              
        # layer 6
        model.add(Conv2D(self.nb_filter,self.kernel_size,strides=1,padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        # layer 7
        model.add(Conv2D(self.nb_filter,self.kernel_size,strides=1,padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        # layer 8
        model.add(Conv2D(self.nb_filter,self.kernel_size,strides=1,padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        # layer 9
        model.add(Conv2D(self.nb_filter,self.kernel_size,strides=1,padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        # layer 10
        model.add(Conv2D(self.nb_filter,self.kernel_size,strides=1,padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        # layer 11
        model.add(Conv2D(self.nb_filter,self.kernel_size,strides=1,padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        # layer 12
        model.add(Conv2D(self.nb_filter,self.kernel_size,strides=1,padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        # layer 13
        model.add(Conv2D(self.nb_filter,self.kernel_size,strides=1,padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        # layer 14
        model.add(Conv2D(self.nb_filter,self.kernel_size,strides=1,padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        # layer 15
        model.add(Conv2D(self.nb_filter,self.kernel_size,strides=1,padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        # layer 16
        model.add(Conv2D(self.nb_filter,self.kernel_size,strides=1,padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        # last layer
        model.add(Conv2D(self.in_channel,self.kernel_size,strides=1,padding='same'))

        model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['acc'])
        
        print("[*] model construction success")
        return model

    def train_model(self,model,train_data,train_label):
        
        print("[*] begin to train model")
        model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', save_best_only=True)
        train_hist = model.fit(train_data,train_label,
                            batch_size=self.batch_size,epochs=self.epoch,
                            validation_split=0.1,shuffle=True,callbacks=[model_checkpoint],verbose = 2)
        print("[*] saving model weights")
        model.save_weights('model_weights.h5',overwrite=True)
        return model, train_hist
    
    def test_model(self,model,test_data):
        
        print("[*] begin to test learned model")
#        noise = model.predict(test_data,verbose=1)
#        denoised = test_data - noise
        denoised = model.predict(test_data,verbose=1)
        return denoised