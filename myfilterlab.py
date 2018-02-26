#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 11:53:24 2017

@author: edogawachia
"""
import numpy as np
import sys
from numpy.matlib import repmat

def general_filter(img,fil):    
    img_height = img.shape[0]
    img_width = img.shape[1]
    fil_height = fil.shape[0]
    fil_width = fil.shape[1]

    if (np.mod(fil_height,2) == 0 or np.mod(fil_width,2) == 0):
        print("[*] error: filter size must be odd int")         
        sys.exit(1)
    zeropadding = np.zeros([img_height+fil_height-1,img_width+fil_width-1],dtype=float)
    zeropadding[(fil_height-1)/2:((fil_height-1)/2+img_height),(fil_width-1)/2:((fil_width-1)/2+img_width)] = img    
    zeropadding[0:(fil_height-1)/2,(fil_width-1)/2:((fil_width-1)/2+img_width)] = repmat(img[0,:],(fil_height-1)/2,1)
    zeropadding[((fil_height-1)/2+img_height):,(fil_width-1)/2:((fil_width-1)/2+img_width)] = repmat(img[-1,:],(fil_height-1)/2,1)
    zeropadding[(fil_height-1)/2:((fil_height-1)/2+img_height),0:(fil_width-1)/2] = np.transpose(repmat(img[:,0],(fil_width-1)/2,1))
    zeropadding[(fil_height-1)/2:((fil_height-1)/2+img_height),((fil_width-1)/2+img_width):] = np.transpose(repmat(img[:,-1],(fil_width-1)/2,1))
    smoothed = np.zeros([img_height,img_width],dtype=float)
    for i in range(img_height):
        for j in range(img_width):
            smoothed[i,j] = mul_elementwise(fil,zeropadding[i:i+fil_height,j:j+fil_width])            
    return smoothed

def mul_elementwise(a,b):
    return (a * b).sum()

def mean_filter(img,kernelsize=3):
    fil = (1.0/kernelsize ** 2)*np.ones([kernelsize,kernelsize])
    return general_filter(img,fil)

def median_filter(img,kernelsize=3):    
    img_height = img.shape[0]
    img_width = img.shape[1]

    if (np.mod(kernelsize,2) == 0):
        print("[*] error: filter size must be odd int")
        sys.exit()
    zeropadding = np.zeros([img_height+kernelsize-1,img_width+kernelsize-1],dtype=float)
    zeropadding[(kernelsize-1)/2:((kernelsize-1)/2+img_height),(kernelsize-1)/2:((kernelsize-1)/2+img_width)] = img
    smoothed = np.zeros([img_height,img_width],dtype=float)
    for i in range(img_height):
        for j in range(img_width):
            smoothed[i,j] = np.median(zeropadding[i:i+kernelsize,j:j+kernelsize])
    return smoothed
    
def laplacian_filter(img,kernel='type1'):
    
    if (kernel=='type1'):
        fil = np.array([[1,1,1],[1,-8,1],[1,1,1]])
    elif (kernel=='type2'):
        fil = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    elif (kernel=='type3'):
        fil = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    elif (kernel=='type4'):
        fil = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
    else:
        print("[*] error: filter kernel type incorrect")
        sys.exit()
    
    return general_filter(img,fil)

def x_sobel_filter(img):
    fil = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    return general_filter(img,fil)

def y_sobel_filter(img):
    fil = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    return general_filter(img,fil)








    
    
    
    
    
    
    
    
    