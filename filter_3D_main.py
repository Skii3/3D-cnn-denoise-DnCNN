#coding=utf-8

from utils import load_data
import numpy as np
from network_model import unet_3d_model
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import math
import scipy
import random
from show_data import kernelshow
#------------------ global settings ------------------#
REL_FILE_PATH = './plutdata'
TRAINDATA_SAVE_PATH = './traindata_save'
SAVEPSNR = './savepsnr'
ind1 = random.randint(0,99)
ind2 = random.randint(0,15)
start_point = [ind1,ind1,ind2]
end_point = [876,900,100]
stride = [100,100,16]
max_epochs = 500
step_decay = 1000
decay_rate = 0.5
lr = 1e-8
beta1 = 0.5
bn_select = 0
batch_size = 40
kernel_size = 4
n_kernel = 3
num_filter = 16
prelu = True
# train/test/onetest/show_kernel
mode = 'train'
if mode == 'train':
    patch_size = [40, 40, 40]
else:
    patch_size = [40, 40, 40]
    batch_size = 40
#------------------ global settings ------------------#

tf.device('/gpu:0')

CNNclass = unet_3d_model(batch_size=batch_size,
                                 input_size=patch_size,
                                 kernel_size=kernel_size,
                                 in_channel=1,
                                 num_filter=num_filter,
                                 stride=[1,1,1],
                                 epochs=2)
input = tf.placeholder('float32', [None, patch_size[0], patch_size[1], patch_size[2], 1], name='input')
target = tf.placeholder('float32', [None, patch_size[0], patch_size[1], patch_size[2], 1], name='target')

if mode == 'train':
        print "tv + l1 loss, conv"+str(n_kernel)
        print "bn_select:",bn_select
        print "start point without random"
        print "end_point:",end_point
        print "stride:",stride
        print "max_epochs:",max_epochs
        print "lr:",lr
        print "decay_rate:",decay_rate
        print "step_decay",step_decay
        print "beta1:",beta1
        print "batch_size:",batch_size
        print "patch_size:",patch_size
        print "num_filter:",num_filter
        print "kernel_size:",kernel_size

        output, loss, l1_loss, tv_loss, snr = CNNclass.build_model2(input, target, True,bn_select,prelu)

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(lr, global_step,
                                                   step_decay, 0.5, staircase=True)
        train_vars = tf.trainable_variables()
        vars_forward = [var for var in train_vars if 'net' in var.name]

        #optim_forward = tf.train.AdamOptimizer(learning_rate=lr,beta1=beta1).minimize(loss)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        optim_forward = optimizer.minimize(loss, global_step=global_step,var_list=vars_forward)

        with tf.name_scope('summaries'):
            tf.summary.scalar('learning rate',learning_rate)
        sess = tf.Session()

        saver = tf.train.Saver()
        merged = tf.summary.merge_all(key='summaries')
        train_writer = tf.summary.FileWriter('./log',sess.graph)
        sess.run(tf.global_variables_initializer())

        ind1 = 0  # random.randint(0, 99)
        ind2 = 0  # random.randint(0, 15)
        start_point = [ind1, ind1, ind2]

        data_label_epoch, data_epoch, _ = load_data(rel_file_path=REL_FILE_PATH,
                                                    start_point=start_point,
                                                    end_point=end_point,
                                                    patch_size=patch_size,
                                                    stride=stride,
                                                    traindata_save=TRAINDATA_SAVE_PATH)
        data_epoch = np.expand_dims(data_epoch, axis=4)
        data_label_epoch = np.expand_dims(data_label_epoch, axis=4)
        g = tf.get_default_graph()
        print("---------------------------training model---------------------------")
        for epoch in range(0, max_epochs + 1):
            if (epoch) % 20 == 0:
                kernelshow(g,n_kernel, sess, epoch,bn_select)
            if (epoch) % 1 == 0:
                ind = np.arange(np.shape(data_epoch)[0])
                ind = np.random.permutation(ind)
                data_test = data_epoch[ind[:2],:,:,:,:]
                data_label_test = data_label_epoch[ind[:2],:,:,:,:]
                denoise_test,summary = sess.run([output, merged],feed_dict={input:data_test, target:data_label_test})
                train_writer.add_summary(summary,epoch)
                for j in range(np.shape(data_test)[0]):
                    for i in range(4):
                        indd = random.randint(0,np.shape(data_test)[3]-1)
                        temp1 = denoise_test[j,:,:,indd,0]
                        temp2 = data_test[j,:,:,indd,0]
                        temp3 = data_label_test[j,:,:,indd,0]
                        if i == 0:
                            result = np.concatenate((temp1.squeeze(),temp2.squeeze(),temp3.squeeze()),axis=1)
                        else:
                            temp = np.concatenate((temp1.squeeze(),temp2.squeeze(),temp3.squeeze()),axis=1)
                            result = np.concatenate((result,temp),axis=0)
                scipy.misc.imsave('./train_result' + '/denoise_noisedata_label%d.png' % epoch, result)

            epoch_time = time.time()
            ind = np.arange(np.shape(data_epoch)[0])
            ind = np.random.permutation(ind)
            data_epoch = data_epoch[ind,:,:,:,:]
            data_label_epoch = data_label_epoch[ind,:,:,:,:]
            sum_all_loss, sum_tvDiff_loss, sum_L1_loss, sum_snr = 0, 0, 0, 0
            n_iter = np.shape(data_epoch)[0] // batch_size
            for step in range(0, n_iter,1):
                data_step = data_epoch[step:step+batch_size,:,:,:,:]
                data_label_step = data_label_epoch[step:step+batch_size,:,:,:,:]
                '''
                # show train data per step
                for data_i in range(batch_size):
                    for data_i2 in range(np.shape(data_step)[3]):
                        scipy.misc.imsave('./train_data_step' + '/%d_%dnoisedata.png' % (data_i, data_i2),np.squeeze(data_step[data_i,:,:,data_i2,:]))
                        scipy.misc.imsave('./train_data_step' + '/%d_%dlabeldata.png' % (data_i, data_i2),
                                          np.squeeze(data_label_step[data_i, :, :, data_i2, :]))
                '''
                tvDiff_loss,L1_loss,_,SNR, lr = sess.run(\
                    [tv_loss,l1_loss,optim_forward,snr,learning_rate],\
                    feed_dict={input:data_step, target:data_label_step})
                sum_all_loss = sum_all_loss + tvDiff_loss + L1_loss
                sum_tvDiff_loss = sum_tvDiff_loss + tvDiff_loss
                sum_L1_loss = sum_L1_loss + L1_loss
                sum_snr = sum_snr + SNR
                print("[*] Step %d: all loss: %.8f, tvDiff loss: %.8f, l1 loss: %.8f, snr: %.2f, lr:%.8f") \
                     % (step,tvDiff_loss+L1_loss, tvDiff_loss, L1_loss, SNR, lr)
            print ("[*] Epoch [%2d/%2d] %4d time: %4.4fs, sum all loss: %.8f, sum tvDiff loss: %.8f, sum l1 loss: %.8f, sum snr: %.8f") \
                  % (epoch+1,max_epochs,np.shape(data_epoch)[0] // batch_size,
                     time.time()-epoch_time,sum_all_loss/n_iter,sum_tvDiff_loss/n_iter,sum_L1_loss/n_iter,sum_snr/n_iter)

            if (epoch+1) % 100 == 0:
                tf.train.Saver().save(sess, './model_save' + '/model%d' % epoch)

elif mode == 'test':
    data = np.expand_dims(test_data,4)
    sigma_random = np.random.randint(0, np.shape(noise_sigma)[0], 1)
    max_intensity = np.max(data)
    rand_data = np.random.normal(0,
                                 noise_sigma[sigma_random[0]] * max_intensity,
                                 patch_size)
    data_noise = data + np.expand_dims(rand_data,4)
    denoised = CNNclass.test_model(model=model,test_data=data_noise)

    i = 0
    for j in range(0,np.shape(denoised)[1],40):
        temp = denoised[i,:,:,:]
        temp2 = data_noise[i,:,:,:,0]
        temp3 = data[i,:,:,:,0]
        plt.figure(3*(j-1))
        plt.imshow(temp[:,j,:].reshape(256, 256), cmap='gray')
        plt.figure(3*(j-1)+1)
        plt.imshow(temp2[:, j , :].reshape(256, 256), cmap='gray')
        plt.figure(3*(j-1)+2)
        plt.imshow(temp3[:, j, :].reshape(256, 256), cmap='gray')
        plt.show()
elif mode == 'onetest':

    output,_,_,_,_ = CNNclass.build_model2(input, target, True,bn_select)

    _, _, test_data = load_data(rel_file_path=REL_FILE_PATH,
                                start_point=start_point,
                                end_point=end_point,
                                patch_size=patch_size,
                                stride=stride,
                                traindata_save=TRAINDATA_SAVE_PATH)
    onedata = np.concatenate((test_data[0,:,:,:],test_data[1,:,:,:]),axis=2)    # 876*900*160
    onedata_test = onedata[:,:,:patch_size[2]]

    ref_value = np.max(np.abs(onedata_test))
    onedata_test_noise = onedata_test + np.random.normal(0, random.randint(1, 10) * 1e-2 * ref_value, onedata_test.shape)

    onedata_test_extract = []
    for i in range(0,np.shape(onedata_test)[0]-patch_size[0]+1,patch_size[0]):
        for j in range(0,np.shape(onedata_test)[1]-patch_size[1]+1,patch_size[1]):
            for k in range(0,np.shape(onedata_test)[2]-patch_size[2]+1,patch_size[2]):
                temp_noise = onedata_test_noise[i:i+patch_size[0],j:j+patch_size[1],k:k+patch_size[2]]
                onedata_test_extract.append(temp_noise)
    if np.shape(onedata_test)[0] % patch_size[0] != 0:
        for j in range(0,np.shape(onedata_test)[1]-patch_size[1]+1,patch_size[1]):
            for k in range(0,np.shape(onedata_test)[2]-patch_size[2]+1,patch_size[2]):
                temp_noise = onedata_test_noise[np.shape(onedata_test)[0]-patch_size[0]:,j:j+patch_size[1],k:k+patch_size[2]]
                onedata_test_extract.append(temp_noise)
    if np.shape(onedata_test)[1] % patch_size[1] != 0:
        for i in range(0, np.shape(onedata_test)[0] - patch_size[0] + 1, patch_size[0]):
            for k in range(0,np.shape(onedata_test)[2]-patch_size[2]+1,patch_size[2]):
                temp_noise = onedata_test_noise[i:i+patch_size[0],np.shape(onedata_test)[1]-patch_size[1]:,k:k+patch_size[2]]
                onedata_test_extract.append(temp_noise)
    if np.shape(onedata_test)[0] % patch_size[0] != 0 and np.shape(onedata_test)[1] % patch_size[1] != 0:
        for k in range(0, np.shape(onedata_test)[2] - patch_size[2] + 1, patch_size[2]):
            temp_noise = onedata_test_noise[np.shape(onedata_test)[0]-patch_size[0]:, np.shape(onedata_test)[1]-patch_size[1]::,
                         k:k + patch_size[2]]
            onedata_test_extract.append(temp_noise)

    sess = tf.Session()
    ckpt = tf.train.get_checkpoint_state('./model_save/')
    print ckpt
    tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)
    onedata_test_extract = np.expand_dims(onedata_test_extract,axis=4)
    denoise = np.zeros(np.shape(onedata_test_extract))
    for i in range(np.shape(onedata_test)[0] // batch_size):
        denoise[i*batch_size:(i+1)*batch_size,:,:,:,:] = \
            sess.run(output,feed_dict={input:onedata_test_extract[i*batch_size:(i+1)*batch_size,:,:,:,:]})
    if np.shape(onedata_test)[0] % batch_size != 0:
        denoise[np.shape(onedata_test)[0] // batch_size * batch_size:, :, :, :, :] = \
            sess.run(output, feed_dict\
            ={input: onedata_test_extract[np.shape(onedata_test)[0] // batch_size * batch_size:, :, :, :, :]})

    count = 0
    denoise_onedata = np.zeros(np.shape(onedata_test))
    for i in range(0,np.shape(onedata_test)[0]-patch_size[0]+1,patch_size[0]):
        for j in range(0,np.shape(onedata_test)[1]-patch_size[1]+1,patch_size[1]):
            denoise_onedata[i:i+patch_size[0],j:j+patch_size[1],k:k+patch_size[2]]\
                = denoise[count,:,:,:,0]
            count = count + 1
    if np.shape(onedata_test)[0] % patch_size[0] != 0:
        for j in range(0,np.shape(onedata_test)[1]-patch_size[1]+1,patch_size[1]):
            for k in range(0,np.shape(onedata_test)[2]-patch_size[2]+1,patch_size[2]):
                ind = np.shape(onedata_test)[0] - np.shape(onedata_test)[0] // patch_size[0] * patch_size[0]
                denoise_onedata[-ind:,j:j+patch_size[1],k:k+patch_size[2]]\
                    = denoise[count,-ind:,:,:,0]
                count = count + 1
    if np.shape(onedata_test)[1] % patch_size[1] != 0:
        for i in range(0, np.shape(onedata_test)[0] - patch_size[0] + 1, patch_size[0]):
            for k in range(0,np.shape(onedata_test)[2]-patch_size[2]+1,patch_size[2]):
                ind = np.shape(onedata_test)[1] - np.shape(onedata_test)[1] // patch_size[1] * patch_size[1]
                denoise_onedata[i:i+patch_size[0],-ind:,k:k+patch_size[2]] \
                    =denoise[count,:,-ind:,:,0]
                count = count + 1
    if np.shape(onedata_test)[0] % patch_size[0] != 0 and np.shape(onedata_test)[1] % patch_size[1] != 0:
        for k in range(0, np.shape(onedata_test)[2] - patch_size[2] + 1, patch_size[2]):
            ind1 = np.shape(onedata_test)[0] - np.shape(onedata_test)[0] // patch_size[0] * patch_size[0]
            ind2 = np.shape(onedata_test)[1] - np.shape(onedata_test)[1] // patch_size[1] * patch_size[1]
            denoise_onedata[-ind1:,-ind2:,:] = \
                denoise[count, -ind1:,-ind2:,:,0]
            count = count + 1
    plt.figure()
    plt.imshow(onedata_test[:,:,0])
    plt.title('label')
    plt.figure()
    plt.imshow(denoise_onedata[:,:,0])
    plt.title('denoised')
    plt.figure()
    plt.imshow(onedata_test_noise[:,:,0])
    plt.title('noisedata')

    for i in range(np.shape(onedata_test)[2]):
        scipy.misc.imsave('./test_result' + '/%dlabel.png'%i, onedata_test[:,:,i])
        scipy.misc.imsave('./test_result' + '/%ddenoised.png'%i, denoise_onedata[:,:,i])
        scipy.misc.imsave('./test_result' + '/%dnoisedata.png'%i, onedata_test_noise[:, :, i])

    print 'ok'
elif mode == 'show_kernel':
    CNNclass.build_model2(input, target, True,bn_select)
    sess = tf.Session()
    ckpt = tf.train.get_checkpoint_state('./model_save/')
    tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)
    g = tf.get_default_graph()

    kernelshow(g, n_kernel, sess,1,bn_select)

    print 'ok'






