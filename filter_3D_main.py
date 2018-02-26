#coding=utf-8

from utils import load_data
import numpy as np
from network_model import unet_3d_model
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import scipy
import random
#------------------ global settings ------------------#
REL_FILE_PATH = './plutdata'
TRAINDATA_SAVE_PATH = './traindata_save'
SAVEPSNR = './savepsnr'
ind1 = random.randint(0,99)
ind2 = random.randint(0,15)
start_point = [ind1,ind1,ind2]
end_point = [876,900,100]
stride = [100,100,16]
max_epochs = 1000
lr = 1e-4
batch_size = 40
mode = 'train'
if mode == 'train':
    patch_size = [40, 40, 40]
elif mode == 'test':
    patch_size = [150, 150, 150]
#------------------ global settings ------------------#

tf.device('/gpu:0')

CNNclass = unet_3d_model(batch_size=batch_size,
                                 input_size=patch_size,
                                 kernel_size=4,
                                 in_channel=1,
                                 num_filter=16,
                                 stride=[1,1,1],
                                 epochs=2)
input = tf.placeholder('float32', [None, patch_size[0], patch_size[1], patch_size[2], 1], name='input')
target = tf.placeholder('float32', [None, patch_size[0], patch_size[1], patch_size[2], 1], name='target')

if mode == 'train':

        output, loss, l1_loss, tv_loss, snr = CNNclass.build_model(input, target)
        optim_forward = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        print("---------------------------training model---------------------------")
        for epoch in range(0, max_epochs + 1):

            ind1 = random.randint(0, 99)
            ind2 = random.randint(0, 15)
            start_point = [ind1, ind1, ind2]

            data_label_epoch, data_epoch, _ = load_data(rel_file_path=REL_FILE_PATH,
                                                        start_point=start_point,
                                                        end_point=end_point,
                                                        patch_size=patch_size,
                                                        stride=stride,
                                                        traindata_save=TRAINDATA_SAVE_PATH)
            data_epoch = np.expand_dims(data_epoch, axis=4)
            data_label_epoch = np.expand_dims(data_label_epoch, axis=4)

            epoch_time = time.time()
            ind = np.arange(np.shape(data_epoch)[0])
            ind = np.random.permutation(ind)
            data_epoch = data_epoch[ind,:,:,:,:]
            data_label_epoch = data_label_epoch[ind,:,:,:,:]
            sum_all_loss, sum_tvDiff_loss, sum_L1_loss, sum_snr = 0, 0, 0, 0
            for step in range(0, np.shape(data_epoch)[0] // batch_size,1):
                data_step = data_epoch[step:step+batch_size,:,:,:,:]
                data_label_step = data_label_epoch[step:step+batch_size,:,:,:,:]
                tvDiff_loss,L1_loss,_,SNR = sess.run([tv_loss,l1_loss,optim_forward,snr],feed_dict={input:data_step, target:data_label_step})
                sum_all_loss = sum_all_loss + tvDiff_loss + L1_loss
                sum_tvDiff_loss = sum_tvDiff_loss + tvDiff_loss
                sum_L1_loss = sum_L1_loss + L1_loss
                sum_snr = sum_snr + SNR
                print("[*] Step %d: all loss: %.8f, tvDiff loss: %.8f, l1 loss: %.8f, snr: %.2f") \
                     % (step,tvDiff_loss+L1_loss, tvDiff_loss, L1_loss, SNR)
            print ("[*] Epoch [%2d/%2d] %4d time: %4.4fs, sum all loss: %.8f, sum tvDiff loss: %.8f, sum l1 loss: %.8f, sum snr: %.8f") \
                  % (epoch+1,max_epochs,np.shape(data_epoch)[0] // batch_size,
                     time.time()-epoch_time,sum_all_loss,sum_tvDiff_loss,sum_L1_loss,sum_snr)

            if (epoch+1) % 1 == 0:
                ind = np.arange(np.shape(data_epoch)[0])
                ind = np.random.permutation(ind)
                data_test = data_epoch[ind[:2],:,:,:,:]
                data_label_test = data_label_epoch[ind[:2],:,:,:,:]
                denoise_test = sess.run(output,feed_dict={input:data_test, target:data_label_test})
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
    print 'ok'
elif mode == 'show_kernel':
    print 'ok'






