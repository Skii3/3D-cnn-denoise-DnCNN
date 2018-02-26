#coding=utf-8

from utils import load_data
import numpy as np
from network_model import unet_3d_model
import matplotlib.pyplot as plt
#------------------ global settings ------------------#
REL_FILE_PATH = './traindata/plutdata'
SAVEPSNR = './savepsnr'
start_point = [80,80,80]
sample_size = [80,80,80]
stride = [2,2,2]
max_epochs = 10
batch_size = 10
random_seed = 123
noise_sigma = [0.00, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.2]   # range from 10% of max pixel intensity to 70% of max pixel intensity
mode = 'train'
if mode == 'train':
    patch_size = [10, 10, 10]
elif mode == 'test':
    patch_size = [256, 256, 256]
#------------------ global settings ------------------#

train_data,test_data = load_data(rel_file_path = REL_FILE_PATH,
                                 start_point=start_point,
                                 sample_size=sample_size,
                                 patch_size=patch_size,
                                 stride=stride)
print "[*] finish reading data"

train_data_size = np.shape(train_data)[0]           # 46656
CNNclass = unet_3d_model(batch_size=batch_size,
                                 input_size=patch_size,
                                 kernel_size=[3,3,3],
                                 in_channel=1,
                                 num_filter=16,
                                 stride=[1,1,1],
                                 epochs=2)
model = CNNclass.build_model()

def prepare_data():
    sigma_random = np.random.randint(0, np.shape(noise_sigma)[0], train_data_size)
    np.random.seed(random_seed)
    noise_data = np.zeros([train_data_size, patch_size[0], patch_size[1], patch_size[2]])
    for i in range(train_data_size):
        max_intensity = np.max(train_data[i, :, :, :])
        rand_data = np.random.normal(0,
                                     noise_sigma[sigma_random[i]] * max_intensity,
                                     patch_size)
        noise_data[i] = rand_data
    data_step = train_data + noise_data  # train data for every step
    data_step = np.reshape(data_step, [train_data_size, patch_size[0], patch_size[1], patch_size[2], 1])
    data_label_step = train_data
    data_label_step = np.reshape(data_label_step, [train_data_size, patch_size[0], patch_size[1], patch_size[2], 1])
    print "[*] finish prepare data"
    return data_step,data_label_step

if mode == 'train':
    for epoch in range(max_epochs):
        data_step,data_label_step = prepare_data()

        print "-------------------epoch:{}------------------".format(epoch+1)
        real_epoch = 0
        if (epoch+1)%2 == 0:
            real_epoch = 1

        model, hist = CNNclass.train_model(model=model,
                                          train_data=data_step,
                                          train_label=data_label_step,
                                          real_epochs=real_epoch)

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







