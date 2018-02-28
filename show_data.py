# -*- coding: utf-8 -*-
import numpy as np
import scipy
import math


def kernelshow(g,n_kernel,sess,epoch):
    kernel_all = []
    min_all = 1
    for k in range(n_kernel):
        temp = sess.run(g.get_tensor_by_name('conv' + str(k + 1) + '/kernel:0'))
        min = np.min(temp)
        if min < min_all:
            min_all = min
        kernel_all.append(temp)

    for k in range(n_kernel):
        temp = sess.run(g.get_tensor_by_name('conv' + str(k + 1) + '/kernel:0'))
        temp_size = np.shape(temp)[0] * np.shape(temp)[1] * np.shape(temp)[2] * np.shape(temp)[3] * \
                    np.shape(temp)[4]
        temp4 = []
        for i in range(np.shape(temp)[4]):
            temp_temp3 = []
            for j in range(np.shape(temp)[3]):
                temp_temp1 = np.concatenate((temp[:, :, 0, j, i], np.ones([np.shape(temp)[0], 3]) * min,
                                             temp[:, :, 1, j, i], np.ones([np.shape(temp)[0], 3]) * min)
                                            , axis=1)
                temp_temp2 = np.concatenate((temp[:, :, 2, j, i], min * np.ones([np.shape(temp)[0], 3]),
                                             temp[:, :, 3, j, i], np.ones([np.shape(temp)[0], 3]) * min)
                                            , axis=1)
                temp_12 = np.concatenate((temp_temp1, np.ones([3, np.shape(temp_temp1)[1]]) * min, temp_temp2,
                                          np.ones([3, np.shape(temp_temp1)[1]]) * min)
                                         , axis=0)  # 4*4 -> 8*8
                temp_temp3.append(temp_12)
            if np.shape(temp)[3] == 16:
                temp_3_1 = np.concatenate((temp_temp3[0], temp_temp3[1], temp_temp3[2], temp_temp3[3]), axis=1)
                temp_3_2 = np.concatenate((temp_temp3[4], temp_temp3[5], temp_temp3[6], temp_temp3[7]), axis=1)
                temp_3_3 = np.concatenate((temp_temp3[8], temp_temp3[9], temp_temp3[10], temp_temp3[11]),
                                          axis=1)
                temp_3_4 = np.concatenate((temp_temp3[12], temp_temp3[13], temp_temp3[14], temp_temp3[15]),
                                          axis=1)
                temp_3 = np.concatenate((temp_3_1, temp_3_2, temp_3_3, temp_3_4), axis=0)
                temp4.append(temp_3)  # 8*8 -> (4*8) * (4*8)
            elif np.shape(temp)[3] == 1:
                temp4.append(temp_temp3)
        if np.shape(temp)[3] == 1:
            temp_4_1 = np.concatenate((temp4[0][0], temp4[1][0], temp4[2][0], temp4[3][0]), axis=1)
            temp_4_2 = np.concatenate((temp4[4][0], temp4[5][0], temp4[6][0], temp4[7][0]), axis=1)
            temp_4_3 = np.concatenate((temp4[8][0], temp4[9][0], temp4[10][0], temp4[11][0]), axis=1)
            temp_4_4 = np.concatenate((temp4[12][0], temp4[13][0], temp4[14][0], temp4[15][0]), axis=1)
            temp_4 = np.concatenate((temp_4_1, temp_4_2, temp_4_3, temp_4_4),
                                    axis=0)  # (4*8)*(4*8) -> (4*4*8) * (4*4*8)
        elif np.shape(temp)[4] == 1:
            continue
        elif np.shape(temp)[3] == 16:
            temp_4_1 = np.concatenate((temp4[0], temp4[1], temp4[2], temp4[3]), axis=1)
            temp_4_2 = np.concatenate((temp4[4], temp4[5], temp4[6], temp4[7]), axis=1)
            temp_4_3 = np.concatenate((temp4[8], temp4[9], temp4[10], temp4[11]), axis=1)
            temp_4_4 = np.concatenate((temp4[12], temp4[13], temp4[14], temp4[15]), axis=1)
            temp_4 = np.concatenate((temp_4_1, temp_4_2, temp_4_3, temp_4_4),
                                    axis=0)  # (4*8)*(4*8) -> (4*4*8) * (4*4*8)

        scale = 5
        kernel_show = np.zeros([int(math.sqrt(temp_size)) * scale, int(math.sqrt(temp_size)) * scale])
        for ii in range(np.shape(temp_4)[0]):
            for jj in range(np.shape(temp_4)[1]):
                kernel_show[ii * scale:(ii + 1) * scale, jj * scale:(jj + 1) * scale] = temp_4[ii, jj]

        scipy.misc.imsave('./kernel_save' + '/%dkernel_%diter.png' % (k, epoch), kernel_show)