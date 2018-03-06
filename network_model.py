# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
import math
BN_DECAY = 0.9997
BN_EPSILON = 0.001
UPDATE_OPS_COLLECTION = 'update_ops'

class unet_3d_model(object):
    def __init__(self,
                 batch_size=10,
                 input_size=[10,10,10],
                 kernel_size=4,
                 in_channel=1,
                 num_filter = 16,
                 stride = [1,1,1],
                 epochs = 2):
        self.batch_size = batch_size
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.num_filter = num_filter
        self.stride = stride
        self.epochs = epochs

    def build_model(self,input, target, is_training,bn_select):
        with tf.variable_scope('net', reuse=False) as vs:
            conv = self.conv3d(input,self.kernel_size,self.in_channel,self.num_filter,'conv1')
            relu = tf.nn.relu(conv)

            conv = self.conv3d(relu,self.kernel_size,self.num_filter,self.num_filter,'conv2')
            if bn_select == 1:
                bn = self.batchnorm(conv,'bn2')
            elif bn_select == 2:
                bn = self.bn(conv,is_training,'bn2')
            else:
                bn = conv
            relu = tf.nn.relu(bn)

            conv = self.conv3d(relu, self.kernel_size, self.num_filter, self.num_filter, 'conv3')
            if bn_select == 1:
                bn = self.batchnorm(conv, 'bn3')
            elif bn_select == 2:
                bn = self.bn(conv, is_training, 'bn3')
            else:
                bn = conv
            relu = tf.nn.relu(bn)

            conv = self.conv3d(relu, self.kernel_size, self.num_filter, self.num_filter, 'conv4')
            if bn_select == 1:
                bn = self.batchnorm(conv, 'bn4')
            elif bn_select == 2:
                bn = self.bn(conv, is_training, 'bn4')
            else:
                bn = conv
            relu = tf.nn.relu(bn)

            conv = self.conv3d(relu, self.kernel_size, self.num_filter, self.num_filter, 'conv5')
            if bn_select == 1:
                bn = self.batchnorm(conv, 'bn5')
            elif bn_select == 2:
                bn = self.bn(conv, is_training, 'bn5')
            else:
                bn = conv
            relu = tf.nn.relu(bn)

            conv = self.conv3d(relu, self.kernel_size, self.num_filter, self.num_filter, 'conv6')
            if bn_select == 1:
                bn = self.batchnorm(conv, 'bn6')
            elif bn_select == 2:
                bn = self.bn(conv, is_training, 'bn6')
            else:
                bn = conv
            relu = tf.nn.relu(bn)

            conv = self.conv3d(relu, self.kernel_size, self.num_filter, self.num_filter, 'conv7')
            if bn_select == 1:
                bn = self.batchnorm(conv, 'bn7')
            elif bn_select == 2:
                bn = self.bn(conv, is_training, 'bn7')
            else:
                bn = conv
            relu = tf.nn.relu(bn)

            conv = self.conv3d(relu, self.kernel_size, self.num_filter, self.num_filter, 'conv8')
            if bn_select == 1:
                bn = self.batchnorm(conv, 'bn8')
            elif bn_select == 2:
                bn = self.bn(conv, is_training, 'bn8')
            else:
                bn = conv
            relu = tf.nn.relu(bn)

            conv = self.conv3d(relu, self.kernel_size, self.num_filter, self.num_filter, 'conv9')
            if bn_select == 1:
                bn = self.batchnorm(conv, 'bn9')
            elif bn_select == 2:
                bn = self.bn(conv, is_training, 'bn9')
            else:
                bn = conv
            relu = tf.nn.relu(bn)

            conv = self.conv3d(relu, self.kernel_size, self.num_filter, self.num_filter, 'conv10')
            if bn_select == 1:
                bn = self.batchnorm(conv, 'bn10')
            elif bn_select == 2:
                bn = self.bn(conv, is_training, 'bn10')
            else:
                bn = conv
            relu = tf.nn.relu(bn)


            conv = self.conv3d(relu, self.kernel_size, self.num_filter, self.num_filter, 'conv11')
            if bn_select == 1:
                bn = self.batchnorm(conv, 'bn11')
            elif bn_select == 2:
                bn = self.bn(conv, is_training, 'bn11')
            else:
                bn = conv
            relu = tf.nn.relu(bn)

            output_noise = self.conv3d(relu,self.kernel_size,self.num_filter,self.in_channel,'conv12')
            output = input - output_noise
            L1_loss_forward = tf.reduce_mean(tf.abs(output - target))
            pixel_num = self.input_size[0] * self.input_size[1]
            #output_flatten = tf.reduce_sum(output,axis=3)
            #tvDiff_loss_forward = \
            #    tf.reduce_mean(tf.image.total_variation(output_flatten)) / pixel_num * 200 / 10000
            r = 2000000
            for i in range(self.input_size[2]):
                if i == 0:
                    tvDiff_loss_forward = \
                    tf.reduce_mean(tf.image.total_variation(output[:, :, :, i, :])) / pixel_num * r / 10000
                else:
                    tvDiff_loss_forward = tvDiff_loss_forward + \
                                          tf.reduce_mean(tf.image.total_variation(output[:,:,:,i,:])) / pixel_num * r / 10000
            for i in range(self.input_size[1]):
                if i == 0:
                    tvDiff_loss_forward = \
                    tf.reduce_mean(tf.image.total_variation(output[:, :, i, :, :])) / pixel_num * r / 10000
                else:
                    tvDiff_loss_forward = tvDiff_loss_forward + \
                                          tf.reduce_mean(tf.image.total_variation(output[:,:,i,:,:])) / pixel_num * r / 10000
            for i in range(self.input_size[0]):
                if i == 0:
                    tvDiff_loss_forward = \
                    tf.reduce_mean(tf.image.total_variation(output[:, i, :, :, :])) / pixel_num * r / 10000
                else:
                    tvDiff_loss_forward = tvDiff_loss_forward + \
                                          tf.reduce_mean(tf.image.total_variation(output[:,i,:,:,:])) / pixel_num * r / 10000

            tvDiff_loss_forward = tvDiff_loss_forward / self.input_size[2] / self.input_size[1] / self.input_size[0]
            loss = L1_loss_forward + tvDiff_loss_forward
            del_snr, snr = self.snr(input, output, target)
            input_snr = self.input_snr(input,target)
            with tf.name_scope('summaries'):
                tf.summary.scalar('all loss', loss)
                tf.summary.scalar('L1_loss',L1_loss_forward)
                tf.summary.scalar('tv_loss',tvDiff_loss_forward)
                tf.summary.scalar('snr',snr)
            return output, loss, L1_loss_forward, tvDiff_loss_forward, snr, del_snr, output_noise,input_snr

    def build_model2(self,input, target, is_training,bn_select,prelu):
        with tf.variable_scope('net', reuse=False) as vs:
            conv = self.conv3d(input,self.kernel_size,self.in_channel,self.num_filter,'conv1')
            if prelu == True:
                relu = self.prelu(conv,'relu1')
            else:
                relu = tf.nn.relu(conv)

            conv = self.conv3d(relu, self.kernel_size, self.num_filter, self.num_filter, 'conv2')
            if bn_select == 1:
                bn = self.batchnorm(conv, 'bn2')
            elif bn_select == 2:
                bn = self.bn(conv, is_training, 'bn2')
            else:
                bn = conv
            if prelu == True:
                relu = self.prelu(bn,'relu2')
            else:
                relu = tf.nn.relu(bn)

            conv = self.conv3d(relu, self.kernel_size, self.num_filter, self.num_filter, 'conv3')
            if bn_select == 1:
                bn = self.batchnorm(conv, 'bn3')
            elif bn_select == 2:
                bn = self.bn(conv, is_training, 'bn3')
            else:
                bn = conv
            if prelu == True:
                relu = self.prelu(bn, 'relu3')
            else:
                relu = tf.nn.relu(bn)

            conv = self.conv3d(relu, self.kernel_size, self.num_filter, self.num_filter, 'conv4')
            if bn_select == 1:
                bn = self.batchnorm(conv, 'bn4')
            elif bn_select == 2:
                bn = self.bn(conv, is_training, 'bn4')
            else:
                bn = conv
            if prelu == True:
                relu = self.prelu(bn, 'relu4')
            else:
                relu = tf.nn.relu(bn)

            conv = self.conv3d(relu, self.kernel_size, self.num_filter, self.num_filter, 'conv5')
            if bn_select == 1:
                bn = self.batchnorm(conv, 'bn5')
            elif bn_select == 2:
                bn = self.bn(conv, is_training, 'bn5')
            else:
                bn = conv
            if prelu == True:
                relu = self.prelu(bn, 'relu5')
            else:
                relu = tf.nn.relu(bn)

            output_noise = self.conv3d(relu,self.kernel_size,self.num_filter,self.in_channel,'conv6')
            output = input - output_noise

            L1_loss_forward = tf.reduce_sum(tf.abs(output - target))
            L2_loss_forward = tf.reduce_sum(tf.square(output - target))
            pixel_num = self.input_size[0] * self.input_size[1]
            #output_flatten = tf.reduce_sum(output,axis=3)
            #tvDiff_loss_forward = \
            #    tf.reduce_mean(tf.image.total_variation(output_flatten)) / pixel_num * 200 / 10000

            tv_lambda = 20000
            for i in range(self.input_size[2]):
                if i == 0:
                    tvDiff_loss_forward = \
                    tf.reduce_mean(tf.image.total_variation(output[:, :, :, i, :])) / pixel_num * tv_lambda / 10000
                else:
                    tvDiff_loss_forward = tvDiff_loss_forward + \
                                          tf.reduce_mean(tf.image.total_variation(output[:,:,:,i,:])) / pixel_num * 200 / 10000
            for i in range(self.input_size[1]):
                if i == 0:
                    tvDiff_loss_forward = \
                    tf.reduce_mean(tf.image.total_variation(output[:, :, i, :, :])) / pixel_num * tv_lambda / 10000
                else:
                    tvDiff_loss_forward = tvDiff_loss_forward + \
                                          tf.reduce_mean(tf.image.total_variation(output[:,:,i,:,:])) / pixel_num * 200 / 10000
            for i in range(self.input_size[0]):
                if i == 0:
                    tvDiff_loss_forward = \
                    tf.reduce_mean(tf.image.total_variation(output[:, i, :, :, :])) / pixel_num * tv_lambda / 10000
                else:
                    tvDiff_loss_forward = tvDiff_loss_forward + \
                                          tf.reduce_mean(tf.image.total_variation(output[:,i,:,:,:])) / pixel_num * tv_lambda / 10000

            tvDiff_loss_forward = tvDiff_loss_forward * pixel_num # / self.input_size[2] / self.input_size[1] / self.input_size[0]
            loss = L1_loss_forward + tvDiff_loss_forward
            loss2 =  L2_loss_forward + tvDiff_loss_forward
            del_snr, snr = self.snr(input,output,target)
            with tf.name_scope('summaries'):
                tf.summary.scalar('all loss', loss)
                tf.summary.scalar('L1_loss',L1_loss_forward)
                tf.summary.scalar('tv_loss',tvDiff_loss_forward)
                tf.summary.scalar('snr',snr)
            return output,loss,L1_loss_forward,tvDiff_loss_forward,snr,del_snr,output_noise

    def batchnorm(self,input, name):
        with tf.variable_scope(name):
            input = tf.identity(input)
            channels = input.get_shape()[-1:]
            offset = tf.get_variable("gamma", [channels[0]], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=True)
            scale = tf.get_variable("beta", [channels[0]], dtype=tf.float32,
                                    initializer=tf.random_normal_initializer(1, 0.02),trainable=True)
            mean, variance = tf.nn.moments(input, axes=[0, 1, 2, 3], keep_dims=False)
            variance_epsilon = 1e-5
            normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale,
                                                 variance_epsilon=variance_epsilon)
        return normalized

    def bn(self,x,is_training,name):
        with tf.variable_scope(name):
            x_shape = x.get_shape()
            params_shape = x_shape[-1:]

            axis = list(range(len(x_shape) - 1))

            beta = tf.get_variable('beta', params_shape, dtype=tf.float32, initializer=tf.zeros_initializer())
            gamma = tf.get_variable('gamma', params_shape, dtype=tf.float32, initializer=tf.ones_initializer())

            moving_mean = tf.get_variable('moving_mean', params_shape, initializer=tf.zeros_initializer(), trainable=False)
            moving_variance = tf.get_variable('moving_variance', params_shape, initializer=tf.ones_initializer(), trainable=False)

            # these op will only be performed when traing
            mean, variance = tf.nn.moments(x, axis)
            update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, BN_DECAY)
            update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, BN_DECAY)
            tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
            tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

            is_training2 = tf.convert_to_tensor(is_training,dtype='bool',name='is_training')

            mean, variance = control_flow_ops.cond(is_training2, lambda :(mean,variance), lambda :(moving_mean,moving_variance))

            x = tf.nn.batch_normalization(x,mean,variance,beta,gamma,BN_EPSILON)
        self.variable_summaries(beta)
        self.variable_summaries(gamma)

        return x


    def snr(self,x,y,y_true):
        tmp_snr = tf.reduce_sum(tf.square(tf.abs(y_true))) / tf.reduce_sum(tf.square(tf.abs(y_true - y)))
        out = 10.0 * tf.log(tmp_snr) / tf.log(10.0)             # 输出图片的snr

        tmp_snr0 = tf.reduce_sum(tf.square(tf.abs(y_true))) / tf.reduce_sum(tf.square(tf.abs(y_true - x)))
        out0 = 10.0 * tf.log(tmp_snr0) / tf.log(10.0)           # 输入图片的snr

        del_snr = out - out0
        return del_snr, out

    def input_snr(self,input,target):
        tmp_snr0 = tf.reduce_sum(tf.square(tf.abs(target))) / tf.reduce_sum(tf.square(tf.abs(target - input)))
        out0 = 10.0 * tf.log(tmp_snr0) / tf.log(10.0)  # 输入图片的snr
        return out0

    def conv3d(self,x,k,in_channel,out_channel,name):
        with tf.variable_scope(name):
            kernel = tf.get_variable('kernel', [k,k,k,in_channel,out_channel],
                                     dtype=tf.float32, initializer=tf.random_normal_initializer(0.0,0.05),
                                     trainable=True)
            bias = tf.get_variable('biases',[1,out_channel],initializer=tf.constant_initializer(0.0,dtype=tf.float32))
            #kernel = tf.get_variable('kernel', shape=None,
            #                         dtype=tf.float32, initializer=tf.ones([k, k, k, in_channel, out_channel]) * 0.005,
            #                         trainable=True)
            self.variable_summaries(kernel)
            conv = tf.add(tf.nn.conv3d(x,kernel,strides=[1,1,1,1,1],padding="SAME"), bias)
        return conv

    def maxpool3d(self,x):
        return tf.nn.max_pool3d(x,ksize=[1,2,2,2,1],strides=[1,2,2,2,1],padding="SAME")

    def prelu(self,_x,name):
        with tf.variable_scope(name):
            alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                                     initializer=tf.constant_initializer(0.0),
                                     dtype=tf.float32)
            pos = tf.nn.relu(_x)
            neg = alphas * (_x - abs(_x)) * 0.5

        return pos + neg

    def _get_conv_variable(self,input_size):
        out = tf.Variable(tf.truncated_normal(input_size,stddev=0.01,name="weights"))
        return out

    def _get_bias_variable(self,input_size):
        out = tf.Variable(tf.zeros(input_size),name="biases")
        return out

    def variable_summaries(self,var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(var,mean))))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)
