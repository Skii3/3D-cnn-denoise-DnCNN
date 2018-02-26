import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Activation
from keras.layers.convolutional import Conv3D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint

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

    def build_model(self,input, target):

        conv1 = self.conv3d(input,self.kernel_size,self.in_channel,self.num_filter,'conv1')
        relu1 = tf.nn.relu(conv1)

        conv2 = self.conv3d(relu1,self.kernel_size,self.num_filter,self.num_filter,'conv2')
        bn2 = self.batchnorm(conv2,'bn2')
        relu2 = tf.nn.relu(bn2)

        conv3 = self.conv3d(relu2,self.kernel_size,self.num_filter,self.num_filter,'conv3')
        bn3 = self.batchnorm(conv3,'bn3')
        relu3 = tf.nn.relu(bn3)

        conv4 = self.conv3d(relu3,self.kernel_size,self.num_filter,self.num_filter,'conv4')
        bn4 = self.batchnorm(conv4,'bn4')
        relu4 = tf.nn.relu(bn4)

        conv5 = self.conv3d(relu4,self.kernel_size,self.num_filter,self.num_filter,'conv5')
        bn5 = self.batchnorm(conv5,'bn5')
        relu5 = tf.nn.relu(bn5)

        conv6 = self.conv3d(relu5,self.kernel_size,self.num_filter,self.num_filter,'conv6')
        bn6 = self.batchnorm(conv6,'bn6')
        relu6 = tf.nn.relu(bn6)

        conv7 = self.conv3d(relu6,self.kernel_size,self.num_filter,self.num_filter,'conv7')
        bn7 = self.batchnorm(conv7,'bn7')
        relu7 = tf.nn.relu(bn7)

        conv8 = self.conv3d(relu7,self.kernel_size,self.num_filter,self.num_filter,'conv8')
        bn8 = self.batchnorm(conv8,'bn8')
        relu8 = tf.nn.relu(bn8)

        conv9 = self.conv3d(relu8,self.kernel_size,self.num_filter,self.num_filter,'conv9')
        bn9 = self.batchnorm(conv9,'bn9')
        relu9 = tf.nn.relu(bn9)

        conv10 = self.conv3d(relu9,self.kernel_size,self.num_filter,self.num_filter,'conv10')
        bn10 = self.batchnorm(conv10,'bn10')
        relu10 = tf.nn.relu(bn10)

        conv11 = self.conv3d(relu10,self.kernel_size,self.num_filter,self.num_filter,'conv11')
        bn11 = self.batchnorm(conv11,'bn11')
        relu11 = tf.nn.relu(bn11)

        conv12 = self.conv3d(relu11,self.kernel_size,self.num_filter,self.num_filter,'conv12')
        bn12 = self.batchnorm(conv12,'bn12')
        relu12 = tf.nn.relu(bn12)

        output = self.conv3d(relu12,self.kernel_size,self.num_filter,self.in_channel,'conv13')

        L1_loss_forward = tf.reduce_mean(tf.abs(output - target))
        pixel_num = self.input_size[0] * self.input_size[1]
        output_flatten = tf.reduce_sum(output,axis=3)
        tvDiff_loss_forward = \
            tf.reduce_mean(tf.image.total_variation(output_flatten)) / pixel_num * 200 / 10000
        '''
        for i in range(self.input_size[2]):
            if i == 0:
                tvDiff_loss_forward = \
                tf.reduce_mean(tf.image.total_variation(output[:, :, :, i, :])) / pixel_num * 200 / 10000
            else:
                tvDiff_loss_forward = tvDiff_loss_forward + \
                                      tf.reduce_mean(tf.image.total_variation(output[:,:,:,i,:])) / pixel_num  * 200 / 10000
        '''
        loss = L1_loss_forward + tvDiff_loss_forward
        snr = self.snr(output,target)

        return output,loss,L1_loss_forward,tvDiff_loss_forward,snr

    def batchnorm(self,input, name):
        with tf.variable_scope(name):
            input = tf.identity(input)
            channels = input.get_shape()[4]
            offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=True)
            scale = tf.get_variable("scale", [channels], dtype=tf.float32,
                                    initializer=tf.random_normal_initializer(1, 0.02),trainable=True)
            mean, variance = tf.nn.moments(input, axes=[0, 1, 2, 3], keep_dims=False)
            variance_epsilon = 1e-5
            normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale,
                                                 variance_epsilon=variance_epsilon)
        return normalized

    def snr(self,y,y_true):
        tmp_snr = tf.reduce_sum(tf.square(tf.abs(y_true))) / tf.reduce_sum(tf.square(tf.abs(y_true - y)))
        out = 10.0 * tf.log(tmp_snr) / tf.log(10.0)
        return out

    def conv3d(self,x,k,in_channel,out_channel,name):
        with tf.variable_scope(name):
            kernel = tf.get_variable('kernel', [k,k,k,in_channel,out_channel],
                                     dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.02),trainable=True)
            conv = tf.nn.conv3d(x,kernel,strides=[1,1,1,1,1],padding="SAME")
        return conv

    def maxpool3d(self,x):
        return tf.nn.max_pool3d(x,ksize=[1,2,2,2,1],strides=[1,2,2,2,1],padding="SAME")

    def _get_conv_variable(self,input_size):
        out = tf.Variable(tf.truncated_normal(input_size,stddev=0.01,name="weights"))
        return out

    def _get_bias_variable(self,input_size):
        out = tf.Variable(tf.zeros(input_size),name="biases")
        return out

    def train_model(self,model,train_data,train_label,real_epochs):

        model_checkpoint = ModelCheckpoint('unet.hdf5',
                                           monitor='loss',
                                           save_best_only=True)
        train_hist = model.fit(train_data,train_label,
                               batch_size=self.batch_size,
                               epochs=self.epochs,
                               validation_split=0.1,
                               shuffle=True,
                               callbacks=[model_checkpoint],
                               verbose=1)
        if real_epochs == 1:
            print "[*] saving model weights"
            model.save_weights('model_weights.h5',overwrite=True)

        return model,train_hist

    def test_model(self,model,test_data):
        model.load_weights('model_weights.h5')
        print "[*] begin to test learned model"
        denoised = model.predict(test_data,verbose=1)
        return denoised