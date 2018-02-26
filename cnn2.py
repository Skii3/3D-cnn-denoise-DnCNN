import tensorflow as tf
import numpy as np

IMG_SIZE_PX = 50
SLICE_COUNT = 20

n_classes = 2
batch_size = 10
val_batch_size = 40
x = tf.placeholder('float') #, shape=[batch_size, IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, 1])
y = tf.placeholder('float') #,shape=[batch_size,2])

keep_rate = 0.8

def conv3d(x, W):
    '''tf.nn.conv3d(input,filter,stride,padding)
      input: [batch,in_depth,in_height,in_width,in_channels]
      filter:[filter_depth,filter_height,filter_width,in_channels,out_channels]
      stride: 1D tensor of length  5 with stride[0]=stride[4]=1
      padding:"SAME","VALID"
    '''
    return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding='SAME')  

def maxpool3d(x):
    #                        size of window         movement of window as you slide abouta
    '''tf.nn.max_pool3d(input,ksize,strides,padding)
       input:[batch,depth,rows,cols,channels]
       ksize:1D tensor with length 5 and ksize[0]=ksize[4]=1
       strides
       padding
    '''
    return tf.nn.max_pool3d(x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')

def convolutional_neural_network(x):
    #                # 5 x 5 x 5 patches, 1 channel, 32 features to compute.
    weights = {'W_conv1':tf.Variable(tf.random_normal([3,3,3,1,32])),
               #       5 x 5 x 5 patches, 32 channels, 64 features to compute.
               'W_conv2':tf.Variable(tf.random_normal([3,3,3,32,64])),
               #                                  64 features
               'W_fc':tf.Variable(tf.random_normal([54080,1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    #                            image X      image Y        image Z
    x = tf.reshape(x, shape=[-1, IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, 1])

    conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool3d(conv1)


    conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool3d(conv2)

    fc = tf.reshape(conv2,[-1, 54080])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output

def data_prepare():
    much_data = np.load('muchdata-50-50-20.npy')
# If you are working with the basic sample data, use maybe 2 instead of 100 here... you don't have enough data to really do this
    train_data = much_data[:-100] 				# 1297*2,[i,0]:20*50*50,[i,1]:1*2(one hot)a
    train_data_size = np.shape(train_data)[0]
    print "train size:"
    print np.shape(train_data)
    validation_data = much_data[-100:]
    print "validation size:"
    print np.shape(validation_data)
    count1 = 0
    count2 = 0
    for i in range(0,train_data_size,1):
        if train_data[i,1][0] == 1:
            count1 = count1 + 1
        else:
            count2 = count2 + 1
    train_data_pos = np.zeros([count1,SLICE_COUNT,IMG_SIZE_PX,IMG_SIZE_PX,1])
    train_data_pos_label = np.zeros([count1,2])
    train_data_pos_label[:,0] = 1
    train_data_neg = np.zeros([count2,SLICE_COUNT,IMG_SIZE_PX,IMG_SIZE_PX,1])
    train_data_neg_label = np.zeros([count2,2])
    train_data_neg_label[:,1] = 1
    count1 = 0
    count2 = 0
    for i in range(0,train_data_size,1):
        if train_data[i,1][0] == 1:
            train_data_pos[count1,:,:,:,:] = np.reshape(train_data[i,0][:SLICE_COUNT,:,:],[SLICE_COUNT,IMG_SIZE_PX,IMG_SIZE_PX,1])
            count1 = count1 + 1
        else:
            train_data_neg[count2] =  np.reshape(train_data[i,0][:SLICE_COUNT,:,:],[SLICE_COUNT,IMG_SIZE_PX,IMG_SIZE_PX,1])
            count2 = count2 + 1
    neg_repeat = np.random.randint(0,count2,size=(count1-count2))
    train_data_new = np.concatenate((train_data_pos,train_data_neg,train_data_neg[neg_repeat]),axis = 0)
    train_data_new_label = np.concatenate((train_data_pos_label,train_data_neg_label,train_data_neg_label[neg_repeat]),axis = 0)

    print np.shape(train_data_new)[0]
    shuffle = np.arange(np.shape(train_data_new)[0])
    np.random.shuffle(shuffle)
    train_data_new = train_data_new[shuffle,:,:,:,:]
    train_data_new_label = train_data_new_label[shuffle,:]

    print "train pos size"
    print np.shape(train_data_pos)
    print "train new size"
    print np.shape(train_data_new)

    return train_data_new,train_data_new_label,validation_data

tf.device('/gpu:0')
hm_epochs = 3000
def train_neural_network(x,train_data,train_label,val_data):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
    with tf.Session() as sess:
        #sess.run(tf.initialize_all_variables())
        sess.run(tf.global_variables_initializer())

        successful_runs = 0
        total_runs = 0

        for epoch in range(hm_epochs):
            epoch_loss = 0
            #for data in train_data:
            total_runs += 1
               # try:
            choice = np.random.permutation(np.shape(train_data)[0])[:batch_size]
            _, c, pred = sess.run([optimizer, cost, prediction], feed_dict={x: train_data[choice], y: train_label[choice]})
            epoch_loss += c
            successful_runs += 1
               # except Exception as e:
                    # I am passing for the sake of notebook space, but we are getting 1 shaping issue from one
                    # input tensor. Not sure why, will have to look into it. Guessing it's
                    # one of the depths that doesn't come to 20.
               #     print str(e)
               #     pass
                    # print(str(e))
            if epoch % 20 == 0:
              print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)
              print np.argmax(train_label[choice],1)
              print np.argmax(pred,1)
              #print train_label[choice]

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            if epoch % 100 == 0: 
                sample_choice = np.random.permutation(100)[:val_batch_size]
                temp = validation_data[sample_choice,0]

                val_X = np.zeros([val_batch_size,SLICE_COUNT,IMG_SIZE_PX,IMG_SIZE_PX,1])
                for i in range(0,val_batch_size,1):
                    if np.shape(temp[i])[0] != SLICE_COUNT:
                        val_X[i,:,:,:,:] = np.reshape(temp[i][:SLICE_COUNT,:,:],[SLICE_COUNT,IMG_SIZE_PX,IMG_SIZE_PX,1])
                    else:
                        val_X[i,:,:,:,:] = np.reshape(temp[i],[SLICE_COUNT,IMG_SIZE_PX,IMG_SIZE_PX,1])
                val_X = np.reshape(val_X,[-1, IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, 1])

                tempY = validation_data[sample_choice,1]
                val_Y = np.zeros([val_batch_size,2])
                for i in range(0,val_batch_size,1):
                  val_Y[i,:]=tempY[i]
                print('--------------------------Accuracy:{}'.format(accuracy.eval({x:val_X,y:val_Y})))

        print('Done. Finishing accuracy:')
        print('Accuracy:', accuracy.eval({x: val_X, y: val_Y}))

        print('fitment percent:', successful_runs , total_runs)

        saver_path = saver.save(sess, str("result/model_"+ str(hm_epochs) +".ckpt")) 

def test_neural_network(x,validation_data):
     prediction = convolutional_neural_network(x)
     correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
     accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
     saver = tf.train.Saver()
     with tf.Session() as sess:
         saver.restore(sess,str("result/model.ckpt"))
         temp = validation_data[:,0]

         val_X = np.zeros([np.shape(validation_data)[0],SLICE_COUNT,IMG_SIZE_PX,IMG_SIZE_PX,1])
         for i in range(0,val_batch_size,1):
             if np.shape(temp[i])[0] != SLICE_COUNT:
                 val_X[i,:,:,:,:] = np.reshape(temp[i][:SLICE_COUNT,:,:],[SLICE_COUNT,IMG_SIZE_PX,IMG_SIZE_PX,1])
             else:
                 val_X[i,:,:,:,:] = np.reshape(temp[i],[SLICE_COUNT,IMG_SIZE_PX,IMG_SIZE_PX,1])
         val_X = np.reshape(val_X,[-1, IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, 1])

         tempY = validation_data[:,1]
         val_Y = np.zeros([np.shape(validation_data)[0],2])
         for i in range(0,val_batch_size,1):
             val_Y[i,:]=tempY[i]
         print('--------------------------Accuracy:{}'.format(accuracy.eval({x:val_X,y:val_Y})))
         pred = sess.run([prediction], feed_dict={x: val_X, y: val_Y})
         print np.argmax(val_Y,1)
         print np.argmax(np.reshape(pred,[-1,2]),1)


if __name__ == '__main__':
    [train_data_new,train_data_new_label,validation_data] = data_prepare()
    #train_neural_network(x,train_data_new,train_data_new_label,validation_data)
    test_neural_network(x,validation_data)
