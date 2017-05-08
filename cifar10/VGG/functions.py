import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as layers
def conv(name, x, out_channels, kernel_size = [3,3], stride = [1,1,1,1], trainable = True):
    '''
    
    :param name: eg:conv1, pool1...
    :param x: input tensor [batch_size, height, width, channels]
    :param out_channels: output channels
    :param kernel_size: size of filter e.g.:[3,3]  [5,5]
    :param stride: step length of filter
    :param trainable: if we wangt to load fun-tune some models... False it
    :return: 4-D tensor
    '''

    in_channels = x.get_shape()[-1]
    with tf.variable_scope(name):
        w = tf.get_variable(name='weights',
                            trainable=trainable,
                            shape = [kernel_size[0], kernel_size[1], in_channels, out_channels],
                            initializer=layers.xavier_initializer())

        b = tf.get_variable(name='biases',
                            trainable=trainable,
                            shape=[out_channels],
                            initializer=tf.constant_initializer(0.0))

        x = tf.nn.conv2d(input=x,filter=w,strides=stride,padding='SAME',name='conv')
        x = tf.nn.bias_add(x,b,name='bias_add')
        x = tf.nn.relu(x, name='relu')

        return x

def pool(name, x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True):
    '''
    
    :param name: layer name
    :param x: input tensor
    :param kernel: kernelsize  usually[1,k,k,1]
    :param stride: step length
    :param is_max_pool: ifTrue maxpool else avg pool
    :return: 4-D tensor
    '''
    if is_max_pool:
        x = tf.nn.max_pool(x, kernel, strides=stride, padding='SAME', name='layer_name_max_pool')
    else:
        x = tf.nn.avg_pool(x, kernel, strides=stride, padding='SAME', name='layer_name_avg_pool')

    return x




def batch_norm(x):
    '''
    batch norm function  for quicker training
    :param x: 
    :return: 
    '''
    epsilon = 1e-3
    batch_mean, batch_var = tf.nn.moments(x,axes=[0])
    x = tf.nn.batch_normalization(x,
                                  mean=batch_mean,
                                  variance=batch_var,
                                  offset=None,
                                  scale=None,
                                  variance_epsilon=epsilon)
    return x




def FC_layer(name, x, out_nodes):
    '''
    
    :param name: layer name
    :param x: input tensor
    :param out_nodes: output num
    :return: 2-D tensor [batch_size,out_nodes]
    '''

    shape = x.get_shape()
    if len(shape) == 4:   #convolution input
        size = shape[1].value * shape[2].value * shape[3].value
    else:                 #fc input
        size = shape[-1].value

    with  tf.variable_scope(name):
        w = tf.get_variable(name='weights',
                            shape=[size,out_nodes],
                            initializer=layers.xavier_initializer())

        b = tf.get_variable(name='biases',
                            shape=[out_nodes],
                            initializer=tf.constant_initializer(0.0))

        flat_x = tf.reshape(x, [-1,size])

        x = tf.nn.bias_add(tf.matmul(flat_x, w), b)
        x = tf.nn.relu(x)

        return x


def loss(logits,labels):
    '''
    
    :param logits: [batch_size, n_classes]
    :param labels: [batchh_size, n_classes]   (one-hot labels matrix)
    :return: scalar  loss
    '''

    with tf.name_scope('loss') as scope:
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels,logits=logits)
        tf.summary.scalar(scope+'/loss', loss)
    return loss


def accuracy(logits, labels):
    '''
    
    :param logits: 
    :param labels: 
    :return: 
    '''
    with tf.name_scope('accuracy') as scope:
        correct = tf.equal(tf.arg_max(logits,1), tf.arg_max(labels,1))
        correct = tf.cast(correct,tf.float32)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope+'/accuracy', accuracy)

    return accuracy


def optimize(loss, learning_rate, global_step):
    with tf.name_scope('optimizer') as scope:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op


def load_conv(data_path, session, skip_layer):
    data_dict = np.load(data_path, encoding='latin1').item()
    for key in data_dict:
        if key not in skip_layer:
            with tf.variable_scope(key, reuse=True):
                for subkey, data in zip(('weights','biases'), data_dict[key]):
                    session.run(tf.get_variable(subkey).assign(data))


def num_correct_prediction(logits, labels):
    correct = tf.equal(tf.arg_max(logits,1), tf.arg_max(labels,1))
    correct = tf.cast(correct, tf.int32)
    n_correct = tf.reduce_sum(correct)
    return n_correct