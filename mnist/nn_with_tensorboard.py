# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 13:34:47 2017

@author: Zhao Yue
This is a full-connection network
accuracy is about 98%
only to show how to use tensorboard
"""
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
learning_rate = 0.001
max_steps = 1000
dropout_rate = 0.9
log_dir = 'D:/tmp/MNIST_log/'
#import data (4 gz)
data_path = "D:/tmp/MNIST_data/"
mnist = input_data.read_data_sets(data_path,one_hot = True)


if tf.gfile.Exists(log_dir):
    tf.gfile.DeleteRecursively(log_dir)
tf.gfile.MakeDirs(log_dir)
#start a session:
sess = tf.InteractiveSession()


def add_variable_summaries(var):
    '''
    add some summaries such as mean, std, sqrt ... to a variable
    '''
    with tf.name_scope('variable_summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean',mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        
        
#def a funtion to help us construct the complex nn layers
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act):
    '''
    include  1 conv layer (weights, bias), 1 activate layer (Relu? sigmoid? idnetity?)
    also, this func generate a name scope for a nn layer,
    so it would be clear in tensorboard
    '''
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            #truncated_normal to initialize
            weights = tf.Variable(initial_value = tf.truncated_normal(shape=[input_dim,output_dim], stddev = 0.1))
            add_variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(initial_value = tf.constant(0.1, shape = [output_dim]))
            add_variable_summaries(biases)
        with tf.name_scope('linear'):
            preact = tf.matmul(input_tensor, weights) + biases
            #or use tf.add()
            tf.summary.histogram('pre_act', preact)
        activations = act(preact, name = 'activation')
        tf.summary.histogram('activations', activations)
        return activations

def feed_dict(bool_istrain):
    if bool_istrain:
        xs, ys = mnist.train.next_batch(100)
        k = dropout_rate
    else:
        xs, ys = mnist.test.images, mnist.test.labels
        k = 1.0
    return {x: xs, y_: ys, keep_prob: k}




#Now, build our graph:
    
'''
input layer
'''
with tf.name_scope("INPUT"):
    x = tf.placeholder(tf.float32, [None,28 * 28], name = 'X-INPUT')
    y_ = tf.placeholder(tf.float32, [None, 10], name = 'Y-INPUT')
    
    with tf.name_scope('INPUT_IMAGE'):
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', image_shaped_input, 10)            

'''
hidden layer 1 (fc)
'''
hidden1 = nn_layer(x, 28 *28, 500, 'layer1', act = tf.nn.relu)

'''
drop out
'''
with tf.name_scope('dropout_handler'):
    keep_prob = tf.placeholder(tf.float32, name = 'dp_keep_probility')
    tf.summary.scalar('dp_keep_probility', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)

'''
next nn layer (fc)
'''
y = nn_layer(dropped, 500, 10, 'layer2', act = tf.identity) #with no activation here

with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(labels = y_,logits = y)
    with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(diff)
tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_pred'):
        correct_prediction = tf.equal(tf.arg_max(y, dimension = 1), tf.arg_max(y_, dimension = 1)) #bool tensor
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

'''
merge all summaries
'''
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(log_dir+"/train", sess.graph)
test_writer = tf.summary.FileWriter(log_dir+"/test")

tf.global_variables_initializer().run()







#Now, run our graph

# main loop
for i in range(max_steps):
    if i%10 == 0:
        summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(bool_istrain=False))
        test_writer.add_summary(summary, i)
        print("Step:%s, Accuracy:%s" % (i, acc))
    elif i%100 == 99:
        run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(bool_istrain=True),options=run_options,run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d'% i)
        train_writer.add_summary(summary,i)
        print("Adding run metadata for",i)
    else:
        summary, _ = sess.run([merged, train_step], feed_dict= feed_dict(bool_istrain=True))
        train_writer.add_summary(summary,i)
train_writer.close()
test_writer.close()
