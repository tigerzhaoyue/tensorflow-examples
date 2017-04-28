# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 13:45:47 2017
@author: Zhao Yue
using rnn to mnist dataset 
acc: >=96
"""
from __future__ import print_function
import tensorflow as tf

#basic paras
LEARNING_RATE = 0.001
TRAINING_EPOCHS = 10000
BATCH_SIZE = 128
DISPLAY_MARGIN = 10
CLASSES = 10

#rnn paras
INPUT_SIZE = 28
INPUT_STEP = 28
N_HIDDEN = 28


def load_mnist(data_path):
    # Load MNIST data
    from tensorflow.examples.tutorials.mnist import input_data
    if data_path:
        mnist = input_data.read_data_sets(data_path, one_hot=True)
    else:
        print ("Please edit the data path!")

    return mnist


def rnn(mnist):
    with tf.name_scope('Input'):
        x = tf.placeholder("float", [None, INPUT_STEP, INPUT_SIZE])
        y = tf.placeholder("float", [None, CLASSES])
    with tf.name_scope('Paras'):
        weights = {
            'out': tf.Variable(tf.random_normal([N_HIDDEN, CLASSES]),name='weights_out')
        }
        biases = {
            'out': tf.Variable(tf.random_normal([CLASSES]),name='biases_out')
        }

    def RNN_graph(x, weights, biases):
        with tf.name_scope('RNN_graph'):
            #we need to unzip our data into INPUT_STEP
            x = tf.unstack(x, INPUT_STEP, axis = 1)
            from tensorflow.contrib import rnn
            cell =rnn.BasicRNNCell(N_HIDDEN)

            #a 28 list,
            outputs,_ = rnn.static_rnn(cell, x, dtype=tf.float32,scope='RNN_graph')

            final_output = tf.matmul(outputs[INPUT_STEP-1], weights['out']) + biases['out']

            return final_output

    pred = RNN_graph(x, weights, biases)
    with tf.name_scope('loss'):
        loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=pred)
        tf.summary.scalar(name="loss", tensor=loss)
    with tf.name_scope('train'):
        global_step = tf.Variable(initial_value=0, trainable=False,name='global_step')
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE,name='optimizer')
        train_op = optimizer.minimize(loss=loss,global_step=global_step,name='train_op')

    with tf.name_scope("eval"):
        correct_pred = tf.equal(tf.argmax(pred,axis=1), tf.argmax(y, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32),name='eval_acc')

    init = tf.global_variables_initializer()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('/tmp/rnn_MNIST',tf.get_default_graph())
    with tf.Session() as sess:
        sess.run(init)
        while(global_step.eval(sess)  < TRAINING_EPOCHS ):
            batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
            batch_x = batch_x.reshape(BATCH_SIZE, INPUT_STEP, INPUT_SIZE)
            sess.run(train_op, feed_dict = {x:batch_x, y:batch_y})
            if global_step.eval(sess) % DISPLAY_MARGIN ==0:
                sumry,acc = sess.run([merged,accuracy], feed_dict={x:batch_x, y:batch_y})
                writer.add_summary(summary=sumry,global_step=global_step.eval())
                cost = sess.run(loss, feed_dict={x:batch_x, y:batch_y})
                print("Iter " + str(global_step.eval(sess)) + ", Minibatch Loss= " + \
                      "{:.6f}".format(cost) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
        print("OK!!")


        # Calculate accuracy for 128 mnist test images
        test_len = 128
        test_data = mnist.test.images[:test_len].reshape((-1, INPUT_STEP, INPUT_SIZE))
        test_label = mnist.test.labels[:test_len]
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
        print("Now, open tensorboard!")


if __name__ == "__main__":
    #first, you need to edit data_path here.
    #If you already download MNIST data_set, set the data_path to your 4 tars.
    #If not set it whatever, tensorflow will auto download it (Just wait)
    log_dir='/tmp/rnn_MNIST/'
    if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MkDir(log_dir)
    data_path = "/home/mi/tf/MNIST_data"
    rnn(load_mnist(data_path))