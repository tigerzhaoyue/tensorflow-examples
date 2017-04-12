#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 15:25:42 2017

@author: Zhao Yue

Project : https://github.com/tigerzhaoyue/tensorflow-examples
"""

from __future__ import print_function

import tensorflow as tf

#DEFINE:
LEARNING_RATE = 0.01
TRAINING_EPOCHS = 100 #more epochs, more acc
BATCH_SIZE = 100
DISPLAY_MARGIN = 1 #display every step

def load_mnist(data_path):
    #Load MNIST data
    from tensorflow.examples.tutorials.mnist import input_data
    if  data_path:
    	mnist = input_data.read_data_sets(data_path, one_hot=True)
    else:
    	print ("Please edit the data path!")
    
    return mnist




def logistic_reg(mnist):
    
    #alloc space for raw data and paras
    x = tf.placeholder(tf.float32,[None, 28 * 28])
    y = tf.placeholder(tf.float32, [None, 10])
    W = tf.Variable(tf.zeros([28 * 28, 10]))   #set 0 here not random because sigmoid func would be unsure of a specific class if  x*W+b equals 0,
    b = tf.Variable(tf.zeros([10]))	     # so it's reasonable for paras = 0 at the beginning of training
    
               
    pred = tf.nn.softmax(tf.matmul(x, W) + b)
    
    loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), \
                                         axis=1)) #axis = 1 here indicates we keep dim0 unchanged and sum dimention1(10 elements)
    
    
    #Define a optimizer  using GD here    Of course and its initializer
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
    init = tf.global_variables_initializer()
    
    
    
    #Start the session:
    with tf.Session() as sess:
        sess.run(init)
        
        #train:
        for epoch in range(TRAINING_EPOCHS):
            mean_cost = 0.
            total_batch = int(mnist.train.num_examples/BATCH_SIZE)
            for batch in range(total_batch):
                batch_xs,batch_ys = mnist.train.next_batch(BATCH_SIZE)
                opt,l = sess.run([optimizer,loss], feed_dict={x: batch_xs,y: batch_ys})
                
                mean_cost = mean_cost + l / total_batch
            if epoch % DISPLAY_MARGIN == 0:
                print("Epoch:", '%04d' % epoch, "loss=", "{:.9f}".format(mean_cost))
        print("Training Finished!")
                    
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))  #a None * 1 bool vector
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  #acc = number of "True(1)" / total_num
        
        print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    
    
if __name__ == "__main__":
    #first, you need to edit data_path here.
    #If you already download MNIST data_set, set the data_path to your 4 tars.
    #If not set it whatever, tensorflow will auto download it (Just wait)
    data_path='/tmp/MNIST_data'
    logistic_reg(load_mnist(data_path))

            
             
            
    
                    


