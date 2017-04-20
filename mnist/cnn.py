#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:48:06 2017

@author: Zhao Yue

Project : https://github.com/tigerzhaoyue/tensorflow-examples
"""

import tensorflow as tf


#DEFINE:
LEARNING_RATE = 0.001
TRAINING_EPOCHS = 200000 #more epochs, more acc
BATCH_SIZE = 128        #better mod 2 so it would fit GPU in the future
DISPLAY_MARGIN = 10 #display every step
INPUT_SIZE = 28 * 28 #mnist image size
CLASSES = 10
DROPOUT = 0.75 #you can try other value 


def load_mnist(data_path):
    #Load MNIST data
    from tensorflow.examples.tutorials.mnist import input_data
    if  data_path:
    	mnist = input_data.read_data_sets(data_path, one_hot=True)
    else:
    	print ("Please edit the data path!")
    
    return mnist

def cnn(mnist):
    #note that we have 3 input placeholder, one for droupout para
    x = tf.placeholder(tf.float32, [None, INPUT_SIZE])
    y = tf.placeholder(tf.float32, [None, CLASSES])
    dp = tf.placeholder(tf.float32) #dropout 
    
                       
                       
    def graph(x,weights,biases,dropout):
        #make input a 28*28 image
        flow = tf.reshape(x, shape=[-1, 28, 28, 1]) #-1 here means None for unknown batch size
        print("Graph Overview:")
        print("\nReshape layer:")
        print(flow) 
        
        #first layer  --conv layer     
        flow = tf.nn.conv2d(flow,weights['wc1'],strides = [1,1,1,1],padding = 'SAME')
        flow = tf.nn.bias_add(flow,biases['bc1'])
        flow = tf.nn.relu(flow)
        print("\nFirst layer:")
        print(flow) 
        
        #second layer -- max pooling
        flow = tf.nn.max_pool(flow,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'SAME')
        print("\nSecond layer:")
        print(flow) 
        
        #third layer  --conv layer
        flow = tf.nn.conv2d(flow,weights['wc2'],strides = [1,1,1,1],padding = 'SAME')
        flow = tf.nn.bias_add(flow,biases['bc2'])
        flow = tf.nn.relu(flow)
        print("\nThird layer:")
        print(flow) 
        
        #forth layer --max pooling
        flow = tf.nn.max_pool(flow,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'SAME')
        print("\nForth layer:")
        print(flow) 
        
        #fifth layer --fully connected layer
        #first we need to reshape flow to a colunm
        flow = tf.reshape(flow, [-1, weights['wd1'].get_shape().as_list()[0]])
        flow = tf.add(tf.matmul(flow, weights['wd1']), biases['bd1'])
        print("\nFifth layer:")
        print(flow) 
        
        #sixth layer --dropout
        flow = tf.nn.dropout(flow,dropout)
        print("\nSixth layer:")
        print(flow) 
        
        #seventh layer -- output layer
        flow = tf.add(tf.matmul(flow, weights['out']), biases['out'])
        print("\nSeventh layer:")
        print(flow) 
        return flow
        
     
        
    
    
    # Use python:dic to store our paras ,more clear~
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([1024, CLASSES]))
    }
    
    biases = {
        'bc1': tf.Variable(tf.random_normal([32])),
        'bc2': tf.Variable(tf.random_normal([64])),
        'bd1': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([CLASSES]))
    }
    
    
    
    
    
    #our model:
    pred = graph(x,weights,biases,dp)
    softmax_pred = tf.nn.softmax(pred)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred,labels = y))
    optimizer = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE).minimize(loss)
    
    #test tensors:
    corret_pred = tf.equal(tf.argmax(softmax_pred,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(corret_pred,tf.float32))
    
    #initializer
    init = tf.global_variables_initializer()
    
    
    #train and pred
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        while step * BATCH_SIZE < TRAINING_EPOCHS:
            batch_x,batch_y = mnist.train.next_batch(BATCH_SIZE)
            sess.run(optimizer,feed_dict = {x:batch_x,y:batch_y,dp:DROPOUT})
            
            #validation and display
            if step%DISPLAY_MARGIN ==0:
                #when validation  REMEBER turn off dropout!
                los,acc = sess.run([loss,accuracy],feed_dict = {x:batch_x,y:batch_y,dp:1.})
                print("Iter " + str(step*BATCH_SIZE) + ", Loss of ths Batch= " + \
                  "{:.6f}".format(los) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
            step = step + 1
            
        print("Training Finished!")
        
        
        
        
        #test section
        print("Testing Accuracy:",sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                      y: mnist.test.labels[:256],
                                      dp: 1.}))

     
        
        
if __name__ == "__main__":
    #first, you need to edit data_path here.
    #If you already download MNIST data_set, set the data_path to your 4 tars.
    #If not set it whatever, tensorflow will auto download it (Just wait)
    data_path='D:/tmp/MNIST_data'
    cnn(load_mnist(data_path))