#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:17:50 2017

@author: Zhao Yue

Project : https://github.com/tigerzhaoyue/tensorflow-examples
"""

from __future__ import print_function
import tensorflow as tf


#DEFINE:
LEARNING_RATE = 0.001
TRAINING_EPOCHS = 100 #more epochs, more acc
BATCH_SIZE = 100
DISPLAY_MARGIN = 1 #display every step
HIDDEN_1 = 256  #first layer units
HIDDEN_2 = 256  #second layer units
INPUT_SIZE = 28 * 28 #mnist image size
CLASSES = 10


def load_mnist(data_path):
    #Load MNIST data
    from tensorflow.examples.tutorials.mnist import input_data
    if  data_path:
    	mnist = input_data.read_data_sets(data_path, one_hot=True)
    else:
    	print ("Please edit the data path!")
    
    return mnist

def mult_layer_perception(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_SIZE])
    y = tf.placeholder(tf.float32, [None, CLASSES])

    # Use python:dic to store our paras ,more clear~
    weights = {
        'h1': tf.Variable(tf.random_normal([INPUT_SIZE, HIDDEN_1])),
        'h2': tf.Variable(tf.random_normal([HIDDEN_1, HIDDEN_2])),
        'out': tf.Variable(tf.random_normal([HIDDEN_2, CLASSES]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([HIDDEN_1])),
        'b2': tf.Variable(tf.random_normal([HIDDEN_2])),
        'out': tf.Variable(tf.random_normal([CLASSES]))
    }
    
    def graph():
        # RELU/sigmoid activate
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.sigmoid(layer_1)
        #layer_1 = tf.nn.relu(layer_1)
        
        
        # RELU/sigmoid activate
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.sigmoid(layer_2)
        # Output layer with linear activation
        
        
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        return out_layer
    
    
    pred =  graph()
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    #AdamOptimizer here is better, or you can try
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
    #optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
    init = tf.global_variables_initializer()
    
    
    with tf.Session() as sess:
        sess.run(init)
    
        # Training cycle
        for epoch in range(TRAINING_EPOCHS):
            mean_loss = 0.
            total_batch = int(mnist.train.num_examples/BATCH_SIZE)
            # Loop over all batches
            for batch in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, l = sess.run([optimizer, loss], feed_dict={x: batch_x,
                                                              y: batch_y})
                # Compute average loss
                mean_loss =mean_loss + l / total_batch
            # Display logs per epoch step
            if epoch % DISPLAY_MARGIN == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(mean_loss))
        print("Training Finished!")
    
        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        
        
        
        
        
if __name__ == "__main__":
    #first, you need to edit data_path here.
    #If you already download MNIST data_set, set the data_path to your 4 tars.
    #If not set it whatever, tensorflow will auto download it (Just wait)
    data_path='/tmp/MNIST_data'
    mult_layer_perception(load_mnist(data_path))
