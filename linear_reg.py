#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 18:17:40 2017

@author: Zhao Yue

Project : https://github.com/tigerzhaoyue/tensorflow-examples

using the data_set form Coursera-machine-learning  Andrew Ng
"""

from __future__ import print_function

import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
import csv
rng = numpy.random


LEARNING_RATE = 0.01
TRAINING_EPOCH = 4000
DISPLAY_MARGIN = 50

csvfile=file('ex1data1.csv','rb')
reader = csv.reader(csvfile)
rows = [row for row in reader]
csvfile.close()

X = []
Y = []
for row in rows:
    X.append(float(row[0]))
    Y.append(float(row[1]))
  
train_X = numpy.asarray(X)
train_Y = numpy.asarray(Y)
n_samples = train_X.shape[0]

#tensors for input and paras
X = tf.placeholder("float")
Y = tf.placeholder("float")
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")


#our model
pred = tf.add(tf.multiply(X, W), b)

# 1/2 Mean squared error for loss
loss = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)

# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

# Initializing the variables
init = tf.global_variables_initializer()

#train
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(TRAINING_EPOCH):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Display logs per epoch step
        if (epoch+1) % DISPLAY_MARGIN == 0:
            c = sess.run(loss, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(loss, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()

    # Testing example, as requested (Issue #2)
    test_X = numpy.asarray([6.20, 20.27, 11.71, 5.30, 10.136, 13.20, 8.57, 7.07])
    test_Y = numpy.asarray([3.132, 21.757, 8.01, 1.80, 7.75, 14.69, 12.1, 5.40])

    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))

    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
    