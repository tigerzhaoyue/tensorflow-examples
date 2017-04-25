#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 09:48:22 2017

@author: mi
"""

import tensorflow as tf
import numpy as np


log_dir = '/tmp/linear_log'
if tf.gfile.Exists(log_dir):
    tf.gfile.DeleteRecursively(log_dir)
tf.gfile.MkDir(log_dir)

with tf.name_scope("input"):
    x=tf.placeholder(tf.float32)
    y=tf.placeholder(tf.float32)


with tf.name_scope("linear_model"):
    with tf.name_scope("Weight"):
        W=tf.Variable([0.3],tf.float32)
    with tf.name_scope("bias"):
        b=tf.Variable([-0.3],tf.float32)
    linear_model = W*x+b

with tf.name_scope("loss"):
    loss = tf.reduce_sum(tf.square(linear_model-y))##sum of square
    l_s = tf.summary.scalar(name = 'loss summary', tensor= loss)



with tf.name_scope("train"):
    # global step
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss, global_step=global_step)

#training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]

#training loop
init = tf.global_variables_initializer()#initial handler
sess = tf.Session()#running session
sess.run(init)#initial

#summary
merged =tf.summary.merge_all()
train_writer = tf.summary.FileWriter(log_dir+"/train", sess.graph)
test_writer = tf.summary.FileWriter(log_dir+"/test")
for i in range(1000):
    _, summary = sess.run([train,merged],{x:x_train,y:y_train})
    train_writer.add_summary(summary, global_step.eval(sess))
train_writer.close()
#evaluate training accuracy
curr_W,curr_b,curr_loss = sess.run([W,b,loss],{x:x_train,y:y_train})
print ("W:%s b:%s loss: %s"%(curr_W,curr_b,curr_loss))



