import os
import model
import tensorflow as tf
import input_data

data = input_data.read_data_sets('MNIST_data', one_hot=True)

'''定义模型
'''
with tf.variable_scope("convolutional"):
    x = tf.placeholder(tf.float32, [None, 784], name='x')
    keep_prob = tf.placeholder(tf.float32)
    y, variables = model.convolutional(x, keep_prob)


