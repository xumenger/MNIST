import os

import input_data
import model
import tensorflow as tf

data = input_data.read_data_sets('MNIST_data', one_hot=True)

'''建立模型
'''
with tf.variable_scope("regression"):
    # x要求用户输入的
    # 第一个参数：格式是tf.float32
    # 第二个参数是张量
    x = tf.placeholder(tf.float32, [None, 784])
    y, variables = model.regression(x)

'''训练模型
'''
y_ = tf.placeholder("float", [None, 10])
# 设置训练的交叉商
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
# 定义训练步骤（Gradientdescentoptimizer是优化器，设置步长为0.01）
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# 定义一个预测
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 计算准确率（把correct_prediction转换为tf.float32格式）
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

