'''定义接口供前端界面调用
'''

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request

from mnist import model

# 声明一个输入
x = tf.placeholder("float", [None, 784])
# 定义一个Session
sess = tf.Session()

'''获取一个线性回归模型
'''
with tf.variable_scope("regression"):
    y1, variables = model.regression(x)
saver = tf.train.Saver(variables)
# ckpt是一个模型文件
saver.restore(sess, "mnist/data/regression.ckpt")

'''获取一个卷积模型
'''
with tf.variable_scope("convolutional"):
    keep_prob = tf.placeholder("float")
    y2, variables = model.convolutional(x, keep_prob)
saver = tf.train.Saver("variables")
saver.restore(sess, "mnist/data/convolutional.ckpt")


