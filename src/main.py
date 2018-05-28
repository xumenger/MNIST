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
saver = tf.train.Saver(variables)
saver.restore(sess, "mnist/data/convolutional.ckpt")


'''根据输入进行识别（线性）
需要从前端获取输入
'''
def regression(input):
    return sess.run(y1, feed_dict={x: input}).flatten().tolist()

'''根据输入进行识别（卷积）
'''
def convolutional(input):
    return sess.run(y2, feed_dict={x: input, keep_prob: 1.0}).flatten().tolist()


'''Flask定义接口
'''
app = Flask(__name__)

@qpp.route('/')
def index():
    return render_template('inde.html')

@app.route('/api/mnist', methods=['post'])
def mnist():
    input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 784)
    output1 = regression(input)
    output2 = convolutional(input)
    return jsonify(results = [output1, output2])


'''启动应用
'''
if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=8000)
