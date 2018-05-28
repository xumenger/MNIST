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

'''保存训练结果、参数
'''
saver = tf.train.Saver(variables)

'''开始训练
'''
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 训练1000次
    for _ in range(1000):
        batch_xs, batch_ys = data.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    
    # 训练完成后输出计算准确度
    # x: data.test.images。测试结果集的数据
    # y_: data.test.labels。测试结果集的标签
    print((sess.run(accuracy, feed_dict={x: data.test.images, y_: data.test.labels})))
    
    # 保存参数、模型
    path = saver.save(
        sess, os.path.join(os.path.dirname(__file__), 'data', 'regression.ckpt'),
        write_meta_graph=False, write_state=False)
    # 打印模型路径
    print("Saved: ", path)
