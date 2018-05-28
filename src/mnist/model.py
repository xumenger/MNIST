import tensorflow as tf

'''线性模型
Y = W * x + b
'''
def regression(x):
    # 定义一个全零的、784*10的图像
    W = tf.Variable(tf.zeros([784, 10]), name='W')
    b = tf.Variable(tf.zeros([10]), name='b')
    # softmax是一个用来进行简单向量运算的函数
    # 下面是使用一个向量运算的乘法，也就是公式：Y = W * x + b
    Y = tf.nn.softmax(tf.matmul((x, W)) + b)

    return Y, [W, b]
