# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# ----建立输入和输出的占位符--------
X_Data = tf.placeholder(tf.float32, [None, 784])
Y_Data = tf.placeholder(tf.float32, [None, 10])

# ---构建模型-------
w = tf.Variable(tf.zeros([784, 10]))  # 权重矩阵
b = tf.Variable(tf.zeros([10]))  # 偏移矩阵

y = tf.nn.softmax(tf.matmul(X_Data, w) + b)  # 模型

loss = -tf.reduce_sum(Y_Data * tf.log(y))  # 损失函数
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # 梯度下降算法进行拟合

# ----初始化图----------
init = tf.global_variables_initializer()
sess = tf.Session();
sess.run(init)

# ----------进行训练------------------
for i in range(2000):
    batch_x, batch_y = mnist.train.next_batch(100)
    _loss, _ = sess.run([loss, train], feed_dict={X_Data: batch_x, Y_Data: batch_y})
    if i%10 == 0:
        print "loss is ", _loss




# ----------评估模型---------------
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(Y_Data,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print sess.run(accuracy, feed_dict={X_Data: mnist.test.images, Y_Data: mnist.test.labels})

sess.close()