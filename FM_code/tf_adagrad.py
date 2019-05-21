import numpy as np
import tensorflow as tf


x_s_dim = 4
x_i_dim = 3
d =  2

# input
x_s = tf.placeholder(tf.float32, name='x_s')
x_i = tf.placeholder(tf.float32, name='x_i')

#parameters
Q = tf.get_variable("Q", shape=(x_s_dim, d))
P = tf.get_variable("P", shape=(x_i_dim, d))
b_p = tf.get_variable("b_p", shape=(x_s_dim))
b_q = tf.get_variable("b_q", shape=(x_i_dim))

#calculation
dot_Q = tf.matmul(tf.expand_dims(x_s,0), Q)
dot_P = tf.matmul(tf.expand_dims(x_i,0), P)

dot = tf.reshape(tf.matmul(dot_Q, tf.transpose(dot_P)), [])

w_s = tf.reduce_sum(tf.multiply(x_s, b_p))
w_i = tf.reduce_sum(tf.multiply(x_i, b_q))

y_pred = dot + w_s + w_i

# parameters
# Q = tf.Variable(tf.random_uniform([d, x_s_dim]))
# dot_Q = tf.tensordot(x_s, Q, 0)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # print(sess.run(huh, feed_dict={x_s: [2.0, 1, 5, 1]}))
    print(sess.run((dot, w_s, w_i), feed_dict={x_s: [2.0, 1, 5, 1], x_i:[1,4,2]}))


