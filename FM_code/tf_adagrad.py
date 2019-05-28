import numpy as np
import tensorflow as tf


x_s_dim = 4
x_i_dim = 3
d =  2


# data 4 sets of 3 items
X = [[[1,2,3],
    [2,7,2],
    [6,1,1],],
    [[1,2,3],
    [2,2,5],
    [6,1,1],],
    [[1,2,3],
    [2,7,2],
    [9,1,7],],
    [[3,2,1],
    [2,2,5],
    [9,1,7],]]
# 4 sessions
S =     [[1,2,1,3],
        [1,7,9,5],
        [9,2,3,1],
        [1,4,7,9]]

# True
T = [0,0,2,1]

# def sigmoid(x): return 1/(1+np.exp(-x))
def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm


# a loss function
def BPR(pred_scores, pos_item):
    N_s = len(pred_scores) # number of samples
    return (1/(N_s))*sum((np.log(sigmoid(pred_scores[pos_item] - pred_scores[j])) for j in range(N_s) if not j == pos_item))

# another loss function
def TOP1(pred_scores, pos_item):
    N_s = len(pred_scores) # number of samples
    return (1/(N_s))*sum(sigmoid(pred_scores[j] - pred_scores[pos_item]) + sigmoid(pred_scores[j]**2) for j in range(N_s) if not j == pos_item)


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

# loss

# BPR =
#
# TOP1 =
#

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # print(sess.run(huh, feed_dict={x_s: [2.0, 1, 5, 1]}))
    for s in range(len(S)):
        print("session: {}".format(s))
        ratings = [0 for _ in range(len(X[s]))]
        for i in range(len(X[s])):
            rating = sess.run((y_pred), feed_dict={x_s: S[s], x_i:X[s][i]})
            ratings[i] = rating
        print(ratings)

        print("loss")
        print("BPR: {}".format(BPR(ratings, T[s])))
        print("TOP1: {}".format(TOP1(ratings, T[s])))



