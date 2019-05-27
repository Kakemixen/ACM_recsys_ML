import numpy as np
import pandas as pd
import tensorflow as tf
import os

import headers

# tf.enable_eager_execution()

dir_path = os.path.dirname(os.path.realpath(__file__))

sessions_csv = dir_path + "/../data/FM_session_vectors_medium.csv"
items_csv = dir_path + "/../data/FM_item_vectors.csv"

### define parameters
num_epochs = 10

x_s_dim = 165
x_i_dim = 157

d = 3




# def sigmoid(x): return 1/(1+np.exp(-x))
def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm

# input
x_s = tf.placeholder(tf.float32, name='x_s')
X_i = tf.placeholder(tf.float32, name='X_i')
y_true = tf.placeholder(tf.float32, name='X_i')

# #parameters
Q = tf.get_variable("Q", shape=(x_s_dim, d))
P = tf.get_variable("P", shape=(x_i_dim, d))
b_p = tf.get_variable("b_p", shape=(x_s_dim))
b_q = tf.get_variable("b_q", shape=(x_i_dim))

#calculation
dot_Q = tf.matmul(tf.expand_dims(x_s,0), Q)
dot_P = tf.matmul(X_i, P)

dot = tf.matmul(dot_P, tf.transpose(dot_Q))

w_s = tf.reduce_sum(tf.multiply(x_s, b_p))
w_i = tf.reduce_sum(tf.multiply(X_i, b_q))

y_pred = dot + w_s + w_i

# loss

# BPR =
#
# TOP1 =
#

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    item_vectors = pd.read_csv(items_csv, index_col=0)
    # x_i_dim = len(item_vectors.iloc[0].values)
    # x_s_dim = x_i_dim + 8

    for epoch in range(num_epochs):
        for chunk in pd.read_csv(sessions_csv, chunksize=512, index_col=0):
            for index, row in chunk.iterrows():
                print("e: {:<3} i: {}".format(epoch, index))


                print(sess.run((y_pred, y_true),
                    feed_dict={
                        x_s:    row.drop(["choice", "items", "prices", "person_id", "session_id"]).values,
                        y_true: row["items"].split("|").index(str(row["choice"])),
                        X_i:    [item_vectors.loc[int(x)].values for x in row["items"].split("|")]
                    }))

                break
            break
        break

