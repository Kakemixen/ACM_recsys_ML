import numpy as np
import pandas as pd
import tensorflow as tf
import os
from time import time
from sys import exit

import headers

# tf.enable_eager_execution()

dir_path = os.path.dirname(os.path.realpath(__file__))

sessions_csv = dir_path + "/../data/FM_session_vectors_decent.csv"
items_csv = dir_path + "/../data/FM_item_vectors.csv"

### define parameters
DEBUG_DATA = False
num_epochs = 10

x_s_dim = 165
x_i_dim = 157

d = 3


# input
x_s = tf.placeholder(tf.float32, name='x_s')
X_i = tf.placeholder(tf.float32, name='X_i')
y_true = tf.placeholder(tf.int32, name='y_t')

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
y_pred = tf.reshape(y_pred, [-1])

# loss

# a vector containing the differences y_pred[y_true] - y_pred
y_diff = tf.gather(y_pred, y_true) - y_pred

"""
tf.sigmoid returns 0 if x less than some number
since log, this is bad
solution.
add epsilon?
"""

# BPR = -1/25 * tf.reduce_sum(tf.log(tf.sigmoid(y_diff)), 0)
BPR = -1/25 * tf.reduce_sum(tf.log_sigmoid(y_diff), 0)

TOP1 = 1/25 * tf.reduce_sum(tf.sigmoid(y_diff) + tf.sigmoid(tf.square(y_pred)), 0)

optimizer = tf.train.AdagradOptimizer(0.1)

train_BPR = optimizer.minimize(BPR)
train_TOP1 = optimizer.minimize(TOP1)

debug_BPR = tf.is_nan(BPR)
debug_TOP1 = tf.is_nan(TOP1)

with tf.Session() as sess:
    loss_BPR = 0
    loss_TOP1 = 0
    print()
    sess.run(tf.global_variables_initializer())
    print("reading item vectors")
    item_vectors = pd.read_csv(items_csv, index_col=0)
    # x_i_dim = len(item_vectors.iloc[0].values)
    # x_s_dim = x_i_dim + 8
    print("starting training")
    train_start = time()
    for epoch in range(num_epochs):
        epoch_start = time()
        total_BPR = 0
        total_TOP1 = 0
        sessions = 0
        for chunk in pd.read_csv(sessions_csv, chunksize=512, index_col=0):
            for index, row in chunk.iterrows():
                print("e: {:<3} i: {:<6}|| BPR: {:<4} | TOP1: {:<4}".format(epoch, index, round(loss_BPR, 3), round(loss_TOP1, 3)), end="\r")

                try:
                    loss_BPR, _, loss_TOP1, _, dB, dT = sess.run(
                        (
                            BPR, train_BPR,
                            TOP1, train_TOP1,
                            debug_BPR, debug_TOP1
                            ),
                        feed_dict={
                            x_s:    row.drop(["choice", "items", "prices", "person_id", "session_id"]).values,
                            y_true: row["items"].split("|").index(str(row["choice"])),
                            X_i:    [item_vectors.loc[int(x)].values for x in row["items"].split("|")]
                            })

                    total_BPR += loss_BPR
                    total_TOP1 += loss_TOP1
                    sessions += 1

                except KeyError:
                    if(DEBUG_DATA):
                        print("\nitem impressions without metadata")
                except ValueError:
                    if(DEBUG_DATA):
                        print("\nchoice not in impressions")

                if dB or dT:
                    exit()

        epoch_end = time()
        print("\nAverage loss in epoch:")
        print("epoch duration: {}".format(epoch_end-epoch_start))
        print("BPR:  {}".format(total_BPR/sessions))
        print("TOP1: {}".format(total_TOP1/sessions))

    print("training duration: {}".format(epoch_end - train_start))


                # if index > 10: break
            # break
        # break

