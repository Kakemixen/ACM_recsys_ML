import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import os
from time import time
from sys import exit

import headers

# tf.enable_eager_execution()

dir_path = os.path.dirname(os.path.realpath(__file__))

sessions_csv = dir_path + "/../data/FM_session_vectors.csv"
items_csv = dir_path + "/../data/FM_item_vectors.csv"

### define parameters
DEBUG_DATA = False
num_epochs = 3

x_s_dim = 165
x_i_dim = 158

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

"""
tf.sigmoid returns 0 if x less than some number
since log, this is bad
solution.
add epsilon?
"""

# BPR = -1/25 * tf.reduce_sum(tf.log(tf.sigmoid(y_diff)), 0)
BPR = -1/25 * tf.reduce_sum(tf.log_sigmoid(tf.gather(y_pred, y_true) - y_pred), 0)

TOP1 = 1/25 * tf.reduce_sum(tf.sigmoid(y_pred - tf.gather(y_pred, y_true)) + tf.sigmoid(tf.square(y_pred)), 0)

optimizer = tf.train.AdagradOptimizer(0.1)

train_BPR = optimizer.minimize(BPR)
train_TOP1 = optimizer.minimize(TOP1)

# debug_BPR = tf.is_nan(BPR)
# debug_TOP1 = tf.is_nan(TOP1)
#


# read the things
print("reading session vectors")
session_vectors = pd.read_csv(sessions_csv, index_col=0)

#consider movign in to epoch, with shuffle
train_vectors, test_vectors = train_test_split(session_vectors) #splits default 0.75 / 0.25

plt_training_BPR = []
plt_training_TOP1 = []
plt_validation_BPR = []
plt_validation_TOP1 = []

with tf.Session() as sess:
    loss_BPR = 0
    loss_TOP1 = 0
    print()
    sess.run(tf.global_variables_initializer())
    # x_i_dim = len(item_vectors.iloc[0].values)
    # x_s_dim = x_i_dim + 8
    train_start = time()
    for epoch in range(num_epochs):
        epoch_start = time()
        training_BPR = 0
        training_TOP1 = 0
        print("\nstarting training")
        iter_num = 0
        for index, row in train_vectors.iterrows():
            # print(row)
            # print(row.drop(["choice", "items", "person_id", "session_id"]).values)
            # print(row["choice"])
            # print([x.split(" ")[1:] for x in row["items"].split("|")])
            # exit()
            iter_num += 1
            print("e: {:<3} i: {:<6}|| BPR: {:<5} | TOP1: {:<5}".format(epoch, iter_num, str(loss_BPR)[:5], str(loss_TOP1)[:5]), end="\r")

            try:
                loss_BPR, _, loss_TOP1, _ = sess.run(
                    (
                        BPR, train_BPR,
                        TOP1, train_TOP1
                        ),
                    feed_dict={
                        x_s:    row.drop(["choice", "items", "person_id", "session_id"]).values,
                        y_true: row["choice"],
                        X_i:    [x.split(" ")[1:] for x in row["items"].split("|")]
                        })

                training_BPR += loss_BPR
                training_TOP1 += loss_TOP1

            except KeyError:
                if(DEBUG_DATA):
                    print("\nitem impressions without metadata")
            except ValueError:
                if(DEBUG_DATA):
                    print("\nchoice not in impressions")

        epoch_end = time()
        plt_training_BPR.append(training_BPR/iter_num)
        plt_training_TOP1.append(training_TOP1/iter_num)
        print("\nAverage loss in epoch:")
        print("epoch duration: {}".format(epoch_end-epoch_start))
        print("BPR:  {}".format(plt_training_BPR[epoch]))
        print("TOP1: {}".format(plt_training_TOP1[epoch]))

        print("starting validating")
        valid_BPR = 0
        valid_TOP1 = 0
        iter_num = 0
        test_start = time()
        for index, row in test_vectors.iterrows():
            iter_num += 1
            print("e: {:<3} i: {:<6}".format(epoch, iter_num), end="\r")

            try:
                loss_BPR, loss_TOP1 = sess.run(
                    (BPR,TOP1),
                    feed_dict={
                        x_s:    row.drop(["choice", "items", "person_id", "session_id"]).values,
                        y_true: row["choice"],
                        X_i:    [x.split(" ")[1:] for x in row["items"].split("|")]
                        })

                valid_BPR += loss_BPR
                valid_TOP1 += loss_TOP1

            except KeyError:
                if(DEBUG_DATA):
                    print("\nitem impressions without metadata")
            except ValueError:
                if(DEBUG_DATA):
                    print("\nchoice not in impressions")
        test_end = time()
        plt_validation_BPR.append(valid_BPR/iter_num)
        plt_validation_TOP1.append(valid_TOP1/iter_num)
        print("\nAverage loss in epoch:")
        print("validation duration: {}".format(test_end - test_start))
        print("BPR:  {}".format(plt_validation_BPR[epoch]))
        print("TOP1: {}".format(plt_validation_TOP1[epoch]))

    print("training duration: {}".format(epoch_end - train_start))

    #plotting
    plt.plot(np.arange(num_epochs), plt_training_BPR, 'r-', #label="training BPR",
            np.arange(num_epochs), plt_training_TOP1, 'b-', #label="training TOP1",
            np.arange(num_epochs), plt_validation_BPR, 'r--', #label="validation BPR",
            np.arange(num_epochs), plt_validation_TOP1, 'b--')#, label="validation TOP1")
    plt.legend(["training BPR", "training TOP1", "validation BPR", "validation TOP1"])
    plt.show()


