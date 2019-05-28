import numpy as np
import pandas as pd
import tensorflow as tf
import os
from time import time
from sys import exit

df = pd.read_csv("../data/FM_session_vectors_decent.csv", index_col=0)

y_pred = tf.constant([0,0.5,1000,1.5,0.3])
y_true = tf.constant(2)

# loss

# BPR = -1/25 * tf.reduce_sum(tf.log(tf.sigmoid(y_diff)), 0)
BPR = -1/25 * tf.reduce_sum(tf.log(tf.sigmoid(tf.gather(y_pred, y_true) - y_pred)), 0)

TOP1 = 1/25 * tf.reduce_sum(tf.sigmoid(y_pred - tf.gather(y_pred, y_true)) + tf.sigmoid(tf.square(y_pred)), 0)

debug_BPR = tf.is_nan(BPR)
debug_TOP1 = tf.is_nan(TOP1)

with tf.Session() as sess:

    loss_BPR, loss_TOP1, dB, dT = sess.run(
        (
            BPR,
            TOP1,
            debug_BPR, debug_TOP1
            ))

    if dB or dT:
        pass
    print(y_pred.eval())
    print(loss_BPR)
    print(loss_TOP1)
    print(dB)
    print(dT)

