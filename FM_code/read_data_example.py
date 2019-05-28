import numpy as np
import tensorflow as tf
import os

import headers

tf.enable_eager_execution()

dir_path = os.path.dirname(os.path.realpath(__file__))

sessions_csv = dir_path + "../data/FM_session_vectors_small.csv"
items_csv = dir_path + "../data/FM_item_vectors.csv"

sessions_dataset = tf.data.experimental.CsvDataset(
        filenames = sessions_csv,
        record_defaults = [tf.int32 for _ in range(len(headers.get_session_header()))]
        header=True)


