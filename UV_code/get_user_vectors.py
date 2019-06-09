from time import time
from sys import exit
import os
from pathlib import Path

import numpy as np
import pandas as pd

dir_path = os.path.dirname(os.path.realpath(__file__))
sessions_csv = dir_path + "/../data/FM_session_vectors.csv"

max_money = 300

## processes user data to be used in the PCA
def process_session_data():
    session_vectors = pd.read_csv(sessions_csv, index_col=0)
    print("starting")

    memory = dict()

    for index, row in session_vectors.iterrows():
        print("i: {}".format(index), end="\r")
        user = row["person_id"]
        y = row["choice"]
        x_tmp = [x for x in row["items"].split("|")][int(y)]
        if x_tmp == "None":
            continue

        x =  [int(b) for b in x_tmp.split(" ")[1:]]
        x[-1] /= max_money
        x[-1] = min(x[-1], 1)

        if user in memory.keys():
            memory[user].append(x)
        else:
            memory[user] = [x]

    # average user items
    print()
    for person in memory.keys():
        print("p: {}".format(person), end="\r")
        memory[person] = (np.array([memory[person]]).sum(1) / len(memory[person])).flatten()
    print()
    return memory

def write_to_file(memory, include_id=False):
    if not include_id:
        pd.DataFrame(list(memory.values())).to_csv("./user_vectors_noid.csv", index=False)
    else:
        pd.DataFrame(memory).to_csv("./user_vectors_id.csv", index = False)

if __name__ == "__main__":
    users = process_session_data()
    write_to_file(users)
