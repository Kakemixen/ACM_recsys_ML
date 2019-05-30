from time import time
from sys import exit
from pathlib import Path
import numpy as np
import pandas as pd


df = pd.read_csv("../data/RNN_session_matrices.csv", index_col=0)
for index, row in df.iterrows():
    # print(row["sessions"].split("|"))
    # print([x.split(" ") for x in row["sessions"].split("|")])
    # break
    session_steps = [x.split(" ") for x in row["sessions"].split("|")]
    session_items = [x.split(" ") for x in row["items"].split("|")]

    print("session steps - length: ", end="")
    print(len(session_steps))
    print(session_steps[:5])
    print("session items - length: ", end="")
    print(len(session_items))
    print(session_items[:5])

    break
