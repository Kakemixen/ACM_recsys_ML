from sys import argv, exit
import pandas as pd

path = "../data/train.csv"

valid_actions =  {"change of sort order" ,   "filter selection"}

if len(argv) <= 1:
    print("have argument, among: " + str(valid_actions))
    exit()

if not argv[1] in valid_actions:
    print("valid options: " + str(valid_actions))
    exit()

possibilities = set()


for chunk in pd.read_csv(path, chunksize=1000000): # split up into chunks cus memory error
    for index, item in chunk.iterrows():
        print(index)
        if(item["action_type"] == argv[1]):
            x = item["reference"]
            if not(x in possibilities):
                possibilities.add(x)

df = pd.DataFrame(list(possibilities)).transpose()
df.to_csv("../data/possibilities_for_{}.csv".format(argv[1]), index=False)

