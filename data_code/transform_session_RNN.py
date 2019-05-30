from time import time
from sys import exit
from pathlib import Path
import numpy as np
import pandas as pd

def get_metadata_vector(path):
    if not Path(path).is_file():
        # print("need to process metadata")
        import extract_all_properties as exprop
        exprop.extract_all_properties(path_item, path)
        # print("done processing metadata")

    df = pd.read_csv(path)
    return np.array(df["0"])

def get_sorting_attr():
    return np.array(["price only","rating only","distance only","our recommendations"])

def get_sorting_string(sorting):
    if sorting == "price only":
        return "1 0 0 0"
    elif sorting == "rating only":
        return "0 1 0 0"
        return np.array([0,1,0,0])
    elif sorting == "distance only":
        return "0 0 1 0"
        return np.array([0,0,1,0])
    elif sorting == "our recommendations":
        return "0 0 0 1"
        return np.array([0,0,0,1])
    elif sorting== "price and recommended":
        return "0 0 0 1"
        return np.array([1,0,0,1])
    elif sorting == "distance and recommended":
        return "0 1 0 1"
        return np.array([0,1,0,1])
    elif sorting == "rating and recommended":
        return "0 0 1 1"
        return np.array([0,0,1,1])
    else:
        return "0 0 0 0"
        return np.array([0,0,0,0])

encoded_items = pd.read_csv("../data/FM_item_vectors.csv",index_col=0) # use the same item vectors as FM

# get the item as a string vector separated by " "
def get_item_string(item_id):
    empty_vector = np.array([0 for _ in encoded_items.iloc[0]])
    if item_id == "unknown" or item_id == "empty":
        return "".join(map(lambda x : str(x) + " ", empty_vector))[:-1]
    item_id = int(item_id)
    if item_id in set(encoded_items.index.values):
        item_vector = np.array(encoded_items.loc[item_id])
        return "".join(map(lambda x : str(x) + " ", item_vector))[:-1]
    return "".join(map(lambda x : str(x) + " ", empty_vector))[:-1]

def get_item_matrix(item_ids):
    item_matrix = ["None" for _ in range(len(item_ids))]
    for i in range(len(item_ids)):
        item_matrix[i] = item_ids[i] + " " + get_item_string(item_ids[i])
    return "".join(map(lambda x : x + "|", item_matrix))[:-1]

# return the vector for updating the item vector by applying a filter, al elm in the attributes
def get_filter_update(filter_ref, attr):
    basic = np.array(["0" for _ in range(len(attr))])
    for i in range(len(attr)):
        if filter_ref == attr[i]:
            # weigh the filter higly (higher than interacted item)
            basic[i] = "1"
            return "".join(map(lambda x : x + " ", basic))[:-1]
    return "".join(map(lambda x : x + " ", basic))[:-1]

# returns the session matrix as a string separated by "|" between vecotrs, " " between elements

## encode session and append to file: to_path
# session: pd.DataFrame
def append_session(ids, session, to_path, index, item_path, item_attributes):
    # attributes | total: item, step, sort, device | don't need the attribute vecotrs
    # item_attributes = get_metadata_vector(item_meta_path)

    # init
    encoded_session_item = np.array([0 for _ in range(len(item_attributes))])
    sorting_vector = np.array([0,0,0,0]) #elevate scope, and initialize

    empty_sorting_string = "0 0 0 0"

    interactions = 0
    recorded_step = 0
    valid_session = False
    session_string = ""
    for i, session_row in session.iterrows():
        # update encoded session by info in this row / step
        action = session_row["action_type"]
        # Item ID
        if action == "interaction item image" or action == "interaction item rating" or \
                action == "interaction item info" or action == "interaction item deals" or action == "search for item":
            interactions += 1
            recorded_step += 1
            step_string = str(recorded_step) + " " + get_item_string(session_row["reference"]) + " " + empty_sorting_string +"|"
            if len(step_string) > 0:
                session_string += step_string
        elif action == "filter selection": # filter choice, elm in item_attibutes
            recorded_step += 1
            session_string += str(recorded_step) + " " + get_filter_update(session_row["reference"], item_attributes) + " " + empty_sorting_string + "|"
        elif action == "change of sort order":
            recorded_step += 1
            session_string +=str(recorded_step) + " " + get_item_string("empty") + " " + get_sorting_string(session_row["reference"]) + "|"

        elif action == "clickout item": #the special snowflake
            recorded_step += 1

            item_indexes = session_row["impressions"].split("|")
            clicked_index = session_row["reference"]
            if not clicked_index in item_indexes:
                clicked_out = "Unknown"
            else:
                clicked_out = item_indexes.index(clicked_index)

            item_impressions = get_item_matrix(item_indexes)

            valid_session = True

        else:
            # guess we done here
            pass

    if valid_session:
        # append string to array, then transform to df
        encoded_session = np.append(ids, [session_string[:-1], clicked_out, item_impressions])
        encoded_df = pd.DataFrame([encoded_session], index = [index], columns = ["person_id", "session_id","sessions","choice", "items"])
        # append the  encoded vector to file
        encoded_df.to_csv(to_path, mode='a', header=False)
    # exit()
    return valid_session

## processes session data to be used in the FM
def process_session_data(path, to_path, item_path, item_meta_path):
    """
    The idea is to find all rows connected to a session, assuming a session is only in following rows.
    encode that session
    write that session to file
    """
    print("This will take some time. All processes up until last session will be kept")

    #index fro writingo
    session_number = 0

    # get header for file
    item_attributes = get_metadata_vector(item_meta_path)

    # wiriting the header to file
    df = pd.DataFrame([], columns=["person_id", "session_id","sessions","choice", "items"])
    df.to_csv(to_path, mode="w+", header=True)

    # we need to make sure sessions spanning multiple chunks are preserved, elevate scope
    # last_session_ids=(0,0) # basic start
    row1 = pd.read_csv(path, nrows=1)
    last_session_ids = (row1.loc[0,"user_id"], row1.loc[0,"session_id"]) # more sophisticated start

    new_session=True # flag to use for creating a new chunk matrix
    start=time()
    for chunk in pd.read_csv(path, chunksize=256): # split up into chunks cus is too damn much!

        for index, session_row in chunk.iterrows():
            if not session_row["session_id"] == last_session_ids[1]:
                written = append_session(last_session_ids, session.T, to_path, session_number, item_path, item_attributes)
                if written: session_number += 1

                new_session = True # makes session be new np.array([])

                # set last session to this session
                last_session_ids = (session_row["user_id"],session_row["session_id"])


            if new_session:
                print("Processing session {:<6} {}".format(session_number, last_session_ids), end="\r")
                session = pd.DataFrame(session_row)#start the DF
                new_session = False
            else:
                session = pd.concat([session,session_row], axis=1)

    end = time()

    print("\nDone!")
    print("diraction: {}".format(end-start))
    print("File can be found at: {}".format(to_path))

if __name__ == "__main__":
    path_processed_meta = "../data/all_properties.csv"
    path_session = "../data/train.csv"

    to_path_item ="../data/FM_item_vectors.csv"
    to_path_session ="../data/RNN_session_matrices.csv"

    process_session_data(path_session, to_path_session, to_path_item, path_processed_meta)

#example data_read
if False:
    df = pd.read_csv("../data/RNN_session_matrices.csv", index_col=0)
    session_steps = [x.split(" ") for x in df["sessions"].split("|")]
    session_items = [x.split(" ") for x in df["items"].split("|")]

