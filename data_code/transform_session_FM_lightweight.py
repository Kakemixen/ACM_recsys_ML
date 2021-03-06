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

def get_device_attr():
    return np.array(["mobile","desktop","tablet"])

def get_device(device):
    if(device == "mobile"):
        return np.array([1,0,0])
    elif(device == "desktop"):
        return np.array([0,1,0])
    else:
        return np.array([0,0,1])

def get_sorting_attr():
    return np.array(["price only","rating only","distance only","our recommendations"])

def get_sorting_update(sorting):
    if sorting == "price only":
        return np.array([1,0,0,0])
    elif sorting == "rating only":
        return np.array([0,1,0,0])
    elif sorting == "distance only":
        return np.array([0,0,1,0])
    elif sorting == "our recommendations":
        return np.array([0,0,0,1])
    elif sorting== "price and recommended":
        return np.array([1,0,0,1])
    elif sorting == "distance and recommended":
        return np.array([0,1,0,1])
    elif sorting == "rating and recommended":
        return np.array([0,0,1,1])
    else:
        return np.array([0,0,0,0])

def get_item_vector(item_id, memory, item_path):
    if item_id == "unknown":
        return (np.array([]), memory) # pls not happen
    item_id = int(item_id)
    if item_id in memory.keys():
        # print("found old item")
        return (memory[item_id], memory)
    # not in memory, find from file
    for chunk in pd.read_csv(item_path, chunksize=200000, index_col=0):
        if item_id in set(chunk.index.values):
            item_vector = np.array(chunk.loc[item_id])
            memory[item_id] = item_vector
            # print("found new item")
            return (item_vector, memory)
    # print("not existing")
    memory[item_id] = np.array([])
    return (np.array([]), memory) # pls not happen

def get_item_matrix(item_ids, prices, item_path):
    item_ids = [int(item_id) for item_id in item_ids]
    item_matrix = ["None" for _ in range(len(item_ids))]
    for chunk in pd.read_csv(item_path, chunksize=100000, index_col=0):
        for itid, item in chunk.iterrows():
            if itid in item_ids:
                matrix_index = item_ids.index(itid)
                item_string = "".join(map(lambda x : str(x) + " ", item.values))
                item_matrix[matrix_index] = str(itid) + " " + item_string + prices[matrix_index]
    return "".join(map(lambda x : x + "|", item_matrix))[:-1]

# return the vector for updating the item vector by applying a filter, al elm in the attributes
def get_filter_update(filter_ref, attr):
    basic = np.array([0 for _ in range(len(attr))])
    for i in range(len(attr)):
        if filter_ref == attr[i]:
            # weigh the filter higly (higher than interacted item)
            basic[i] = 1 # orsomething TODO find a way to let algorithm now what was filtered, could be done by capping
            return basic
    return basic

## encode session and append to file: to_path
# session: pd.DataFrame
def append_session(ids, session, to_path, index, column_names, item_path, item_attributes):
    # attributes | total: item, step, sort, device | don't need the attribute vecotrs
    # item_attributes = get_metadata_vector(item_meta_path)

    # init
    encoded_session_item = np.array([0 for _ in range(len(item_attributes))])
    device_vector = get_device(session.iloc[0]["device"])
    sorting_vector = np.array([0,0,0,0]) #elevate scope, and initialize

    interactions = 0
    memory=dict()
    valid_session = False
    for i, session_row in session.iterrows():
        # update encoded session by info in this row / step
        action = session_row["action_type"]
        if action == "interaction item image": # Item ID
            interactions += 1
            # stuff = get_item_vector(session_row["reference"], memory, item_path)
            # print(stuff)
            # exit()
            item_vector, memory = get_item_vector(session_row["reference"], memory, item_path)
            if len(item_vector) > 0:
                encoded_session_item += item_vector
        elif action == "interaction item rating": # Item ID
            interactions += 1
            item_vector, memory = get_item_vector(session_row["reference"], memory, item_path)
            if len(item_vector) > 0:
                encoded_session_item += item_vector
        elif action == "interaction item info": # Item ID
            interactions += 1
            item_vector, memory = get_item_vector(session_row["reference"], memory, item_path)
            if len(item_vector) > 0:
                encoded_session_item += item_vector
        elif action == "interaction item deals": # Item ID
            interactions += 1
            item_vector, memory = get_item_vector(session_row["reference"], memory, item_path)
            if len(item_vector) > 0:
                encoded_session_item += item_vector
        elif action == "search for item": # Item ID
            interactions += 1
            item_vector, memory = get_item_vector(session_row["reference"], memory, item_path)
            if len(item_vector) > 0:
                encoded_session_item += item_vector
        elif action == "filter selection":
            encoded_session_item += get_filter_update(session_row["reference"], item_attributes) #TODO move to one update when finding clockout item
        elif action == "change of sort order":
            sorting_vector = get_sorting_update(session_row["reference"]) # remembering last sorting
        elif action == "clickout item": #the special snowflake
            # indexes.append(index)

            item_indexes = session_row["impressions"].split("|")
            clicked_index = session_row["reference"]
            if not clicked_index in item_indexes: continue
            item_impressions = get_item_matrix(item_indexes, session_row["prices"].split("|"), item_path)

            clicked_out = item_indexes.index(clicked_index)

            # set final step
            step_vector = np.array([min(session_row["step"]/20, 1)])

            valid_session = True

        else:
            # guess we done here
            pass

    if valid_session:
        # total: item, step, sort, device
        # append teh different vectors to make the session vector with item\price references
        if interactions > 1:
            encoded_session_item = encoded_session_item / interactions #TODO cap at 1 maybe?
        encoded_session = np.append(encoded_session_item, step_vector)
        encoded_session = np.append(encoded_session, sorting_vector)
        encoded_session = np.append(encoded_session, device_vector)
        encoded_session = np.append(encoded_session, [clicked_out, item_impressions])
        encoded_session = np.append(ids, encoded_session)
        encoded_df = pd.DataFrame([encoded_session], index = [index], columns = np.append(["person_id", "session_id"],np.append(column_names,["choice", "items"])))
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
    non_item_attributes = np.append(
                np.array(["step"]), # begins index 0
                np.append(
                    get_sorting_attr(), #begins index 1
                    get_device_attr() #begins index 5
            )) #total elm 8
    attributes = np.append(item_attributes, non_item_attributes)
    n_attr = len(attributes)

    # wiriting the header to file
    df = pd.DataFrame([], columns=np.append(["person_id", "session_id"],np.append(attributes,["choice", "items"])))
    df.to_csv(to_path, mode="w+", header=True)

    # we need to make sure sessions spanning multiple chunks are preserved, elevate scope
    # last_session_ids=(0,0) # basic start
    row1 = pd.read_csv(path, nrows=1)
    last_session_ids = (row1.loc[0,"user_id"], row1.loc[0,"session_id"]) # more sophisticated start

    new_session=True # flag to use for creating a new chunk matrix
    for chunk in pd.read_csv(path, chunksize=256): # split up into chunks cus is too damn much!

        for index, session_row in chunk.iterrows():
            if not session_row["session_id"] == last_session_ids[1]:
                written = append_session(last_session_ids, session.T, to_path, session_number, attributes, item_path, item_attributes)
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



    print("\nDone!")
    print("File can be found at: {}".format(to_path))

if __name__ == "__main__":
    path_processed_meta = "../data/all_properties.csv"
    path_session = "../data/train.csv"

    to_path_item = "../data/FM_item_vectors.csv"
    to_path_session = "../data/FM_session_vectors.csv"

    process_session_data(path_session, to_path_session, to_path_item, path_processed_meta)

