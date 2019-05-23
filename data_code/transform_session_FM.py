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
def append_session(ids, session, to_path, index, column_names, item_path, item_meta_path):
    print()
    # attributes | total: item, step, sort, device | don't need the attribute vecotrs
    item_attributes = get_metadata_vector(item_meta_path)
    # step_attributes = np.array(["step"]), # begins index 0
    # sorting_attributes = get_sorting_attr(), #begins index 1
    # device_attributes = get_device_attr() #begins index 5

    # init
    encoded_session_item = np.array([0 for _ in range(len(item_attributes))])
    device_vector = get_device(session.iloc[0]["device"])
    sorting_vector = np.array([0,0,0,0]) #elevate scope, and initialize

    interactions = 0
    memory=dict()
    valid_session = False
    for i, session_row in session.iterrows():
        # update encoded session by info in this step
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

            #TODO get some item possibilities as a column
            #TODO get  column where possibilities are split by , per clickout split by | (1,2,3|2,6,1|7,8,9)
            item_impressions = "impression list" # find them
            item_prices = "price list" #find them
            #TODO get a column with clickouts split by | (2|6|8)
            clicked_out = "item id" # find if exists

            #TODO set final sorting - from elif above. igmore for now
            #TODO set final step
            step_vector = np.array([min(session_row["step"]/20, 1)])

            valid_session = True

        else:
            # guess we done here
            pass

    print(valid_session, end="")
    if valid_session:
        # total: item, step, sort, device
        if interactions > 1:
            encoded_session_item = encoded_session_item / interactions #TODO cap at 1 maybe?
        encoded_session = np.append(encoded_session_item, step_vector)
        encoded_session = np.append(encoded_session, sorting_vector)
        encoded_session = np.append(encoded_session, device_vector)
        encoded_session = np.append(encoded_session, [clicked_out, item_impressions, item_prices])
        encoded_session = np.append(ids, encoded_session)
        encoded_df = pd.DataFrame([encoded_session], index = [index], columns = np.append(["person_id", "session_id"],np.append(column_names,["choice", "items", "prices"])))
        #TODO write to file
        print(" - writing",end="")
        encoded_df.to_csv(to_path, mode='a', header=False)
    print()
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
    # wiriting the header to file
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
    df = pd.DataFrame([], columns=np.append(["person_id", "session_id"],np.append(attributes,["choice", "items", "prices"])))
    df.to_csv(to_path, mode="w+", header=True)

    # we need to make sure sessions spanning multiple chunks are preserved, elevate scope
    # last_session_ids=(0,0) # basic start
    row1 = pd.read_csv(path, nrows=1)
    last_session_ids = (row1.loc[0,"user_id"], row1.loc[0,"session_id"]) # more sophisticated start

    new_session=True # flag to use for creating a new chunk matrix
    for chunk in pd.read_csv(path, chunksize=512): # split up into chunks

        # encodes item as one-hot of attributes
        # chunk_row=0
        for index, session_row in chunk.iterrows():
            if not session_row["session_id"] == last_session_ids[1]:
                #TODO append last session to file from last session found
                written = append_session(last_session_ids, session.T, to_path, session_number, attributes, item_path, item_meta_path)
                if written: session_number += 1

                #TODO empty found session
                new_session = True # makes session be new np.array([])
                #TODO set last session to this session
                last_session_ids = (session_row["user_id"],session_row["session_id"])
                print(last_session_ids)

            #TODO find session

            if new_session:
                session = pd.DataFrame(session_row)#start the DF
                new_session = False
            else:
                session = pd.concat([session,session_row], axis=1)

            print("Porcessing session {}".format(session_number), end="\r")
            # chunk_row += 1


        break #TODO remove

    print("\nDone!")
    print("File can be found at: {}".format(to_path))

if __name__ == "__main__":
    path_processed_meta = "../data/all_properties.csv"
    path_session = "../data/train.csv"
    path_item = "../data/item_metadata.csv"

    to_path_item = "../data/FM_item_vectors.csv"
    to_path_session = "../data/FM_session_vectors.csv"

    process_session_data(path_session, to_path_session, to_path_item, path_processed_meta)

