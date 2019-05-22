from pathlib import Path
import numpy as np
import pandas as pd

## gets the metadata vector described in path
# @ param
#   path: the path of processed metadata file | string
# @ return
#  the metadata vector | np.array([])
def get_metadata_vector(path):
    if not Path(path).is_file():
        print("need to process metadata")
        import extract_all_properties as exprop
        exprop.extract_all_properties(path_item, path)
        print("done processing metadata")

    df = pd.read_csv(path)
    return np.array(df["0"])

## processes item data to be used in the FM
# @ param
#   path: the path of processed metadata file | string
#   to_path: the path tostore file in | string
#   processed_metadata_path: the file storing the file with the metadata attributes (from extract_all_properties.py)| string
# @ return
#   None, but writes csv to file to_path
def process_item_data(path, to_path, processed_metadata_path):
    print("This will take some time. Progress(will still keep up to last batch if stopped early):")

    one_hot_attributes = get_metadata_vector(processed_metadata_path) # one hot encoding header list
    n_attr = len(one_hot_attributes)

    empty_item = np.array([[0 for _ in range(len(one_hot_attributes))]], dtype=bool) # baseline used to start a new item

    # wiriting the header to file
    df = pd.DataFrame([], columns=one_hot_attributes)
    df.to_csv(to_path, mode="w+", header=True)

    for chunk in pd.read_csv(path, chunksize=512): # split up into chunks cus memory error
        first_in_chunk=True # flag to use for creating a new chunk matrix
        indexes = np.array([], dtype=np.int64)

        # encodes item as one-hot of attributes
        for index, item in chunk.iterrows():
            encoded_item = empty_item.copy()
            for item_attribute in item["properties"].split("|"):
                for i in range(n_attr):
                    if(item_attribute == one_hot_attributes[i]):
                        encoded_item[0][i] = 1

            # storing calculated encoded_item in a matrix | np.array([[]])
            if first_in_chunk:
                encoded_items = np.array(encoded_item) #start the list
                first_in_chunk = False
            else:
                encoded_items = np.append(encoded_items,encoded_item,0)
            indexes = np.append(indexes, int(item["item_id"]))
            print("i:{:<7} t:{}   ".format(index, 927141), end="\r")

        #writing the data as processed under the header already written
        df = pd.DataFrame(encoded_items, columns=one_hot_attributes, index=indexes)
        df.to_csv(to_path, mode='a', header=False)
    print("\nDone!")
    print("File can be found at: {}".format(to_path))



def get_device_attr():
    return np.array(["mobile",
        "desktop",
        "tablet"
    ])

def get_device(device):
    if(device == "mobile"):
        return np.array([1,0,0])
    elif(device == "desktop"):
        return np.array([0,1,0])
    else:
        return np.array([0,0,1])

def get_item_vector(item_id, memory, item_path):
    if item_id in memory.keys():
        return memory[item_id]
    # not in memory, find from file
    for chunk in pd.read_csv(item_path, chunksize=512):
        if item_id in chunk.index.values:
            item_vector = chunk.loc[item_id]
            memory[item_id] = item_vector
            return item_vector, memory


def get_filter_update(filter_ref, meta_path):
    meta = get_metadata_vector(meta_path)
    basic = np.array([0 for _ in range(len(meta))])
    for i in range(len(meta)):
        if filter_ref == meta[i]:
            basic[i] = 10 # orsomething TODO find a way to let algorithm now what was filteredal
            return basic
    return basic

def get_sorting_attr():
    return np.array(["price only",
        "rating only",
        "distance only",
        "our recommendations"
        # "price and recommended",      | can be used by vectors like (0,1,0,1)
        # "distance and recommended",
        # "rating and recommended"
        ])

def get_sorting_update(sorting, encoded):
    if sorting == "price only":
        return np.array([encoded[0],1,0,0,0,encoded[5], encoded[6], encoded[7]])
    elif sorting == "rating only":
        return np.array([encoded[0],0,1,0,0,encoded[5], encoded[6], encoded[7]])
    elif sorting == "distance only":
        return np.array([encoded[0],0,0,1,0,encoded[5], encoded[6], encoded[7]])
    elif sorting == "our recommendations":
        return np.array([encoded[0],0,0,0,1,encoded[5], encoded[6], encoded[7]])
    elif sorting== "price and recommended":
        return np.array([encoded[0],1,0,0,1,encoded[5], encoded[6], encoded[7]])
    elif sorting == "distance and recommended":
        return np.array([encoded[0],0,1,0,1,encoded[5], encoded[6], encoded[7]])
    elif sorting == "rating and recommended":
        return np.array([encoded[0],0,0,1,1,encoded[5], encoded[6], encoded[7]])
    else:
        return np.array([encoded[0],0,0,0,0,encoded[5], encoded[6], encoded[7]])

## processes session data to be used in the FM
# @ param
#   path: the path of processed metadata file | string
#   to_path: the path tostore file in | string
#   item_path: the path to the processed item file | string
#   item_meta_path: the file storing the file with the metadata attributes (from extract_all_properties.py)| string
# @ return
#   None, but writes csv to file to_path
def process_session_data(path, to_path, item_path, item_meta_path):
    """
    The plan is to update encoded_session_item, and encoded_session_non_item during a session.
    When that session ends, save the concated arrays in a matrix.
    When the batch is done, append the matrix to file
    """
    print("This will take some time. Progress(will still keep up to last batch if stopped early):")

    memory = dict{} # to store items interacted with in a session
    item_attributes = get_metadata_vector(item_meta_path)

    non_item_attributes = np.append(
                np.array(["step"]), # begins index 0
                np.append(
                    get_sorting_attr(), #begins index 1
                    get_device_attr() #begins index 5
            )) #total elm 8
    attributes = np.append(item_attributes, non_item_attributes)
    print(attributes)
    n_attr = len(attributes)
    empty_session_item = np.array([0 for _ in range(len(item_attributes))]) # baseline used to start a new session
    empty_session_non_item = np.array([0 for _ in range(len(non_item_attributes))]) # baseline used to start a new session

    # wiriting the header to file
    df = pd.DataFrame([], columns=attributes)
    df.to_csv(to_path, mode="w+", header=True)

    # we need to make sure sessions spanning multiple chunks are preserved
    # last_session=(0,0) # basic start
    row1 = pd.read_csv(path, nrows=1)
    last_session = (row1["user_id"], row1["session_id"]) # more sophisticated start, to avoid storing initial state when iteration starts

    # to extend scope of variables between batches
    encoded_session_item = empty_session_item.copy()
    encoded_session_non_item = empty_session_non_item.copy()
    last_interactions = 0

    session_index = 0
    indexes = np.array([], dtype=np.int64) #indexes to be used when writing

    for chunk in pd.read_csv(path, chunksize=512): # split up into chunks
        first_in_chunk=True # flag to use for creating a new chunk matrix

        # encodes item as one-hot of attributes
        for index, item in chunk.iterrows():
            # check if start new session, which means save last session
            if not item["session_id"] == last_session[1]:
                # storing last calculated encoded_item in a matrix | np.array([[]])
                encoded_session_item = encoded_session_item / last_interactions #average the one-hot
                encoded_session = np.append(encoded_session_item, encoded_session_non_item) # add dimension to enable appending to matrix
                if first_in_chunk:
                    encoded_sessions = np.array([np.append(np.array([last_session[0], last_session[1]]),encoded_session)]) #start the list
                    first_in_chunk = False
                else:
                    encoded_sessions = np.append(encoded_sessions,np.array([np.append(np.array([last_session[0], last_session[1]]),encoded_session)]),0)

                indexes = np.append(indexes, session_index)

                # get ready for new session
                session_index += 1
                last_interactions = 0
                last_session = (item["user_id"], item["session_id"])
                encoded_session_item = empty_session_item.copy()
                encoded_session_non_item = empty_session_non_item.copy()
                encoded_session_non_item += np.append(np.array([0,0,0,0,0]),get_device(item["device"])) #not dependent on actions, update device now
                memory = dict{} #TODO find better forget rule for memory

            # update encoded session by info in this step
            action = item["action_type"]
            encoded_session_non_item[0] += 1 #update step
            if action = "interaction item image": # Item ID
                last_interactions += 1
                item_vector, memory = get_item_vector(item["reference"], memory, item_path)
                encoded_session_item += item_vector
            elif action = "interaction item rating": # Item ID
                last_interactions += 1
                item_vector, memory = get_item_vector(item["reference"], memory, item_path)
                encoded_session_item += item_vector
            elif action = "interaction item info": # Item ID
                last_interactions += 1
                item_vector, memory = get_item_vector(item["reference"], memory, item_path)
                encoded_session_item += item_vector
            elif action = "interaction item deals": # Item ID
                last_interactions += 1
                item_vector, memory = get_item_vector(item["reference"], memory, item_path)
                encoded_session_item += item_vector
            elif action = "search for item": # Item ID
                last_interactions += 1
                item_vector, memory = get_item_vector(item["reference"], memory, item_path)
                encoded_session_item += item_vector
            elif action = "change of sort order":
                encoded_session_non_item = get_sorting_update(item["reference"], encoded_session_non_item)
            elif action = "filter selection":
                encoded_session_item += get_filter_update(item["reference"])
            elif action = "clickout item": #the special snowflake
                pass #TODO get some item possibilities as a column
            else:
                # guess we done here
                pass

            print("i:{:<7} t:{}   ".format(index, "large number"), end="\r")

        #writing the data as processed under the header already written
        df = pd.DataFrame(encoded_sessions, columns=np,append(np.array(["person_id", "session_id"]),attributes), index=indexes)
        df.to_csv(to_path, mode='a', header=False)

        break #TODO remove

    print("\nDone!")
    print("File can be found at: {}".format(to_path))

if __name__ == "__main__":
    path_processed_meta = "../data/all_properties.csv"
    path_session = "../data/train.csv"
    path_item = "../data/item_metadata.csv"

    to_path_item = "../data/FM_item_vectors.csv"
    to_path_session = "../data/FM_session_vectors.csv"

    # process_item_data(path_item, to_path_item, path_processed_meta)
    process_session_data(path_session, to_path_session, to_path_item, path_processed_meta)

