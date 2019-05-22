import numpy as np
import pandas as pd

path_meta = "../data/all_properties.csv"
path_session = "../data/train.csv"
path_item = "../data/item_metadata.csv"

to_path_item = "../data/FM_item_vectors.csv"
to_path_session = "../data/FM_session_vectors.csv"


def get_metadata_vector(path):
    if path == None:
        #TODO import and use extract_all_properties.py
        print("need metadata")
        print("extracting")
        import extract_all_properties
        print("done extracting, run program again")
        return

    df = pd.read_csv(path)
    return df["0"]

def process_item_data(path, to_path, processed_metadata_path=None):
    one_hot_attributes = np.concatenate((np.array(["item"]), get_metadata_vector(processed_metadata_path)), axis=0) # one hot encoding header list
    n_attr = len(one_hot_attributes)

    empty_item = np.array([[0 for _ in range(len(one_hot_attributes))]]) #so that it may be added to other np.array([[]])

    df = pd.DataFrame([], columns=one_hot_attributes)
    df.to_csv(to_path, mode="w+", header=True)

    for chunk in pd.read_csv(path, chunksize=512): # split up into chunks cus memory error
        first_in_chunk=True
        # encoded_items=np.array(empty_item) # because we need this apparently
        indexes = np.array([], dtype=np.int64)
        for index, item in chunk.iterrows():
            encoded_item = empty_item.copy()
            encoded_item[0][0] = item["item_id"]
            for i in range(1, n_attr): # to not count item ID
                for item_attribute in item["properties"].split("|"):
                    if(item_attribute == one_hot_attributes[i]):
                        encoded_item[0][i] = 1
            if first_in_chunk:
                encoded_items = np.array(encoded_item) #start the list
                first_in_chunk = False
            else:
                encoded_items = np.append(encoded_items,encoded_item,0)
            print(index)
            indexes = np.append(indexes, int(index)) #TODO change to item ID's
        df = pd.DataFrame(encoded_items, columns=one_hot_attributes, index=indexes)
        df.to_csv(to_path, mode='a', header=False)
        if(index > 1000): break #TODO remove to transofrm entire csv

process_item_data(path_item, to_path_item, path_meta)




def process_session_data():
    for chunk in pd.read_csv(path, chunksize=512): # split up into chunks cus memory error
        for index, item in chunk.iterrows():
            pass
    df = pd.DataFrame(list(possibilities)).transpose()
    df.to_csv(to_path, mode='a+', header=False)
