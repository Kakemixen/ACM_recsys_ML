from pathlib import Path
import numpy as np
import pandas as pd

path_processed_meta = "../data/all_properties.csv"
path_session = "../data/train.csv"
path_item = "../data/item_metadata.csv"

to_path_item = "../data/FM_item_vectors.csv"
to_path_session = "../data/FM_session_vectors.csv"

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
#   None
def process_item_data(path, to_path, processed_metadata_path):
    print("This will take some time. Progress(will still keep up to last batch if stopped early):")

    one_hot_attributes = get_metadata_vector(processed_metadata_path) # one hot encoding header list
    n_attr = len(one_hot_attributes)

    empty_item = np.array([[0 for _ in range(len(one_hot_attributes))]]) # baseline used to start a new item

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
            print("i:{:<7} t:{}   ".format(index, 927143), end="\r")

        #writing the data as processed under the header already written
        df = pd.DataFrame(encoded_items, columns=one_hot_attributes, index=indexes)
        df.to_csv(to_path, mode='a', header=False)
    print("\nDone!")
    print("File can be found at: {}".format(to_path))



def process_session_data():
    for chunk in pd.read_csv(path, chunksize=512): # split up into chunks cus memory error
        for index, item in chunk.iterrows():
            pass
    df = pd.DataFrame(list(possibilities)).transpose()
    df.to_csv(to_path, mode='a+', header=False)

if __name__ == "__main__":
    process_item_data(path_item, to_path_item, path_processed_meta)

