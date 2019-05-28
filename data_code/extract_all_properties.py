import pandas as pd

def extract_all_properties(path, to_path):
    metadata = pd.read_csv(path)

    all_properties = dict()

    for index, item in metadata.iterrows():
        properties = item["properties"].split("|")
        for x in properties:
            if not(x in all_properties.keys()):
                all_properties[x] = 1
            else:
                all_properties[x] += 1

    df = pd.DataFrame([all_properties.keys(),all_properties.values()]).transpose()
    df.to_csv(to_path, index=False)

if __name__ == "__main__":
    metadata_path = "../data/item_metadata.csv"
    to_path = "../data/all_properties.csv"
    extract_all_properties(metadata_path, to_path)
