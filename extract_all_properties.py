import pandas as pd

metadata_path = "./data/item_metadata.csv"

metadata = pd.read_csv(metadata_path)

all_properties = dict()

for index, item in metadata.iterrows():
    properties = item["properties"].split("|")
    for x in properties:
        if not(x in all_properties.keys()):
            all_properties[x] = 1
        else:
            all_properties[x] += 1

df = pd.DataFrame([all_properties.keys(),all_properties.values()]).transpose()
df.to_csv("./data/all_properties.csv", index=False)

