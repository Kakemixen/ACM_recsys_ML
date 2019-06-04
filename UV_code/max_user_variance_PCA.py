import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from random import random, seed
from math import sqrt
### get data

seed(211196) #Oh, such eggs of easter

# user_vectors = np.loadtxt('./user_vectors.csv', delimiter=',', skiprows=1)
user_vectors = np.array([
    [random()*(j+1)/200 if i > 120 else 0 for i in range(140)]
    for j in range(200)])
# print(user_vectors[0])



### parameters
num_indexes = 5


## PCA
pca = PCA(n_components=5)
# fit the PCA
pca.fit(user_vectors)
component_vectors = pca.components_ # private variable but it's what we need
component_scalars = pca.explained_variance_
print("components")
# print(component_vectors[0])
# print(component_scalars)

# TODO multiply row vector by ther respective explained variance
component_vectors = (component_vectors.T * component_scalars).T
# print(component_vectors)

# print("extended components")
component_vectors = np.array([[abs(x) for x in X] for X in component_vectors])

# decoded_components = (decoded_components.T * component_scalars).T

# TODO aggregate by making over axis=1
max_components = component_vectors.max(0) #
# TODO find max indexes in that vector
max_indexes = np.zeros(len(max_components),dtype = np.int)
for i in range(num_indexes):
    index = max_components.argmax()
    max_indexes[index] = 1
    max_components[index] = 0

print("most important indexes, or indexes with most variance")
print(max_indexes)


