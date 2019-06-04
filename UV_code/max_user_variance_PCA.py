import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from random import random, seed
from math import sqrt
### get data


def calculate_variance(matrix):
    tmp = np.matmul(matrix.T, matrix)
    return np.trace(tmp)


seed(211196) #Oh, such eggs of easter

user_vectors = np.loadtxt('./user_vectors_noid.csv', delimiter=',', skiprows=1)
# user_vectors = np.array([
#     [random()*(j+1)/200 if i > 120 else 0 for i in range(140)]
#     for j in range(200)])
# print(user_vectors[0])

full_variance = calculate_variance(user_vectors)


def variance_or_num_index(num_indexes):
    ## PCA
    pca = PCA()
    # fit the PCA
    pca.fit(user_vectors)
    component_vectors = pca.components_ # private variable but it's what we need
    component_scalars = pca.explained_variance_

    # multiply row vector by ther respective explained variance
    component_vectors = (component_vectors.T * component_scalars).T

    component_vectors = np.array([[abs(x) for x in X] for X in component_vectors])

    # aggregate by making over axis=1
    max_components = component_vectors.max(0) #

    # find indexes that has highest variance in user data
    max_indexes = np.zeros(len(max_components),dtype = np.int)
    for i in range(num_indexes):
        index = max_components.argmax()
        max_indexes[index] = 1
        max_components[index] = 0

    bool_filter = [True if x > 0 else False for x in max_indexes]
    # filtered = user_vectors * max_indexes
    filtered = user_vectors[:,bool_filter]

    filtered_variance = calculate_variance(filtered)
    return filtered_variance

### parameters
num_indexes = 158
index_list = np.arange(1,num_indexes)
variances = [-1 for _ in index_list]
for x in index_list:
    print(x,end="\r")
    variances[x-1] =  variance_or_num_index(x) / full_variance

# Change the color and its transparency
plt.rcParams['axes.facecolor'] = 'salmon'
plt.margins(0)
plt.fill_between( index_list, variances, color="deepskyblue", alpha=1)
plt.plot( index_list, variances, color="blue", alpha=1)
plt.title("Ratio of variance kept")
plt.xlabel("number of indexes of 157")
plt.ylabel("ratio of variance")
plt.show()
plt.savefig("./variance_ratio.png")



