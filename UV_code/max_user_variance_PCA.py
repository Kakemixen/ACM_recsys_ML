import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from random import random, seed
from math import sqrt
### get data
import headers


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

def get_filter(num_indexes, component_vectors, component_scalars):

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
    return bool_filter

def variance_of_num_index(num_indexes, component_vectors, component_scalars):

    bool_filter = get_filter(num_indexes, component_vectors, component_scalars)
    # filtered = user_vectors * max_indexes
    filtered = user_vectors[:,bool_filter]

    filtered_variance = calculate_variance(filtered)
    return filtered_variance

def rank_indexes(component_vectors, component_scalars):

    # get absolute value for explained variance for each index
    component_vectors = (component_vectors.T * component_scalars).T
    component_vectors = np.array([[abs(x) for x in X] for X in component_vectors])

    # aggregate by making over axis=1
    max_components = component_vectors.max(0) #

    #get headers
    header = headers.get_item_header() + ["price"]

    ranking = [(header[i], max_components[i]) for i in range(len(max_components))]
    ranking.sort(key=lambda x: x[1], reverse=True)

    # ranking = dict()
    # for i in range(len(header)):
    #     ranking[header[i]] = max_components[i]

    return ranking




### parameters
num_indexes = 158 #mac number, do not change
index_list = np.arange(1,num_indexes)
variances = [-1 for _ in index_list]

## PCA
pca = PCA()

# fit the PCA
pca.fit(user_vectors)
component_vectors = pca.components_ # private variable but it's what we need
component_scalars = pca.explained_variance_

#rank indexes
print("ranking")
ranking = rank_indexes(component_vectors, component_scalars)
labels = [r[0] for r in ranking]
values = [r[1] for r in ranking]

fig, ax = plt.subplots()
ax.barh(range(0,50), values[0:50], align='center')
ax.set_yticks(range(0,50))
ax.set_yticklabels(labels[0:50], size=5.5)
ax.invert_yaxis()
ax.set_xlabel("influence")
plt.subplots_adjust(left=0.3)
plt.savefig("./ranking_1.png", dpi=500)

plt.clf()
fig, ax = plt.subplots()
ax.barh(range(50,100), values[50:100], align='center')
ax.set_yticks(range(50,100))
ax.set_yticklabels(labels[50:100], size=5.5)
ax.invert_yaxis()
ax.set_xlabel("influence")
plt.subplots_adjust(left=0.3)
plt.savefig("./ranking_2.png", dpi=500)

plt.clf()
fig, ax = plt.subplots()
ax.barh(range(100, len(ranking)), values[100:], align='center')
ax.set_yticks(range(100, len(ranking)))
ax.set_yticklabels(labels[100:], size=4.5)
ax.invert_yaxis()
ax.set_xlabel("influence")
plt.subplots_adjust(left=0.3)
plt.savefig("./ranking_3.png", dpi=500)

# get ratio of explained variance pr index
for x in index_list:
    print(x,end="\r")
    variances[x-1] =  variance_of_num_index(x, component_vectors, component_scalars) / full_variance

variance_gain = [variances[i]-(i/157) for i in range(len(variances))]

# Change the color and its transparency
plt.clf()
plt.rcParams['axes.facecolor'] = 'salmon'
plt.margins(0)
plt.fill_between( index_list, variances, color="deepskyblue", alpha=1)
plt.plot( index_list, variances, color="blue", alpha=1)
# plt.plot( (1,157), (0,1), color="black", alpha=1)
plt.fill_between( index_list, variances, color="deepskyblue", alpha=1)
plt.title("Ratio of variance kept")
plt.xlabel("number of indexes of 157")
plt.ylabel("ratio of variance")
plt.savefig("./variance_ratio.png")

plt.clf()
plt.margins(0)
plt.fill_between( index_list, variance_gain, color="deepskyblue", alpha=1)
plt.plot( index_list, variance_gain, color="blue", alpha=1)
plt.title("variance gain")
plt.xlabel("number of indexes of 157")
plt.ylabel("ratio of variance")
plt.savefig("./variance_gain.png")

# write a file with
