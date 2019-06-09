import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from random import random, seed
from math import sqrt
### get data

def calculate_variance(matrix):
    tmp = np.matmul(matrix.T, matrix)
    return np.trace(tmp)

seed(211196) #Oh, such eggs of easter

user_vectors = np.loadtxt('./user_vectors_noid.csv', delimiter=',', skiprows=1)
full_variance = calculate_variance(user_vectors)

### parameters
num_indexes = 20

# defining batch size, number of epochs and learning rate
batch_size = 30  # how many images to use together for training
hm_epochs = 1000    # how many times to go through the entire dataset
tot_sequences = len(user_vectors) # total number of images

# print(int(tot_sequences/batch_size))

## AA
data_dimensions= len(user_vectors[0]) # = 158 or something
### layer stuffs
## num nodes
#in
nodes_input = data_dimensions
nodes_hl_1 = int(data_dimensions/1.5)
nodes_hl_2 = int(data_dimensions/2)        #out
nodes_hl_3 = int(data_dimensions/1.5)
# nodes_hl_4 = 75
nodes_output = data_dimensions


## Layering
hidden_1_layer_vals = {
    'weights':tf.Variable(tf.random_normal([nodes_input, nodes_hl_1])),
    'biases':tf.Variable(tf.random_normal([nodes_hl_1]))
}
hidden_2_layer_vals = {
    'weights':tf.Variable(tf.random_normal([nodes_hl_1, nodes_hl_2])),
    'biases':tf.Variable(tf.random_normal([nodes_hl_2]))
}
hidden_3_layer_vals = {
    'weights':tf.Variable(tf.random_normal([nodes_hl_2, nodes_hl_3])),
    'biases':tf.Variable(tf.random_normal([nodes_hl_3]))
}
# hidden_4_layer_vals = {
#     'weights':tf.Variable(tf.random_normal([nodes_hl_3, nodes_hl_4])),
#     'biases':tf.Variable(tf.random_normal([nodes_hl_4]))
# }
output_layer_vals = {
    'weights':tf.Variable(tf.random_normal([nodes_hl_3, nodes_output])),
    'biases':tf.Variable(tf.random_normal([nodes_output]))
}


## Networking
# image with shape #K-mers goes in
input_layer = tf.placeholder('float', [None, data_dimensions])

# multiply output of respective layer wth a weight matrix and add biases
layer_1 = tf.nn.sigmoid(
       tf.add(tf.matmul(input_layer,hidden_1_layer_vals['weights']),
       hidden_1_layer_vals['biases']))
layer_2 = tf.nn.sigmoid(
       tf.add(tf.matmul(layer_1,hidden_2_layer_vals['weights']),
       hidden_2_layer_vals['biases']))
layer_3 = tf.nn.sigmoid(
       tf.add(tf.matmul(layer_2,hidden_3_layer_vals['weights']),
       hidden_3_layer_vals['biases']))
# layer_4 = tf.nn.sigmoid(
#        tf.add(tf.matmul(layer_3,hidden_4_layer_vals['weights']),
#        hidden_4_layer_vals['biases']))
output_layer = tf.add(tf.matmul(layer_3,output_layer_vals['weights']),
               output_layer_vals['biases'])

# output_true shall have the original image for error calculations
output_true = tf.placeholder('float', [None, data_dimensions])
# define our cost function
meansq = tf.reduce_mean(tf.square(output_layer - output_true))
# define our optimizer
learn_rate = 0.1   # how fast the model should learn
optimizer = tf.train.AdagradOptimizer(learn_rate).minimize(meansq)

### let's run this shit! cus lief amazing and things

# initialising stuff and starting the session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


losses = [] # if want to plot loss
# total improvement is printed out after each epoch
for epoch in range(hm_epochs):
    epoch_loss = 0    # initializing error as 0
    for i in range(int(tot_sequences/batch_size)):
        epoch_x = user_vectors[ i*batch_size : (i+1)*batch_size ]
        _, c = sess.run([optimizer, meansq],\
               feed_dict={input_layer: epoch_x, \
               output_true: epoch_x})
        epoch_loss += c
    losses.append(epoch_loss)

    print('Epoch', epoch+1, '/', hm_epochs, 'loss:',epoch_loss, end="\r")
print()


### getting encoded

encoded_vectors = np.array(sess.run(
            layer_2,
            feed_dict={input_layer:user_vectors}
        )
    )

decoded_vectors = np.array(sess.run(
        output_layer,
        feed_dict={input_layer:user_vectors}
        ))
print(decoded_vectors[0][110:])
print(user_vectors[0][110:])

# encoded_vectors = [ x[0] for x in encoded_vectors ] #cus appearently one have to do this

## write encoded user_vectors to some file
df = pd.DataFrame(encoded_vectors)
df.to_csv("./reduced_user_vectors.csv".format(2), mode="w+", index=False)
print("saved the encoded user_vectors to: data/encoded_sequences_K={}.csv".format(2))


## PCA
pca = PCA()
# fit the PCA
pca.fit(encoded_vectors)
component_vectors = pca.components_ # private variable but it's what we need
component_scalars = pca.explained_variance_
print("components")
# print(component_vectors[0])
# print(component_scalars)

# TODO multiply row vector by ther respective explained variance
component_vectors = (component_vectors.T * component_scalars).T
# print(component_vectors)


decoded_components = sess.run(
            output_layer,
            feed_dict={layer_2:component_vectors}
        )


# print("extended components")
decoded_components = np.array([[abs(x) for x in X] for X in decoded_components])

# decoded_components = (decoded_components.T * component_scalars).T

# TODO aggregate by making over axis=1
max_components = decoded_components.max(0) #
# TODO find max indexes in that vector
max_indexes = np.zeros(len(max_components),dtype = np.int)
for i in range(num_indexes):
    index = max_components.argmax()
    max_indexes[index] = 1
    max_components[index] = 0

print("most important indexes, or indexes with most variance")
print(max_indexes)


bool_filter = [True if x > 0 else False for x in max_indexes]
# filtered = user_vectors * max_indexes
filtered = user_vectors[:,bool_filter]
print(filtered[19])

filtered_variance = calculate_variance(filtered)
print(full_variance)
print(filtered_variance)
print( filtered_variance / full_variance )
