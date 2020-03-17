

from __future__ import absolute_import, division, print_function


import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# ### Plot graphs inline
get_ipython().magic(u'matplotlib inline')

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


print(tf.__version__)
print(np.__version__)
print(matplotlib.__version__)

# Import stock data
dataset = pd.read_csv('./stock_data/{}.csv'.format('AAPL'))

data = dataset[['Adj Close', 'Volume', 'Open']]
# To have everything one time point behind the price

n = len(data)
train_start = 0 
train_end = int(np.floor(0.8*n))
test_start = train_end
test_end = n

t_plus_one = data['Adj Close'].drop(0).append(pd.Series(99999), ignore_index = True)
y_actual = np.where(t_plus_one >= data['Adj Close'], 1, 0)

data_train = data[train_start:train_end]
data_test = data[test_start:test_end]
data_train = np.array(data_train)
data_test = np.array(data_test)
#data_train = np.array(data_train).reshape(-1,1)
#data_test = np.array(data_test).reshape(-1,1)

training_labels = y_actual[np.arange(train_start,train_end)]
test_labels = y_actual[np.arange(test_start,test_end)]

# ### Placeholders for training and test data
# 
# * The training dataset placeholder can have any number of instances and each instance is an array of 28x28 = 784 pixels
# * The test placeholder will hold just a single digit image at a time
training_digits_pl = tf.placeholder("float", [None, 3])
test_digit_pl = tf.placeholder("float", [3])

# ### Get the label which occurs the most often in our predicted set
# 
# * *labels*: The labels associated with the entire training dataset
# * *indices*: The indices of those images which are the closest to the test image
# * Returns the labels which has been predicted the most often

def get_majority_predicted_label(labels, indices):
    predicted_labels = []
    for i in indices:
        predicted_labels.append(labels[i])
        
    predicted_labels = np.array(predicted_labels)
    
    print(predicted_labels)
    
    # Place the labels into bins and find the bin with the highest number of labels
    counts = np.bincount(predicted_labels)
    
    return np.argmax(counts)

# ### Nearest neighbor calculation using L1 distance
# 
# * Calculate the **absolute value** of the distance between the test image and the training set
# * Calculate the **sum** of the distance between the test image and all training images
# * Find the images in the training data that are the k closest neigbors
# * *top_k* finds the highest values, apply it to the negative of the distances
l1_distance = tf.abs(tf.subtract(training_digits_pl, test_digit_pl))

distance_l1 = tf.reduce_sum(l1_distance, axis=1)

pred_knn_l1 = tf.nn.top_k(tf.negative(distance_l1), k=3)

# ### Nearest neighbor calculation using L2 (Euclidean) distance
# 
# * Calculate the **square** of the distance between the test image and the training set
# * Calculate the **square root of the sum of squares*** of the distance between the test image and all training images
# * Find the images in the training data that are the k closest neigbors
# * *top_k* finds the highest values, apply it to the negative of the distances

# Nearest Neighbor calculation using L2 distance
l2_distance = tf.square(tf.subtract(training_digits_pl, test_digit_pl))

distance_l2 = tf.sqrt(tf.reduce_sum(l2_distance, axis=1))

pred_knn_l2 = tf.nn.top_k(tf.negative(distance_l2), k=3)

accuracy = 0.

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # loop over test data
    for i in range(len(data_test)):
        # Get nearest neighbor
        _, indices = sess.run(pred_knn_l1, feed_dict={training_digits_pl: data_train, test_digit_pl: data_test[i, :]})
    
        predicted_label = get_majority_predicted_label(training_labels, indices)
        # Get nearest neighbor class label and compare it to its true label
        print("Test", i, "Prediction:", predicted_label,             "True Label:", test_labels[i])

        # Calculate accuracy
        if predicted_label == test_labels[i]:
            accuracy += 1./len(data_test)

    print("Done!")
    print("Accuracy:", accuracy)
