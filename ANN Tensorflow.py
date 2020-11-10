# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 10:26:13 2020

@author: callum
"""

import numpy as np
#import tensorflow as tf
import logging
logging.basicConfig(level=logging.DEBUG)
import matplotlib.pyplot as plt
import DataLoader
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#Network parameters
n_hidden1 = 100
n_hidden2 = 10
n_input = 6
n_output = 4
#Learning parameters
learning_constant = 0.001
number_epochs = 1000000
#1,000,000,000
batch_size = 1728

#Defining the input and the output
X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_output])
#DEFINING WEIGHTS AND BIASES
#Biases first hidden layer
b1 = tf.Variable(tf.random_normal([n_hidden1]))
#Biases second hidden layer
b2 = tf.Variable(tf.random_normal([n_hidden2]))
#Biases output layer
b3 = tf.Variable(tf.random_normal([n_output]))
#Weights connecting input layer with first hidden layer
w1 = tf.Variable(tf.random_normal([n_input, n_hidden1]))
#Weights connecting first hidden layer with second hidden layer
w2 = tf.Variable(tf.random_normal([n_hidden1, n_hidden2]))
#Weights connecting second hidden layer with output layer
w3 = tf.Variable(tf.random_normal([n_hidden2, n_output]))

batch_x1 = [] # buying feature
batch_x2 = [] # maint feature
batch_x3 = [] # doors feature
batch_x4 = [] # persons feature
batch_x5 = [] # lug boot feature
batch_x6 = [] # safety feature

output_y = [] # All outputs

def multilayer_perceptron(input_d):
    #Task of neurons of first hidden layer
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(input_d, w1), b1))
    #Task of neurons of second hidden layer
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, w2), b2))
    #Task of neurons of output layer
    out_layer = tf.add(tf.matmul(layer_2, w3),b3)
    return out_layer

#Create model
neural_network = multilayer_perceptron(X)
#Define loss and optimizer
#loss_op = tf.reduce_mean(tf.math.squared_difference(neural_network,Y))
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=neural_network,labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_constant).minimize(loss_op)

#Initializing the variables
init = tf.global_variables_initializer()

DataLoader.load_dataset("car.data.txt", batch_x1, batch_x2, batch_x3, batch_x4, batch_x5, batch_x6, output_y)

multi_output_y = list()

for data in output_y:
    temp = np.zeros(4) # Create empty array of 4 [0, 0, 0, 0]
    temp[data] = 1 # Set the index of the datapoint to true (1) e.g. [0, 0, 0, 1]
    multi_output_y.append(temp) # Add it to the list of multi outputs
    
multi_output_y = np.array(multi_output_y) # Convert list to numpy array

dataset = DataLoader.split_testing_training(0.4, batch_x1, batch_x2, batch_x3, batch_x4, batch_x5, batch_x6, output_y, multi_output_y)  

label_y=multi_output_y
batch_x=np.column_stack((batch_x1, batch_x2, batch_x3, batch_x4, batch_x5, batch_x6))
batch_y=multi_output_y
batch_x_train=np.row_stack(dataset[2])
batch_y_train=np.row_stack(dataset[3])
batch_x_test=np.row_stack(dataset[0])
batch_y_test=np.row_stack(dataset[1])

with tf.Session() as sess:
    sess.run(init)
    pred = (neural_network) # Apply softmax to logits
    accuracy=tf.keras.losses.MSE(pred,Y)
    #Training epoch
    for epoch in range(number_epochs):

            sess.run(optimizer, feed_dict={X: batch_x_train, Y: batch_y_train})
            #Display the epoch
            if epoch % 100 == 0:
                correct_prediction1 = tf.equal(tf.argmax(pred, 1),output_y)
                accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))
                acc = accuracy1.eval({X: batch_x, Y: batch_y_train})
                print("Epoch: %d\tAccuracy: %f%%" % (epoch, acc * 100))

    # Test model
    print("Accuracy:", accuracy.eval({X: batch_x_test, Y: batch_y_test}), file=open("Accuracy.txt", "a"))

    print("Prediction:", pred.eval({X: batch_x_test}), file=open("Prediction.txt", "a"))
    output=tf.argmax(neural_network.eval({X: batch_x_test}),1)
    output = output.eval({X: batch_x, Y: batch_y_train})
    plt.yticks(range(0, 4))
    plt.plot(batch_y_test, 'rs', markersize=3.0)
    plt.plot(output, 'bo', markersize=1.0)
    plt.ylabel('Class')
    plt.show()
    
    correct_prediction1 = tf.equal(tf.argmax(pred, 1),output_y)
    accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))
    print(accuracy1.eval({X: batch_x}))