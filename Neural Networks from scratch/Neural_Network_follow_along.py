#!/usr/bin/env python3
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import csv
from mpl_toolkits.mplot3d import Axes3D

wd = os.getcwd()
data = pd.read_csv('/Users/adamkurth/Documents/vscode/Python/Neural Networks from scratch/digit-recognizer/train.csv')

#make pd data frame into a np.array
data = np.array(data)
m,n = data.shape
np.random.shuffle(data) #shuffle data before splitting into dev and train

# test preformance on this data to compare, and cross validate.
data_dev = data[0:1000].T  #turn into column matrix
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev/255.

# partition of the data which we need to feed to the network.
data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

# initialize parameters 
def init_parameters():
    #update the normalization you get a performance of 0.90
    W1 = np.random.normal(size=(10, 784)) * np.sqrt(1./(784))
    b1 = np.random.normal(size=(10, 1)) * np.sqrt(1./10)
    W2 = np.random.normal(size=(10, 10)) * np.sqrt(1./20)
    b2 = np.random.normal(size=(10, 1)) * np.sqrt(1./(784))
    return W1, b1, W2, b2

def ReLU(Z): 
    # returns 0 if Z<0, and Z if >= 0
    return np.maximum(0,Z)

# applies exp to each value of Z 
# sum refers to summing up each exp(Z), in doing so this 
# preserves the number of columns and collapses the amount of rows.
# this gives us our probability for each column.
def softmax(Z):
    Z -= np.max(Z, axis=0)  # Subtract max value for numerical stability
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A
    
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(Y): 
    # matrix of Y.size = m and Y.max()+1=10
    # recreates the correctly sized matrix 
    
    # goes np.arrange() creates an array of 0 to m training examples, specifying which row/label is being accessed.
    # Y gets the actual column of this array, setting this row,col to 1
    
    # flip matrix so that each column is an example
    
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_ReLU(Z):
    # boolean: T=>1, F=>0 
    # so if 1 element in Z is greater than 0 we return a 1
    return Z > 0

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha*dW1
    b1 =  b1 - alpha*db1
    W2 = W2 - alpha*dW2
    b2 = b2 - alpha*db2    
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def get_percentage(predictions, Y):
    # print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations, data_array):
    W1, b1, W2, b2 = init_parameters()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X) #weighted sums at each layer and activation (outputs) at each layer.
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y) # loss function
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        
        mean_W1 = np.mean(W1); mean_W2 = np.mean(W2)
        mean_b1 = np.mean(b1); mean_b2 = np.mean(b2)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
        data_array.append([mean_b1, mean_b2, mean_W1, mean_W2, get_percentage(predictions, Y), i])
    return mean_b1, mean_b2, mean_W1, mean_W2, data_array

global data_array
data_array = []
W1, b1, W2, b2, data_array = gradient_descent(X_train, Y_train, 0.10, 500, data_array)

with open('output.csv', 'w', newline='') as f:
    write = csv.writer(f)
    for i in data_array: 
        flatten = []
        for comp in i: 
            if isinstance(comp, np.ndarray):
                flatten.append(comp.flatten())
            else: 
                flatten.append(comp)
        write.writerow(flatten)

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)
test_prediction(3, W1, b1, W2, b2)

dev_predictions = make_predictions(X_dev, W1, b1, W2, b2) 
get_accuracy(dev_predictions, Y_dev)
