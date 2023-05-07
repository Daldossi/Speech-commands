# -*- coding: utf-8 -*-
"""
@author: alice
Script: Steps:
    1. visualization of some of the spectrograms with and whithout normalization;
    2. training of a multilayer perceptron without hidden layers using batch 
    gradient descent and then stochastic gradient descent with minibatches of 
    different size;
    3. save the data for later use.
"""

import pvml
import numpy as np
import matplotlib.pyplot as plt


## FUNCTIONS FOR NORMALIZATION

def minmax_normalization(Xtrain, Xtest, Xval): 
    """ Min-max scaling """
    xmin = Xtrain.min(0) 
    xmax = Xtrain.max(0) 
    Xtrain = (Xtrain - xmin) / (xmax - xmin) 
    Xtest = (Xtest - xmin) / (xmax - xmin) 
    Xval = (Xval - xmin) / (xmax - xmin) 
    return Xtrain, Xtest, Xval 

def meanvar_normalization(Xtrain, Xtest, Xval): 
    """ Mean-var scaling """
    mu = Xtrain.mean(0) 
    sigma = Xtrain.std(0) 
    Xtrain = (Xtrain - mu) / sigma 
    Xtest = (Xtest - mu) / sigma 
    Xval = (Xval - mu) / sigma 
    return Xtrain, Xtest, Xval 

def maxabs_normalization(Xtrain, Xtest, Xval): 
    """ Max-abs scaling """
    amax = np.abs(Xtrain).max(0) 
    Xtrain = Xtrain / amax 
    Xtest = Xtest / amax 
    Xval = Xval / amax 
    return Xtrain, Xtest, Xval

def whitening(Xtrain, Xtest, Xval):
    """ Whitening normalization """
    mu = Xtrain.mean(0)
    sigma = np.cov(Xtrain.T) 
    evals, evecs = np.linalg.eigh(sigma) 
    w = evecs / np.sqrt(evals) 
    Xtrain = (Xtrain - mu) @ w 
    Xtest = (Xtest - mu) @ w 
    Xval = (Xval - mu) @ w 
    return Xtrain, Xtest, Xval

def instance(X, ln):
    """ Instance normalization with both l2 and l1 """
    if ln =="l1":
        q = np.abs(X).sum(1, keepdims=True)
    elif ln == "l2":
        q = np.sqrt((X ** 2).sum(1, keepdims=True)) 
    q = np.maximum(q, 1e-15) 
    X = X / q 
    return X 


## OTHER FUNCTIONS

def draw_spect(spectogram, title): 
    """ Plot the spectogram with a certain title """
    plt.imshow(spectogram.reshape(20, 80)) 
    plt.colorbar() 
    plt.title(title) 
    plt.show() 

def accuracy(network, X, Y):
    """ Calculate the accuracy of X, calling Y the true values """
    predictions, logits = network.inference(X)
    accuracy = (predictions == Y).mean()
    return accuracy



###############################################################################
# MAIN FUNCTION 
###############################################################################

if __name__ == "__main__": 
    
    ## Load the data
    words = open("classes.txt").read().split() 
    data = np.load("train.npz") 
    Xtrain = data["arr_0"] 
    Ytrain = data["arr_1"] 
    data = np.load("test.npz") 
    Xtest = data["arr_0"] 
    Ytest = data["arr_1"]
    data = np.load("validation.npz") 
    Xval = data["arr_0"] 
    Yval = data["arr_1"]
    
    
    ## 1. VISUALIZATION
    
    ## Check the mean and the std behaviour
    mu = Xtrain.mean(0) 
    draw_spect(mu, "Mean behaviour") 
    std = Xtrain.std(0) 
    draw_spect(std, "Std behaviour") 
    
    ## Perform normalization
    # Xtrain_mm, Xtest_mm, Xval_mm = minmax_normalization(Xtrain, Xtest, Xval) 
    Xtrain_mv, Xtest_mv, Xval_mv = meanvar_normalization(Xtrain, Xtest, Xval) 
    # Xtrain_abs, Xtest_abs, Xval_abs = maxabs_normalization(Xtrain, Xtest, Xval) 
    # Xtrain_w, Xtest_w, Xval_w = whitening(Xtrain, Xtest, Xval) 
    # Xtrain_i = instance(Xtrain, "l2")
    # Xtest_i = instance(Xtest, "l2")
    # Xval_i = instance(Xval, "l2")
    
    ## Draw the data
    # draw_spect(Xtrain[0, :], "No normalization") 
    # draw_spect(Xtrain_mm[0, :], "MinMax normalization") 
    draw_spect(Xtrain_mv[0, :], "MeanVar normalization") 
    # draw_spect(Xtrain_abs[0, :], "MaxAbs normalization") 
    # draw_spect(Xtrain_w[0, :], "Whitening") 
    # draw_spect(Xtrain_i[0, :], "Instance normalization") 
    
    
    ## 2. TRAINING THE MULTILAYER PERCEPTRON
    
    ## - size of the first layer = 1600: number of input neurons (1 for each component in the spectogram)
    ## - size of the last layer = 35: number of classes
    ARC = [1600, 35] 
    # ARC = [1600, 600, 35]
    # ARC = [1600, 600, 280, 35]
    # ARC = [1600, 600, 280, 100, 35]
    # ARC = [1600, 900, 600, 300, 35]
    
    network = pvml.MLP(ARC) 
    R = 50 # number of epoches
    Xtrain = Xtrain_mv 
    Xtest = Xtest_mv 
    Xval = Xval_mv 
    m = Xtrain.shape[0] 
    
    for epoch in range(R+1): 
        # network.train(Xtrain, Ytrain, lr=1e-4, steps=1, batch=m) 
        # network.train(Xtrain, Ytrain, lr=1e-4, steps=m//20, batch=1) 
        # network.train(Xtrain, Ytrain, lr=1e-4, steps=m//20, batch=30) 
        network.train(Xtrain, Ytrain, lr=1e-4, steps=m//20, batch=60) 
        if epoch % 5 == 0: 
            train_acc = accuracy(network, Xtrain, Ytrain) 
            test_acc = accuracy(network, Xtest, Ytest) 
            val_acc = accuracy(network, Xval, Yval) 
            print(f"Accuracy of epoch {epoch} - train: {train_acc:.3f}, test {test_acc:.3f}, validation {val_acc:.3f}") 
    
    
    ## 3. REMEMBER
    
    network.save("mlp.npz")
    

    
    