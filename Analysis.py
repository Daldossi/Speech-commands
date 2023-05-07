# -*- coding: utf-8 -*-
"""
@author: alice
Script: Analyze the confusion matrix and look at an example of misclassification
"""

from Training import *
import pvml
import numpy as np
import matplotlib.pyplot as plt
import random
from playsound import playsound


def show_weights(network): 
    w = network.weights[0] 
    maxval = np.abs(w).max() 
    for klass in range(35):
        plt.subplot(5, 7, klass+1)
        plt.imshow(w[:, klass].reshape(20, 80), cmap = "seismic", vmin=-maxval, vmax=maxval) 
        # vmin and vmax change the colours to red and blue
        plt.title(words[klass])  
    # plt.figure(figsize=(30,30))
    plt.savefig('weights.jpg', dpi=300)
    plt.show() 

def make_confusion_matrix(predictions, labels):
    """ build the confusion matrix """
    cmat = np.zeros((35, 35))
    errs = [] 
    for i in range(predictions.size): # for each pair of prediction in the true label
        cmat[labels[i], predictions[i]] += 1 
        if predictions[i] != labels[i]: 
            errs.append([i, predictions[i], labels[i]]) 
    s = cmat.sum(1, keepdims=True)
    cmat /= s
    return cmat, errs

def dispaly_confusion_matrix(cmat): 
    """ display a confusion matrix cmat such that: 
        - the columns represent the predictions
        - the rows represent the correct labels """
    print(" " * 10, end="") 
    for j in range(35): 
        print(f"{words[j][:4]:4} ", end="")  
    print() 
    for i in range(35): 
        print(f"{words[i]:5}", end="") 
        for j in range(35): 
            val = cmat[i, j] * 100 
            print(f"{val:4.1f} ", end="") 
        print() 

def dispaly_confusion_matrix2(cmat): 
    """ display the image of the confusion matrix cmat such that: 
        - the columns represent the predictions
        - the rows represent the correct labels """
    plt.imshow(cmat, cmap="Blues") 
    for i in range(35): 
        # print(f"{words[i]:5}", end="") 
        for j in range(35): 
            val = cmat[i, j] * 100 
            # print(f"{val:4.1f} ", end="")
            plt.text(j-0.25, i+0.1, int(val)) 
    plt.title("Confusion matrix") 
    plt.show() 
    print(" " * 10, end="") 
    for j in range(35): 
        print(f"{words[j]:10} ", end="") 
    print() 
    # plt.savefig("matrix.pdf") 
    


## 1. IMPORT THE DATA 
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

## 2. NORMALIZATION
Xtrain, Xtest, Xval = meanvar_normalization(Xtrain, Xtest, Xval) 
# mu = Xtrain.mean(0) 
# std = Xtrain.std(0) 
# Xtrain = (Xtrain - mu) / std 
# Xtest = (Xtest - mu) / std 

## 3. IMPORT THE NETWORK
network = pvml.MLP.load("mlp.npz") 
show_weights(network) 

## 4. FIND THE CONFUSION MATRIX
predictions, logit = network.inference(Xtest) 
cmat, E = make_confusion_matrix(predictions, Ytest) 
dispaly_confusion_matrix(cmat) 
# print('indici di valore massimo:', np.argwhere(cmat == cmat.max()))
# print('parola più corretta:', words[25]) # words[25]='six' è la parola che viene predetta meglio
## six=[0.        , 0.0058651 , 0.00879765, 0.        , 0.00293255,
       # 0.        , 0.09384164, 0.        , 0.        , 0.        ,
       # 0.        , 0.        , 0.        , 0.        , 0.00293255,
       # 0.00293255, 0.        , 0.        , 0.        , 0.0058651 ,
       # 0.00293255, 0.        , 0.        , 0.02346041, 0.        ,
       # 0.6568915 , 0.00293255, 0.03812317, 0.0058651 , 0.07624633,
       # 0.        , 0.0058651 , 0.        , 0.05278592, 0.01173021]
# print(words[28]) # words[28]='tree' ha il set più ridotto di parole con cui viene confusa (riga con più zeri)
# len(list(np.argwhere(cmat == cmat.min())))=322 # numero totale di zeri

## 5. EXAMPLE OF MISCLASSIFICATION
R = random.randint(0, len(E)) 
index_mis = E[R] # chosen misclassification 
print("Example:", words[index_mis[2]] , "misclassified as", words[index_mis[1]]) 
## Draw the spectogram
spectrogram = Xtest[R, :].reshape(20, 80)
plt.imshow(spectrogram)
plt.colorbar()
plt.title(words[Ytest[R]])
plt.show()
## Find the name of the audio
names = open("test-names.txt").read().split()
ttl = names[R+1]
print('File name:', ttl)

# GOOD EXMPLE : R=2304 - word=four - file=four/234d6a48_nohash_1.wav


