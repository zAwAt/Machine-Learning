import sys
import os
import pickle

import numpy as np
from excel_reader import *

#This program is for python3.6.x

class NewralNet:
    def __init__(self,input_size,hide_size,output_size,lr = 0.01,dist_scale = 1.0):
        self.weights = {}
        self.diff_weights = {}
        self.weights["W1"] = dist_scale * np.random.rand(input_size,hide_size)
        self.weights["b1"] = np.zeros(hide_size)
        self.weights["W2"] = dist_scale * np.random.rand(hide_size,output_size)
        self.weights["b2"] = np.zeros(output_size)
        self.lr = lr
        self.N = 0

    def show_all_weights(self):
        for key in self.weights.keys():
            print( key,"=",self.weights[key].shape)
            print(self.weights[key])
            print()
        print("showed all weights\n")

    def save_weights(self):
        for key in self.weights.keys():
            try:
                with open(os.path.join(os.getcwd(),"weights",key + ".pickle"),"wb") as f:
                    pickle.dump(self.weights[key],f)
            except FileNotFoundError:
                print("save failed",key)
                print("End program")
                sys.exit()
            print("saved ",key)
        print("finished save all weights\n")

    def load_weights(self):
        for key in self.weights.keys():
            try:
                with open(os.path.join(os.getcwd(),"weights",key + ".pickle"),"rb") as f:
                    self.weights[key] = pickle.load(f)
                    print("loaded",key)
            except FileNotFoundError:
                print("load failed",key)
                print("End program")
                sys.exit()
        print("finished load all weights\n")

    def predict(self,x):
        self.N = x.shape[0]
        if self.N == 0:
            print("N set as 0")
            print("End program")
            sys.exit()
        a =  np.dot(x,self.weights["W1"]) + self.weights["b1"] # A = X*W1+b1
        z = self._sigmoid(a) #Z = sigmoid(A)
        i = np.dot(z,self.weights["W2"]) + self.weights["b2"] # I = Z*W2+b2
        y = self._softmax(i)
        return y

    def learn(self,x,t):
        #FORWARD
        self.N = x.shape[0]
        if self.N == 0:
            print("N set as 0. End program")
            sys.exit()
        a =  np.dot(x,self.weights["W1"]) + self.weights["b1"] # A = X*W1+b1
        z = self._sigmoid(a) #Z = sigmoid(A)
        i = np.dot(z,self.weights["W2"]) + self.weights["b2"] # I = Z*W2+b2
        y = self._softmax(i) #Y = softmax(I)

        #BACKWARD
        dy = (y-t)/self.N #(dL/dI)
        self.diff_weights["W2"] = np.dot(z.T,dy) #diff_W2 = z.T*(dL/dI)
        self.diff_weights["b2"] = np.sum(dy,axis = 0) #diff_b2 = SUM{(dL/dI)}
        dz = np.dot(dy,self.weights["W2"].T) #(dL/dZ)
        da = self._sigmoid_grad(a) * dz #(dL/dA)
        self.diff_weights["W1"] = np.dot(x.T,da) #diff_W1 = x.T*(dL/dA)
        self.diff_weights["b1"] = np.sum(da,axis = 0) #diff_b1 = SUM{(dL/dA)}
        self._weight_update()

    def _sigmoid(self,array):
        return 1/(1+np.exp(-1*array))

    def _sigmoid_grad(self,array):
        return self._sigmoid(array)*(1.0 - self._sigmoid(array))

    def _softmax(self,array):
        if array.ndim == 2:
            array = array.T
            max = np.max(array,axis = 0)
            array = array - max
            return (np.exp(array)/np.sum(np.exp(array),axis = 0)).T

        elif array.ndim == 1:
            max = np.max(array)
            array = array - max
            return np.exp(array)/np.sum(np.exp(array))

    def _weight_update(self):
        for key in self.weights.keys():
            self.weights[key] = self.weights[key] - (self.lr * self.diff_weights[key])
        print("updated all weights!\n")

x,t = readxlsx("data.xlsx")
net = NewralNet(3,5,3)
net.show_all_weights()
for i in range(10000):
    random = np.random.choice(x.shape[0],20)
    x_train = x[random]
    t_train = t[random]
    net.learn(x,t)
xc = np.array([0.4,0.3,0.2])
net.show_all_weights()
print(net.predict(xc))
