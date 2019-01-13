import numpy as np
import sys,pickle
#HOW TO LEARN?

class NewralNet:
    def __init__(self,input_size,hide_size,output_size,dist_scale = 1.0):
        self.weights = {}
        self.diff_weights = {}
        self.weights["W1"] = dist_scale * np.random.rand(input_size,hide_size)
        self.weights["b1"] = np.zeros(hide_size)
        self.weights["W2"] = dist_scale * np.random.rand(hide_size,output_size)
        self.weights["b2"] = np.zeros(output_size)
        self.lr = 0.01
        self.N = 0

    def save_weights(self):
        for key in self.weights.keys():
            with open(key + ".pickle","wb") as f:
                pickle.dump(self.weights[key],f)
            print("saved ",key)
        print("finished save all")
        print()

    def load_weights(self):
        for key in self.weights.keys():
            try:
                with open(key + ".pickle","rb") as f:
                    self.weights[key] = pickle.load(f)
                    print("loaded",key)
            except FileNotFoundError:
                print("load failed",key)
                print("End program")
                sys.exit()
        print("finished load all")
        print()

    def predict(self,x,t):
        self.N = x.shape[0]
        if self.N == 0:
            print("N set as 0")
            print("End program")
            sys.exit()
        a =  np.dot(x,self.weights["W1"]) + self.weights["b1"] # A = X*W1+b1
        z = self._sigmoid(a) #Z = sigmoid(A)
        i = np.dot(z,self.weights["W2"]) + self.weights["b2"] # I = Z*W2+b2
        y = self._softmax(i)

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
            print(self.weights[key])
            self.weights[key] = self.weights[key] - (self.lr * self.diff_weights[key])
            print(self.weights[key])
            print()
        print("weights have been updated!")

x = np.array([[1,-2],[2,-4],[3,-5]])
t = np.array([[1,0],[0,1],[1,0]])

net = NewralNet(2,3,2)
for i in range(50):
    net.learn(x,t)
net.save_weights()
