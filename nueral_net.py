import numpy as np



# Returns sigmoid
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_deriv(x):
    return x*(1-x)


class NueralNet:
    # param x: input array
    # param y: output array
    def __init__(self, x, y):
        self.input = x
        self.syn0 = np.random.rand(self.input.shape[1],4) 
        self.syn1 = np.random.rand(4,1) 
        self.y = y
        self.output = np.zeros(self.y.shape)

    def f_propagation(self):
        self.layer0 = sigmoid(np.dot(self.input, self.syn0))
        self.output = sigmoid(np.dot(self.layer0, self.syn1))