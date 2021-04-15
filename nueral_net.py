import numpy as np


class NueralNet:
    # param x: input array
    # param y: output array
    def __init__(self, x, y):
        self.input = x
        self.syn0 = np.random.rand(self.input.shape[1],4) 
        self.syn1 = np.random.rand(4,1) 
        self.y = y
        self.output = np.zeros(self.y.shape)