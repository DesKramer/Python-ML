# Credit goes to the team behind CS231n at Standford University


import matplotlib.pyplot as plt
import numpy as np

class Spiral:
    # param n : numbers of points for each class
    # param d : the dimensionality
    # param k : number of classes
    # param
    def __init__(self, n, d, k):
        self.X = np.zeros((n*k,d))
        self.y = np.zeros((n*k), dtype='uint8')
        for i in range(k):
            ix = range(n*i,n*(i+1))
            r = np.linspace(0.0,1,n) # radius
            t = np.linspace(i*4,(i+1)*4,n) + np.random.randn(n)*0.2 # theta
            self.X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
            self.y[ix] = i

    def visualize(self):
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, s=40)
        plt.show()


if __name__ == "__main__":
    ds = Spiral(100,2,5)
    ds.visualize()