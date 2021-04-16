import matplotlib.pyplot as pyplot
import numpy as np

class Spiral:
    # param n : numbers of points for each class
    # param d : the dimensionality
    # param k : number of classes
    # param
    def __init__(self, n, d, k):
        self.X = np.zeros((n*k),d)
        self.y = np.zeros((n*k), dtype='uint8')
        for i in range(k):
            ix = range(N*j,N*(j+1))
            r = np.linspace(0.0,1,N) # radius
            t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
            self.X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
            self.y[ix] = i

    def visualize(self):
        plt.scatter(self.X[:, 0], self.X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
        plt.show()