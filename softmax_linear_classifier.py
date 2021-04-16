import matplotlib.pyplot as plt
import numpy as np
import spiral_dataset as sp

if __name__ == "__main__":
    D = 2
    K = 4
    ds = sp.Spiral(100,D,K)

    # Init random weights
    W = 0.01 * np.random.randn(D,K)
    b = np.zeros((1,K))
    print(b)