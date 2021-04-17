import matplotlib.pyplot as plt
import numpy as np
import spiral_dataset as sp

if __name__ == "__main__":
    D = 2
    K = 3
    ds = sp.Spiral(100,D,K)

    # Init random weights
    W = 0.01 * np.random.randn(D,K)
    b = np.zeros((1,K))
    

    scores = np.dot(ds.X,W) + b


    num_examples = ds.X.shape[0]
    # get unnormalized probabilities
    exp_scores = np.exp(scores)
    # normalize them for each example
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # 1d array of probabilities assigned to the correct classes for each example
    correct_logprobs = -np.log(probs[range(num_examples),ds.y])


    # compute the loss: average cross-entropy loss and regularization
    data_loss = np.sum(correct_logprobs)/num_examples
    reg = 1e-3
    reg_loss = 0.5*reg*np.sum(W*W)
    loss = data_loss + reg_loss
    if i % 1000 == 0:
        print("iteration %d: loss %f" % (i, loss))