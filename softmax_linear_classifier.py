import matplotlib.pyplot as plt
import numpy as np
import spiral_dataset as sp

if __name__ == "__main__":
    reg = 1e-3
    step_size = 1e-0

    D = 2
    K = 3
    ds = sp.Spiral(100,D,K)

    # initialize parameters randomly
    h = 100 # size of hidden layer
    W = 0.01 * np.random.randn(D,K)
    b = np.zeros((1,K))
    W2 = 0.01 * np.random.randn(h,K)
    b2 = np.zeros((1,K))
    
    num_examples = ds.X.shape[0]
    for i in range(10000):
        
        # evaluate class scores, [N x K]
        hidden_layer = np.maximum(0, np.dot(ds.X, W) + b) # note, ReLU activation
        scores = np.dot(ds.X,W) + b
        
        # compute the class probabilities
        # get unnormalized probabilities
        exp_scores = np.exp(scores)
        # normalize them for each example
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # 1d array of probabilities assigned to the correct classes for each example
        correct_logprobs = -np.log(probs[range(num_examples),ds.y])
        # compute the loss: average cross-entropy loss and regularization
        data_loss = np.sum(correct_logprobs)/num_examples
        reg_loss = 0.5*reg*np.sum(W*W)
        loss = data_loss + reg_loss
        if i % 1000 == 0:
            print("iteration %d: loss %f" % (i, loss))
        
        # compute the gradient on scores
        dscores = probs
        dscores[range(num_examples),ds.y] -= 1
        dscores /= num_examples


        # backpropate the gradient to the parameters
        # first backprop into parameters W2 and b2
        dW2 = np.dot(hidden_layer.T, dscores)
        db2 = np.sum(dscores, axis=0, keepdims=True)
        # next backprop into hidden layer
        dhidden = np.dot(dscores, W2.T)
        # backprop the ReLU non-linearity
        dhidden[hidden_layer <= 0] = 0
        # finally into W,b
        # backpropate the gradient to the parameters (W,b)
        dW = np.dot(ds.X.T, dhidden)
        db = np.sum(dhidden, axis=0, keepdims=True)


        dW2 += reg * W2 
        dW += reg*W # regularization gradient

        # perform a parameter update
        W += -step_size * dW
        b += -step_size * db
        W2 += -step_size * dW2
        b2 += -step_size * db2


    # evaluate training set accuracy
    hidden_layer = np.maximum(0, np.dot(ds.X, W) + b)
    scores = np.dot(hidden_layer, W2) + b2
    predicted_class = np.argmax(scores, axis=1)
    print('training accuracy: %.2f' % (np.mean(predicted_class == ds.y)))