import numpy as np



# Returns sigmoid
def sigmoid(x):
    return 1/(1+np.exp(-x)) 

def sigmoid_deriv(x):
    return x*(1-x)


class NeuralNet:
    # param x: input array
    # param y: output array
    def __init__(self, x, y):
        self.input = x
        self.syn0 = np.random.rand(self.input.shape[1],4) 
        self.syn1 = np.random.rand(4,1) 
        self.y = y
        self.output = np.zeros(self.y.shape)

    def forward_propagation(self):
        self.layer0 = sigmoid(np.dot(self.input, self.syn0))
        self.output = sigmoid(np.dot(self.layer0, self.syn1))


    def back_propagation(self):
        # using chain rule to find derivative of the loss function
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_deriv(self.output), self.syn1.T) * sigmoid_deriv(self.layer0)))
        d_weights2 = np.dot(self.layer0.T, (2*(self.y - self.output) * sigmoid_deriv(self.output)))

        # update the weights with the derivative of the loss function
        self.syn0 += d_weights1
        self.syn1 += d_weights2


if __name__ == "__main__":
    X = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]])
    y = np.array([[0],[1],[1],[0]])
    nn = NeuralNet(X,y)

    for i in range(5000):
        nn.forward_propagation()
        nn.back_propagation()

    print(nn.output)