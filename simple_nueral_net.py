import numpy as np
import matplotlib.pyplot as plt


# calc mean array value
def mean_value(x):
    sum = 0
    for i in x:
        sum += i
    return (abs(sum/x.size))



# tanh function for testing
def tanh(x, deriv=False):
    if(deriv):
        return 1-x**2
    return (np.exp(x)-np.exp(-x)/np.exp(x)+np.exp(-x))



# sigmoid function
def nonlin(x, deriv=False):
    if(deriv):
        return x*(1-x)
    return 1/(1+np.exp(-x))



# input dataset
X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])

# output dataset
y = np.array([[0,0,1,1]]).T
np.random.seed(1)


# init weights randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1



error_data = np.zeros(100)
delta_data = np.zeros(100)

count = 0
for iter in range(10000):
    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))

    # calc error
    l1_error = y - l1
    l1_delta = l1_error * nonlin(l1, True)

    # adjust syn
    syn0 += np.dot(l0.T, l1_delta)

    if(iter % 100 == 0):
        error_data.put(count, (mean_value(l1_error) * 100))
        delta_data.put(count, (mean_value(l1_delta) * 100))
        count += 1
        # print("Training Report | Error: ", mean_value(l1_error) * 100)
        # print("Training Report | Delta: ", mean_value(l1_delta) * 100)


# print(error_data)
# print(delta_data)

# plt.plot(error_data, 'b--')
# plt.plot(delta_data, 'r--')
# plt.ylabel('Value')
# plt.show()

print("Output after training")
print(l1)





# test input dataset dataset
X1 = np.array([[1,0,1], [1,1,0], [0,1,1], [1,0,0]])
prediction = nonlin(np.dot(X1, syn0))
print("Test inputs \n", X1)
print("Prediction Based of training")
print(prediction)
