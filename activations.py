import numpy as np

sigmoid_s = lambda x: 1 / (1 + np.exp(-x))
sigmoid_v = np.vectorize(sigmoid_s)

relu_s = lambda x: (x > 0) * 1

# Returns a lambda which calculates the derivative of the activation function for a given activation value
def getActivationFDerivative(activation):
    if (activation == 'sigmoid'):
        return lambda x_v: sigmoid_v(x_v) * (np.ones(x_v.shape) - sigmoid_v(x_v))

    elif (activation == 'tanh'):
        return lambda x_v: np.ones(x_v.shape[0]) - np.square(np.tanh(x_v))

    elif (activation == 'relu'):
        return np.vectorize(relu_s)

    else:
        print(f'Unsupported activation function: {activation}')
        exit(1)


# Returns the respective activation function
def getActivationF(activation):
    if (activation == 'sigmoid'):
        return sigmoid_v

    elif (activation == 'tanh'):
        return np.tanh

    elif (activation == 'relu'):
        return np.vectorize(lambda x: x if x > 0 else 0)

    else:
        print(f'Unsupported activation function: {activation}')
        exit(1)