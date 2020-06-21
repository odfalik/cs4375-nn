import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))
sigmoid_v = np.vectorize(sigmoid)

delReluSingleVal = lambda x: (x>0)*1
delRelu_v = np.vectorize(delReluSingleVal)
def relu(x):
    return x if x > 0 else 0
relu_v = np.vectorize(relu)

# Returns a lambda which calculates the derivative of the activation function for a given activation value
def getActivationFDerivative(activation):
    if (activation == 'sigmoid'):
        return lambda x: sigmoid_v(x) * (np.ones(x.shape) - sigmoid_v(x))

    elif (activation == 'tanh'):
        return lambda x: np.ones(x.shape[0]) - np.square(np.tanh(x))

    elif (activation == 'relu'):
        return lambda x: delRelu_v    # TODO double check

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
        return relu_v

    else:
        print(f'Unsupported activation function: {activation}')
        exit(1)