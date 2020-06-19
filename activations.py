import numpy as np


# Returns a lambda which calculates the derivative of the activation function for a given activation value
def getActivationFDerivative(activation):
    if (activation == 'sigmoid'):
        return lambda x: x * (np.ones(x.shape[0]) - x)

    elif (activation == 'tanh'):
        return lambda x: 1-np.square(np.tanh(x))

    elif (activation == 'relu'):
        return lambda x: (x>0)*1    # TODO double check
    
    elif (activation == 'identity'):
        return lambda x: 1

    else:
        print(f'Unsupported activation function: {activation}')
        exit(1)


# Returns the respective activation function
def getActivationF(activation):
    if (activation == 'sigmoid'):
        return lambda x: 1/(1 + np.exp(-x))

    elif (activation == 'tanh'):
        return np.tanh

    elif (activation == 'relu'):
        lambda x: x if x > 0 else 0

    elif (activation == 'identity'):
        return lambda x: x

    else:
        print(f'Unsupported activation function: {activation}')
        exit(1)