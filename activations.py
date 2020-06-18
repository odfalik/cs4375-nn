import numpy as np




def getActivationFDerivative(activation):
    pass

def getActivationF(activation):
    if (activation == 'sigmoid'):
        return lambda x: 1/(1 + np.exp(-x)) 
    elif (activation == 'tanh'):
        return np.tanh
    elif (activation == 'relu'):
        return lambda x: 
    else:
        print(f'Unsupported activation function: {activation}')
        exit(1)