import numpy as np
import activations

class ANN(object):

    def __init__(self, dims, activation):
        self.dims = np.asarray(dims, dtype=np.ushort)   # (1, L) array of layer dimensions
        self.L = len(self.dims)                         # number of layers
        self.W = np.asarray([None] * self.L)            # (1, L) array of weight matrices
        self.B = np.asarray([None] * self.L)            # (1, L) array of biases

        A = [None] * self.L 
        for l, dim in enumerate(self.dims):
            A[l] = np.asarray([0] * dim)
        self.A = np.asarray(A)                          # (1, L) array of activation arrays (1, dim)

        self.activation = activation                    # string containing name of activation function: sigmoid/tanh/relu
        self.activation_f = activations.getActivationF(activation)
        self.activation_f_derivative = activations.getActivationFDerivative(activation)


    def predict__(self, x_v):
        W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
        z1 = x.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return np.argmax(probs, axis=1)

    def predict(self, x_v):
        for l in range(self.L):
            if (l == 0):
                self.A[l] = 
            else:
                
        

    def train(self, x_m, y_m, learning_rate=0.001):
        # Weight initialization
        for l in range(1, self.L):
            l_dim = self.dims[l]
            prev_l_dim = self.dims[l-1]
            
            # Suggested weight/bias initialization techniques from https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78
            if (self.activation == 'sigmoid' or self.activation == 'relu'):
                # Xavier initialization (for sigmoid and relu activations)
                self.W[l] = np.random.randn(l_dim, prev_l_dim) * np.sqrt( 1 / prev_l_dim)
                self.B[l] = np.random.randn(1,1) * np.sqrt(1 / prev_l_dim)
            else:
                # He initialization (for tanh activations)
                self.W[l] = np.random.randn(l_dim, prev_l_dim) * np.sqrt( 2 / prev_l_dim)
                self.B[l] = np.random.randn(1,1) * np.sqrt(2 / prev_l_dim)

        # Until convergence, for each training example
        for sample_idx, sample_x_v in enumerate(x_m):
            # Forward pass
            self.predict(sample_x_v)
            

    def test(self):
        pass