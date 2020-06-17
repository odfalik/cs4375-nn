import numpy as np


class ANN(object):

    def __init__(self, input_dim, hidden_dims, output_dim):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.L = len(hidden_dims) + 1
        self.W = [None] * self.L

    def predict(self, x_v):
        W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
        # Forward propagation
        z1 = x.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return np.argmax(probs, axis=1)

    def train(self, x_m, y_m, learning_rate=0.001):

        # Initialize weights
        for layer in range(self.L):

            if (layer == 0):
                new_W_m_shape = (self.hidden_dims[0], self.input_dim)
            elif (layer < self.L - 1):
                new_W_m_shape = (self.hidden_dims[layer], self.hidden_dims[layer-1])
            else:
                new_W_m_shape = (self.output_dim, self.hidden_dims[layer-1])

            # Suggested weight initialization technique from https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78
            new_W_m = np.random.randn(new_W_m_shape[0], new_W_m_shape[1]) * np.sqrt( 2 / (new_W_m_shape[1] + new_W_m_shape[0]) )
            # print(f'layer {layer} with W_m {new_W_m}')
            self.W[layer] = new_W_m

        # Until convergence, for each training example
        for sample_idx, sample_x_v in enumerate(x_m):
            self.predict(sample_x_v)
            

    def test(self):
        pass