import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import activations

class ANN(object):

    def __init__ (self, dims, activation, plot=False):
        self.dims = np.asarray(dims)                    # (1, L) array of layer dimensions
        self.L = len(self.dims)                         # number of layers
        self.W = [0] * self.L                           # (1, L) array of weight matrices which are indexed by [to_layer][to_index, from_index]

        self.node_template = ([None] * self.L)          # (1, L) jagged array. Effectively describes the shape of nodes of our network
        for l, dim in enumerate(self.dims):
            self.node_template[l] = np.asarray([0] * dim).reshape(-1,1)
        self.Z = deepcopy(self.node_template)           # (1, L) jagged array of net arrays         (1, dim)
        self.A = deepcopy(self.node_template)           # (1, L) jagged array of activation arrays  (1, dim)
        self.D = deepcopy(self.node_template)           # (1, L) jagged array of delta arrays       (1, dim)
        self.B = deepcopy(self.node_template)           # (1, L) jagged array of bias arrays        (1, dim)

        self.activation = activation                    # string containing name of activation function: sigmoid/tanh/relu
        self.activation_f = activations.getActivationF(activation)
        self.activation_f_derivative = activations.getActivationFDerivative(activation)

        self.plot = plot
        if self.plot:
            self.activations_log = np.empty((1, self.dims[self.L-1]))


    def predict(self, x_v, y_v=None):

        for l in range(self.L):
            if (l == 0):    # Input layer
                self.A[l] = x_v.reshape(-1, 1)
            else:           # Hidden or Output layer
                w = self.W[l]
                a = self.A[l - 1].reshape(-1,1)
                b = self.B[l]
                weighted_sum = w @ a
                self.Z[l] = weighted_sum + b
                self.A[l] = self.activation_f(self.Z[l])
        self.prediction = np.argmax(self.A[self.L-1])
        if y_v is not None:
            self.expected = np.argmax(y_v)


    def getError(self, o_v, t_v):
        return np.sum(np.square(t_v - o_v)) / 2

    def updateWeights(self, latest_batch_size, max_batch_size, momentum_factor):
        sum_W_delta = deepcopy(self.W_template)
        sum_B_delta = deepcopy(self.node_template)

        for l in range(1, self.L):                          # For each layer
            for batch_index in range(latest_batch_size):    # For each sample in batch
                sum_W_delta[l] = sum_W_delta[l] + self.W_delta[batch_index][l]       # Sum weight adjustments for all samples in batch **only for this layer
                sum_B_delta[l] = sum_B_delta[l] + self.B_delta[batch_index][l]

            self.batch_W_delta[l] = (sum_W_delta[l] / latest_batch_size) + (momentum_factor * self.batch_W_delta[l])
            self.batch_B_delta[l] = (sum_B_delta[l] / latest_batch_size) + (momentum_factor * self.batch_B_delta[l])
            self.W[l] = self.W[l] + self.batch_W_delta[l]
            self.B[l] = self.B[l] + self.batch_B_delta[l]

        self.W_delta = [deepcopy(self.W_template)] * max_batch_size
        self.B_delta = [deepcopy(self.node_template)] * max_batch_size


    def train(self, x_m, y_m, learning_rate=0.001, max_batch_size=32, momentum_factor=0.1):
        num_samples = len(x_m)

        # Weight initialization
        for l in range(1, self.L):
            l_dim = self.dims[l]
            prev_l_dim = self.dims[l-1]
            
            # Suggested weight/bias initialization techniques from https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78
            if (self.activation == 'sigmoid' or self.activation == 'relu'):
                # Xavier initialization (for sigmoid and relu activations)
                self.W[l] = np.random.randn(l_dim, prev_l_dim) * np.sqrt(1 / prev_l_dim)
                self.B[l] = np.random.randn(l_dim, 1) * np.sqrt(1 / prev_l_dim)
            else:
                # He initialization (for tanh activations)
                self.W[l] = np.random.randn(l_dim, prev_l_dim) * np.sqrt(2 / prev_l_dim)
                self.B[l] = np.random.randn(l_dim, 1) * np.sqrt(2 / prev_l_dim)

            self.W_template = []
            for l in self.W:
                self.W_template.append(l*0)
            self.W_delta = [deepcopy(self.W_template)] * max_batch_size
            self.B_delta = [deepcopy(self.node_template)] * max_batch_size

        self.batch_W_delta = deepcopy(self.W_template)
        self.batch_B_delta = deepcopy(self.node_template)

        self.test(x_m, y_m)


        # Until convergence
        for epoch in range(1000):

            # For each training example
            for sample_idx, sample_x_v in enumerate(x_m):

                idx_in_batch = sample_idx % max_batch_size

                # Forward pass
                self.predict(sample_x_v)

                # Backpropagation
                # Iterate layers in reverse (excluding the input layer)
                for l in range(self.L-1, 0, -1):

                    # Cache delta_j (error term) which represents sensitivity of Cost to unit's activation -- will be used when computing later downstream summations                    
                    if (l == self.L-1):     # Output layer

                        self.D[l] = ( (self.activation_f_derivative(self.A[l])) * (y_m[sample_idx].reshape(-1,1) - self.A[l]) ).reshape(-1, 1)

                    else:                   # Hidden layer
                        self.D[l] = ( self.activation_f_derivative(self.A[l]).reshape(-1, 1) * (self.W[l+1].T @ self.D[l+1]) ).reshape(-1, 1)
                    
                    # Save weight updates
                    for j in range(self.dims[l]):       # For each node j in current layer
                        for i in range(self.dims[l - 1]):       # For each node i in upstream layer
                            self.W_delta[idx_in_batch][l][j,i] = learning_rate * self.D[l][j] * self.A[l-1][i]
                            self.B_delta[idx_in_batch][l][j]   = learning_rate * self.D[l][j]

                # If last sample in batch
                if (idx_in_batch == max_batch_size - 1):
                    self.updateWeights(idx_in_batch + 1, max_batch_size, momentum_factor)
                elif (sample_idx == num_samples - 1):
                    self.updateWeights(idx_in_batch + 1, max_batch_size, momentum_factor)

            # Shuffle dataset
            randomize = np.arange(len(x_m))
            np.random.shuffle(randomize)
            x_m = x_m[randomize]
            y_m = y_m[randomize]


            if not (epoch % 1):
                if self.plot:
                    self.activations_log = np.concatenate((self.activations_log, self.A[self.L-1].T))
                    plt.plot(self.activations_log)
                    plt.pause(1)

                print(f'Epoch {epoch}')
                self.test(x_m, y_m)




    def test(self, x_m, y_m):
        err_sum = 0
        num_correct = 0
        for sample_idx, sample_x_v in enumerate(x_m):
            self.predict(sample_x_v, y_m[sample_idx])
            if (self.prediction == self.expected):
                num_correct += 1
            err_sum = self.getError(self.A[self.L - 1], y_m[sample_idx])
        print(f'\t cost: {err_sum / len(x_m)} accuracy: {100*num_correct/len(x_m)}% \t {self.prediction} {self.expected} confidence:{self.A[self.L-1][self.prediction][0]}')