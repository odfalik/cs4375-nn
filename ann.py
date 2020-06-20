import numpy as np
import activations

class ANN(object):

    def __init__(self, dims, activation):
        self.dims = np.asarray(dims)                    # (1, L) array of layer dimensions
        self.L = len(self.dims)                         # number of layers
        self.W = [None] * self.L                        # (1, L) array of weight matrices which are indexed by [to_layer][to_index, from_index]

        node_template = ([None] * self.L)
        for l, dim in enumerate(self.dims):
            node_template[l] = np.asarray([0] * dim)
        self.Z = np.asarray(node_template)                          # (1, L) jagged array of net arrays         (1, dim)
        self.A = np.asarray(node_template)                          # (1, L) jagged array of activation arrays  (1, dim)
        self.D = np.asarray(node_template)                          # (1, L) jagged array of delta arrays       (1, dim)
        self.B = np.asarray(node_template)                          # (1, L) jagged array of bias arrays        (1, dim)

        self.activation = activation                    # string containing name of activation function: sigmoid/tanh/relu
        self.activation_f = activations.getActivationF(activation)
        self.activation_f_derivative = activations.getActivationFDerivative(activation)


    def predict(self, x_v, y_v=None):
        for l in range(self.L):
            if (l == 0):    # Input layer
                self.A[l] = x_v.reshape(-1, 1)
            else:           # Hidden or Output layer
                a = self.A[l - 1]
                x = self.W[l]
                b = self.B[l]
                weighted_sum = x @ a
                # print(f'weights:{x.shape} * activations:{a.shape} = {weighted_sum.shape} to add with {b.shape}')
                self.Z[l] = weighted_sum + b
                self.A[l] = self.activation_f(self.Z[l])
        self.prediction = np.argmax(self.A[self.L-1])
        if y_v is not None:
            self.expected = np.argmax(y_v)


    def getError(self, o_v, t_v):
        return np.sum(np.square(t_v - o_v)) / 2

    def updateWeights(self, latest_batch_size, max_batch_size):
        sum_W_delta = self.W_template
        for l in range(1, self.L):      # For each layer 
            for batch_index in range(latest_batch_size):  # For each sample in batch
                sum_W_delta[l] = sum_W_delta[l] + self.W_delta[batch_index][l]       # Sum weight adjustments for all samples in batch **only for this layer
            self.W[l] = self.W[l] + (sum_W_delta[l] / latest_batch_size)

        self.W_delta = [self.W_template] * max_batch_size


    def train(self, x_m, y_m, learning_rate=0.001, max_batch_size=32):
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

            self.W_template = self.W.copy()
            self.W_delta = [self.W_template] * max_batch_size


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
                    
                    # save weight updates
                    for j in range(self.dims[l]):       # foreach node j in current layer
                        for i in range(self.dims[l - 1]):       #for each node i in upstream layer
                            self.W_delta[idx_in_batch][l][j,i] = learning_rate * self.D[l][j] * self.A[l-1][i]

                if (idx_in_batch == max_batch_size - 1):
                    self.updateWeights(idx_in_batch + 1, max_batch_size)
                elif (sample_idx == num_samples - 1):
                    self.updateWeights(idx_in_batch + 1, max_batch_size)


            if not (epoch % 5):
                print(f'Epoch {epoch}')
                self.test(x_m, y_m)




    def test(self, x_m, y_m):
        err_sum = 0
        for sample_idx, sample_x_v in enumerate(x_m):
            self.predict(sample_x_v, y_m[sample_idx])
            err_sum = self.getError(self.A[self.L - 1], y_m[sample_idx])
        err_sum / len(x_m)
        print(f'\terror: {err_sum} \t {self.prediction} {self.expected} confidence:{self.A[self.L-1][self.prediction][0]}')