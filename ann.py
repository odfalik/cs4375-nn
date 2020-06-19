import numpy as np
import activations

class ANN(object):

    def __init__(self, dims, activation):
        self.dims = np.asarray(dims, dtype=np.ushort)   # (1, L) array of layer dimensions
        self.L = len(self.dims)                         # number of layers
        self.W = np.asarray([None] * (self.L-1))          # (1, L-1) array of weight matrices which are indexed by [to_index, from_index]

        N = [None] * self.L
        for l, dim in enumerate(self.dims):
            N[l] = np.asarray([0] * dim)
        self.Z = np.asarray(N)                          # (1, L) jagged array of net arrays         (1, dim)
        self.A = np.asarray(N)                          # (1, L) jagged array of activation arrays  (1, dim)
        self.D = np.asarray(N)                          # (1, L) jagged array of delta arrays       (1, dim)

        self.B = [None] * self.L-1
        for l, dim in enumerate(self.dims):
            # self.b[l] = 
            pass
        self.B = np.asarray(N)                          # (1, L-1) jagged array of biases           (1, dim)


        self.activation = activation                    # string containing name of activation function: sigmoid/tanh/relu
        self.activation_f = activations.getActivationF(activation)
        self.activation_f_derivative = activations.getActivationFDerivative(activation)


    def predict(self, x_v):
        for l in range(self.L):
            if (l == 0):
                self.A[l] = x_v.reshape(-1, 1)
            else:
                a = self.A[l - 1].T
                x = self.W[l].T
                b = self.B[0,l]
                print(f'asdfasdf', x.shape, b.shape)
                self.Z = (a @ x) + b


    def train(self, x_m, y_m, learning_rate=0.001):

        # Weight initialization
        for l in range(1, self.L):
            l_dim = self.dims[l]
            prev_l_dim = self.dims[l-1]
            
            # Suggested weight/bias initialization techniques from https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78
            if (self.activation == 'sigmoid' or self.activation == 'relu'):
                # Xavier initialization (for sigmoid and relu activations)
                self.W[l] = np.random.randn(l_dim, prev_l_dim) * np.sqrt( 1 / prev_l_dim)
                self.B[0,l] = np.random.randn() * np.sqrt(1 / prev_l_dim)
            else:
                # He initialization (for tanh activations)
                self.W[l] = np.random.randn(l_dim, prev_l_dim) * np.sqrt( 2 / prev_l_dim)
                self.B[0,l] = np.random.randn() * np.sqrt(2 / prev_l_dim)

        # Until convergence
        for epoch in range(1000):
            print(f'Epoch {epoch}')

            # For each training example
            for sample_idx, sample_x_v in enumerate(x_m):

                # Forward pass
                self.predict(sample_x_v)

                # Backpropagation
                # Iterate layers in reverse (excluding the input layer)
                for l in range(self.L-1, 0, -1):

                    # Cache delta_j (error term) which represents sensitivity of Cost to unit's activation -- will be used when computing later downstream summations                    
                    if (l == self.L-1):     # Output layer
                        self.D[l] = ((y_m[sample_idx] - self.A[l]) * self.activation_f_derivative(self.A[l])).reshape(-1, 1)
                    else:                   # Hidden layer
                        print(self.activation_f_derivative(self.A[l]).shape, self.D[l+1].shape, self.W[l].shape)
                        self.D[l] = self.activation_f_derivative(self.A[l]).reshape(-1, 1) * (self.W[l] @ self.D[l+1])
                    





    def test(self):
        pass