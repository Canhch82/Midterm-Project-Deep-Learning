from model_helper import initialize_parameters_deep, linear_activation_forward, linear_activation_backward
import numpy as np

class Model:
    def __init__(self, layer_dims=[2500, 1048, 512, 64, 1]) -> None:
        self.parameters = initialize_parameters_deep(layer_dims=layer_dims)
    
    def fit(self, data_generator, epochs=50, lr=0.5):
        '''
        Trains the model using the provided data generator.

        Parameters:
        data_generator -- function yielding batches of (X, Y)
        epochs -- number of epochs for training
        lr -- learning rate for gradient descent
        '''
        self.epochs = epochs
        self.lr = lr
       
        print("Start training")
        
        for X, Y in data_generator():
            self.X = X
            self.Y = Y 
            break
        for i in range(epochs):

            self.L_model_forward()
            self.compute_cost()
            if True:
                print(f"Loss at {i}: {self.cost}");                                                                                                                                                                                                                                                 break
            self.L_model_backward()
            self.update_parameters()

    def L_model_forward(self):
        """
        Implements forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation.
        """
        caches = []
        A = self.X
        L = len(self.parameters) // 2  # number of layers in the neural network

        for l in range(1, L):
            A_prev = A
            W = self.parameters['W' + str(l)]
            b = self.parameters['b' + str(l)]
            A, cache = linear_activation_forward(A_prev, W, b, 'relu')
            caches.append(cache)
        
        W, b = self.parameters['W' + str(L)], self.parameters['b' + str(L)]
        AL, cache = linear_activation_forward(A, W, b, 'sigmoid')
        caches.append(cache)
        
        assert(AL.shape == (1, self.X.shape[1]))
        self.AL = AL
        self.caches = caches
    
    def compute_cost(self):
        """
        Computes the cross-entropy cost.
        """
        m = self.Y.shape[1]
        cost = -np.mean(self.Y * np.log(self.AL) + (1 - self.Y) * np.log(1 - self.AL))
        self.cost = np.squeeze(cost)  # Ensure cost is a scalar

    def L_model_backward(self):
        """
        Implements the backward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID group.
        """
        grads = {}
        L = len(self.caches)  # the number of layers
        m = self.AL.shape[1]
        Y = self.Y.reshape(self.AL.shape)  # after this line, Y is the same shape as AL

        dAL = - (np.divide(Y, self.AL) - np.divide(1 - Y, 1 - self.AL))

        current_cache = self.caches[-1]
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")

        for l in reversed(range(L-1)):
            current_cache = self.caches[l]
            dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, "relu")
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        self.grads = grads

    def update_parameters(self):
        """
        Updates parameters using gradient descent.
        """
        L = len(self.parameters) // 2  # number of layers in the neural network

        for l in range(1, L + 1):
            self.parameters['W' + str(l)] = self.parameters['W'+str(l)] - self.lr * self.grads['dW' + str(l)]
            self.parameters['b' + str(l)] =  self.parameters['b'+str(l)] -  self.lr* self.grads['db' + str(l)]
