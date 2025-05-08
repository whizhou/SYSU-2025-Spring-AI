import numpy as np
import matplotlib.pyplot as plt

from layers import *

class MLP:
    def __init__(
        self,
        input_dim=2,
        hidden_dims=[64, 128],
        output_dim=1,
        weight_scale=0.01,
        reg=0.0,
    ):
        self.params = {}
        self.reg = reg
        self.num_layers = len(hidden_dims) + 1

        # Initialize weights and biases for the hidden layers
        self.params["W1"] = np.random.randn(input_dim, hidden_dims[0]) * weight_scale
        self.params["b1"] = np.zeros(hidden_dims[0])
        self.params["gamma1"] = np.ones(hidden_dims[0])
        self.params["beta1"] = np.zeros(hidden_dims[0])

        for i in range(1, len(hidden_dims)):
            self.params[f"W{i+1}"] = np.random.randn(hidden_dims[i-1], hidden_dims[i]) * weight_scale
            self.params[f"b{i+1}"] = np.zeros(hidden_dims[i])
            self.params[f"gamma{i+1}"] = np.ones(hidden_dims[i])
            self.params[f"beta{i+1}"] = np.zeros(hidden_dims[i])
        
        self.params[f"W{len(hidden_dims)+1}"] = np.random.randn(hidden_dims[-1], output_dim) * weight_scale
        self.params[f"b{len(hidden_dims)+1}"] = np.zeros(output_dim)

    def loss(self, x, y=None):
        """
        Compute the loss and gradients for the MLP.
        
        Parameters:
        - x: Input data of shape (N, D)
        - y: Ground truth data of shape (N, output_dim), None for test mode
        
        Returns:
        - loss: Scalar loss value
        - grads: Dictionary of gradients with respect to parameters
        """
        # Forward pass
        scores = self.forward(x)

        if y is None:
            return scores

        # Compute loss and gradients
        loss, grads = self.backward(scores, y)
        
        self.grads = grads

        return loss
    
    def forward(self, x):
        """
        Forward pass through the MLP.
        
        Parameters:
        - x: Input data of shape (N, D)
        
        Returns:
        - scores: Output scores of shape (N, output_dim)
        """
        # Store the cache for backpropagation
        self.cache = {}
        
        # First hidden layer
        out, self.cache["cache1"] = affine_relu_layernorm_forward(
            x, self.params["W1"], self.params["b1"], self.params["gamma1"], self.params["beta1"])
        
        # out, self.cache["cache1"] = affine_relu_forward(
            # x, self.params["W1"], self.params["b1"])

        # Hidden layers
        for i in range(2, self.num_layers):
            out, self.cache[f"cache{i}"] = affine_relu_layernorm_forward(
                out, self.params[f"W{i}"], self.params[f"b{i}"], self.params[f"gamma{i}"], self.params[f"beta{i}"])
            # out, self.cache[f"cache{i}"] = affine_relu_forward(
                # out, self.params[f"W{i}"], self.params[f"b{i}"])
        
        # Output layer
        scores, self.cache[f"cache{self.num_layers}"] = affine_forward(
            out, self.params[f"W{self.num_layers}"], self.params[f"b{self.num_layers}"]
        )
        
        return scores
    
    def backward(self, scores, y):
        """
        Backward pass through the MLP with MSE loss.
        
        Parameters:
        - scores: Output scores of shape (N, output_dim)
        - y: (N, output_dim) ground truth data
        
        Returns:
        - loss: Scalar loss value
        - grads: Dictionary of gradients with respect to parameters
        """
        # Compute the loss
        loss, dscores = mse_loss(scores, y)
        
        # Backward pass
        grads = {}
        
        # Output layer
        dout, grads[f"W{self.num_layers}"], grads[f"b{self.num_layers}"] = affine_backward(
            dscores, self.cache[f"cache{self.num_layers}"]
        )
        
        # Hidden layers
        for i in range(self.num_layers - 1, 1, -1):
            dout, grads[f"W{i}"], grads[f"b{i}"], grads[f"gamma{i}"], grads[f"beta{i}"] = affine_relu_layernorm_backward(
                dout, self.cache[f"cache{i}"]
            )
            # dout, grads[f"W{i}"], grads[f"b{i}"] = affine_relu_backward(dout, self.cache[f"cache{i}"])
        
        # First hidden layer
        dout, grads["W1"], grads["b1"], grads["gamma1"], grads["beta1"] = affine_relu_layernorm_backward(
            dout, self.cache["cache1"])
        # dout, grads["W1"], grads["b1"] = affine_relu_backward(dout, self.cache["cache1"])
        
        # Add regularization to the gradients
        for i in range(1, self.num_layers + 1):
            loss += 0.5 * self.reg * np.sum(self.params[f"W{i}"] ** 2)
            grads[f"W{i}"] += self.reg * self.params[f"W{i}"]
        
        for i in range(1, self.num_layers):
            loss += 0.5 * self.reg * np.sum(self.params[f"gamma{i}"] ** 2)
            loss += 0.5 * self.reg * np.sum(self.params[f"beta{i}"] ** 2)
            grads[f"gamma{i}"] += self.reg * self.params[f"gamma{i}"]
            grads[f"beta{i}"] += self.reg * self.params[f"beta{i}"]
        
        return loss, grads
    
    def step(self, learning_rate=1e-3):
        """
        Update the parameters using gradient descent.
        
        Parameters:
        - learning_rate: Learning rate for the update
        """
        for key in self.params:
            self.params[key] -= learning_rate * self.grads[key]