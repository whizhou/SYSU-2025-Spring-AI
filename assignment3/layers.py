import numpy as np

def affine_forward(x, w, b):
    """
    Computes the forward pass of an affine layer.
    
    Args:
    x -- input data of shape (N, d1, d2, ...) where N is the number of samples and D is the dimensionality
    w -- weights of shape (D, M) where M is the number of output features
    b -- biases of shape (M,) where M is the number of output features
    
    Returns:
    out -- output data of shape (N, M)
    cache -- tuple containing x, w, and b for backpropagation
    """
    N = x.shape[0]
    out = np.matmul(x.reshape(N, -1), w) + b
    cache = (x, w, b)
    return out, cache

def affine_backward(dout, cache):
    """
    Computes the backward pass of an affine layer.
    
    Args:
    dout -- upstream gradient of shape (N, M)
    cache -- tuple containing x, w, and b from the forward pass
    
    Returns:
    dx -- gradient with respect to x of shape (N, D)
    dw -- gradient with respect to w of shape (D, M)
    db -- gradient with respect to b of shape (M,)
    """
    x, w, b = cache
    N = x.shape[0]
    
    dx = np.matmul(dout, w.T).reshape(x.shape)
    dw = np.matmul(x.reshape(N, -1).T, dout)
    db = np.sum(dout, axis=0)
    
    return dx, dw, db

def relu_forward(x):
    """
    Computes the forward pass of a ReLU activation function.
    
    Args:
    x -- input data of any shape
    
    Returns:
    out -- output data of the same shape as x
    cache -- x for backpropagation
    """
    out = np.maximum(x, 0)
    cache = x
    return out, cache

def relu_backward(dout, cache):
    """
    Computes the backward pass of a ReLU activation function.
    
    Args:
    dout -- upstream gradient of the same shape as x
    cache -- x from the forward pass
    
    Returns:
    dx -- gradient with respect to x of the same shape as dout
    """
    x = cache
    dx = dout * (x > 0)
    return dx

def affine_relu_forward(x, w, b):
    """
    Computes the forward pass of an affine layer followed by a ReLU activation.
    
    Args:
    x -- input data of shape (N, d1, d2, ...)
    w -- weights of shape (D, M)
    b -- biases of shape (M,)
    
    Returns:
    out -- output data of shape (N, M)
    cache -- tuple containing the cache from affine and ReLU layers
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache

def affine_relu_backward(dout, cache):
    """
    Computes the backward pass of an affine layer followed by a ReLU activation.
    
    Args:
    dout -- upstream gradient of shape (N, M)
    cache -- tuple containing the cache from affine and ReLU layers
    
    Returns:
    dx -- gradient with respect to x of shape (N, d1, d2, ...)
    dw -- gradient with respect to w of shape (D, M)
    db -- gradient with respect to b of shape (M,)
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

def layernorm_forward(x, gamma, beta, eps=1e-5):
    """
    Computes the forward pass of layer normalization.
    
    Args:
    x -- input data of shape (N, D)
    gamma -- scale parameter of shape (D,)
    beta -- shift parameter of shape (D,)
    eps -- small constant for numerical stability
    
    Returns:
    out -- output data of shape (N, D)
    cache -- tuple containing x, gamma, beta, and the mean and variance for backpropagation
    """
    N, D = x.shape
    mean = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)
    
    x_normalized = (x - mean) / np.sqrt(var + eps)
    out = gamma * x_normalized + beta
    
    cache = (x, x_normalized, mean, var, gamma, beta, eps)
    return out, cache

def layernorm_backward(dout, cache):
    """
    Computes the backward pass of layer normalization.
    
    Args:
    dout -- upstream gradient of shape (N, D)
    cache -- tuple containing x, x_normalized, mean, var, gamma, beta, and eps from the forward pass
    
    Returns:
    dx -- gradient with respect to x of shape (N, D)
    dgamma -- gradient with respect to gamma of shape (D,)
    dbeta -- gradient with respect to beta of shape (D,)
    """
    x, x_normalized, mean, var, gamma, beta, eps = cache
    N, D = dout.shape
    
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_normalized, axis=0)
    
    dx_normalized = dout * gamma
    dvar = np.sum(dx_normalized * (x - mean) * -0.5 * (var + eps) ** -1.5, axis=1, keepdims=True)
    dmean = np.sum(dx_normalized * -1 / np.sqrt(var + eps), axis=1, keepdims=True) + dvar * np.mean(-2 * (x - mean), axis=1, keepdims=True)
    
    dx = dx_normalized / np.sqrt(var + eps) + dvar * 2 * (x - mean) / N + dmean / N
    
    return dx, dgamma, dbeta

def affine_relu_layernorm_forward(x, w, b, gamma, beta, eps=1e-5):
    """
    Computes the forward pass of an affine layer followed by ReLU and layer normalization.
    
    Args:
    x -- input data of shape (N, d1, d2, ...)
    w -- weights of shape (D, M)
    b -- biases of shape (M,)
    gamma -- scale parameter of shape (M,)
    beta -- shift parameter of shape (M,)
    eps -- small constant for numerical stability
    
    Returns:
    out -- output data of shape (N, M)
    cache -- tuple containing the cache from affine, ReLU, and layer normalization layers
    """
    a, fc_cache = affine_forward(x, w, b)
    out_relu, relu_cache = relu_forward(a)
    out_layernorm, ln_cache = layernorm_forward(out_relu, gamma, beta, eps)
    
    cache = (fc_cache, relu_cache, ln_cache)
    return out_layernorm, cache

def affine_relu_layernorm_backward(dout, cache):
    """
    Computes the backward pass of an affine layer followed by ReLU and layer normalization.
    
    Args:
    dout -- upstream gradient of shape (N, M)
    cache -- tuple containing the cache from affine, ReLU, and layer normalization layers
    
    Returns:
    dx -- gradient with respect to x of shape (N, d1, d2, ...)
    dw -- gradient with respect to w of shape (D, M)
    db -- gradient with respect to b of shape (M,)
    dgamma -- gradient with respect to gamma of shape (M,)
    dbeta -- gradient with respect to beta of shape (M,)
    """
    fc_cache, relu_cache, ln_cache = cache
    
    dout_layernorm, dgamma, dbeta = layernorm_backward(dout, ln_cache)
    da = relu_backward(dout_layernorm, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    
    return dx, dw, db, dgamma, dbeta

def mse_loss(y_pred, y_true):
    """
    Computes the Mean Squared Error (MSE) loss.
    
    Args:
    y_pred -- predicted values of shape (N,)
    y_true -- true values of shape (N,)
    
    Returns:
    loss -- MSE loss
    gradient -- gradient of the loss with respect to y_pred
    """

    loss = np.mean((y_pred - y_true) ** 2)
    gradient = 2 * (y_pred - y_true) / y_pred.size
    return loss, gradient