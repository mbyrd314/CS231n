from cs231n.layers import *
from cs231n.fast_layers import *


def affine_relu_forward(x, w, b):
  """
  Convenience layer that perorms an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  out, relu_cache = relu_forward(a)
  cache = (fc_cache, relu_cache)
  return out, cache


def affine_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db


pass


def conv_relu_forward(x, w, b, conv_param):
  """
  A convenience layer that performs a convolution followed by a ReLU.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  
  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  out, relu_cache = relu_forward(a)
  cache = (conv_cache, relu_cache)
  return out, cache


def conv_relu_backward(dout, cache):
  """
  Backward pass for the conv-relu convenience layer.
  """
  conv_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  s, relu_cache = relu_forward(a)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, relu_cache, pool_cache)
  return out, cache


def conv_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db


def conv_relu_batchnorm_forward(x, w, b, conv_param, gamma, beta, bn_param):
  """
  Convenience layer that performs a convolution, a ReLU, and spatial batchorm.
  
  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - gamma, beta: Parameters for the spatial batchnorm layer
  
  Returns a tuple of:
  - out: Output from the spatial batchnorm layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  #print(type(a))
  s, relu_cache = relu_forward(a)
  #print(type(s))
  out, sbn_cache = spatial_batchnorm_forward(s, gamma, beta, bn_param)
  cache = (conv_cache, relu_cache, sbn_cache)
  return out, cache
  

def conv_relu_batchnorm_backward(dout, cache):
  """
  Performs the backward pass for the conv-relu-batchnorm convenience layer.
  """
  conv_cache, relu_cache, sbn_cache = cache
  da, dgamma, dbeta = spatial_batchnorm_backward(dout, sbn_cache)
  da = relu_backward(da, relu_cache)
  dx, dw, db = conv_backward_fast(dout, conv_cache)
  return dx, dw, db, dgamma, dbeta
  

def conv_relu_pool_batchnorm_forward(x, w, b, conv_param, pool_param, gamma, beta, bn_param):
  """
  Convenience layer that performs a convolution, a ReLU, max pooling, and batchnorm.
  
  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Object for the pooling layer
  - gamma, beta, bn_param: Weights and parameters for spatial batchnorm
  
  Returns:
  - out: Output of the pooling layer
  - cache: Object to give to the backward pass
  """
  out, conv_cache = conv_forward_fast(x, w, b, conv_param)
  out, relu_cache = relu_forward(out)
  out, pool_cache = max_pool_forward_fast(out, pool_param)
  out, bn_cache = spatial_batchnorm_forward(out, gamma, beta, bn_param)
  cache = conv_cache, relu_cache, pool_cache, bn_cache
  return out, cache
  
  
def conv_relu_pool_batchnorm_backward(dout, cache):
  """
  Performs the backward pass for the conv-relu-pool-batchnorm convenience layer.
  """
  conv_cache, relu_cache, pool_cache, bn_cache = cache
  dout, dgamma, dbeta = spatial_batchnorm_backward(dout, bn_cache)
  dout = max_pool_backward_fast(dout, pool_cache)
  dout = relu_backward(dout, relu_cache)
  dx, dw, db = conv_backward_fast(dout, conv_cache)
  return dx, dw, db, dgamma, dbeta
  

def affine_relu_batchnorm_forward(x, w, b, gamma, beta, bn_param):
  """
  Convenience layer that performs an affine transformation, a ReLU, and batchnorm.
  
  Inputs:
  - x: Input to the affine layer
  - w, b: Weights and biases for the affine layer
  - gamma, beta: Parameters for the batchnorm layer
  - bn_param: Parameter object for the batchnorm layer
  
  Returns:
  - out: Output of the batchnorm layer
  - cache: Object to give to the backward pass
  """
  out, fc_cache = affine_forward(x, w, b)
  out, relu_cache = relu_forward(out)
  out, bn_cache = batchnorm_forward(out, gamma, beta, bn_param)
  cache = fc_cache, relu_cache, bn_cache
  return out, cache
  
  
def affine_relu_batchnorm_backward(dout, cache):
  """
  Performs the backward pass for the affine-relu-batchnorm convenience layer
  """
  fc_cache, relu_cache, bn_cache = cache
  dout, dgamma, dbeta = batchnorm_backward(dout, bn_cache)
  dout = relu_backward(dout, relu_cache)
  dx, dw, db = affine_backward(dout, fc_cache)
  return dx, dw, db, dgamma, dbeta
  
  
def affine_relu_batchnorm_dropout_forward(x, w, b, gamma, beta, bn_param, dropout_param):
  """
  Convenience layer that performs an affine transformation, a ReLU, batch
  normalization, and dropout.
  
  Inputs:
  - x: Input to the affine layer
  - w, b: Weights and biases for the affine layer
  - gamma, beta: Parameters for the batchnorm layer
  - bn_param: Parameter object for the batchnorm layer
  - dropout_param: Parameter object for the dropout layer
  
  Returns:
  - out: Output from the dropout layer
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_relu_forward(x, w, b)
  a, relu_cache = relu_forward(a)
  a, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
  out, d_cache = dropout_forward(a, dropout_param)
  cache = (fc_cache, relu_cache, bn_cache, d_cache)
  return out, cache
  
  
def affine_relu_batchnorm_dropout_backward(dout, cache):
  """
  Performs the backward pass for the affine-relu-batchnorm-dropout convenience layer
  """
  fc_cache, relu_cache, bn_cache, d_cache = cache
  da = dropout_backward(dout, d_cache)
  da, dgamma, dbeta = batchnorm_backward(da, bn_cache)
  da = relu_backward(da, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db, dgamma, dbeta
