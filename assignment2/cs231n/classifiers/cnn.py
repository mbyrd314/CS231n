import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    W1shape = (num_filters, C, filter_size, filter_size)
    self.params['W1'] = weight_scale * np.random.randn(*W1shape)
    self.params['b1'] = np.zeros(num_filters)
    
    #The shift and scale parameters for the spatial batchnorm layer
    #self.params['gamma1'] = np.ones(C)
    #self.params['beta1'] = np.zeros(C)
    
    #The batchnorm parameter dicts for the batchnorm layers
    self.bn_params = [{'mode':'train'} for i in range(2)]
    
    #Height and width after the conv layer assuming padding=(filter_size-1)/2
    #and stride=1, as described below in the loss function
    pad = (filter_size - 1) / 2
    stride = 1
    H_prime = 1 + (H + 2 * pad - filter_size) / stride
    W_prime = 1 + (W + 2 * pad - filter_size) / stride
    #print('H_prime', H_prime)
    #print('W_prime', W_prime)
    
    #Height and width after the pool layer assuming no padding and 2x2 pooling
    H_pool = 1 + (H_prime - 2) / 2
    W_pool = 1 + (W_prime - 2) / 2
    #print('H_pool', H_pool)
    #print('W_pool', W_pool)
    
    W2shape = (num_filters * H_pool * W_pool, hidden_dim)
    self.params['W2'] = weight_scale * np.random.randn(*W2shape)
    self.params['b2'] = np.zeros(hidden_dim)
    
    #The shift and scale parameters for the affine batchnorm layer
    self.params['gamma2'] = np.ones(hidden_dim)
    self.params['beta2'] = np.zeros(hidden_dim)
    
    self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b3'] = np.zeros(num_classes)
    
    #print('W1', self.params['W1'].shape)
    #print('W2', self.params['W2'].shape)
    #print('b2', self.params['b2'].shape)
    #print('W3', self.params['W3'].shape)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    #print('X', X.shape)
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    #Unpacking weights and biases
    W1 = self.params['W1']
    b1 = self.params['b1']
    gamma1 = self.params['gamma1']
    beta1 = self.params['beta1']
    #sbn_param = self.bn_params[0]
    W2 = self.params['W2']
    b2 = self.params['b2']
    gamma2 = self.params['gamma2']
    beta2 = self.params['beta2']
    bn_param = self.bn_params[1]
    W3 = self.params['W3']
    b3 = self.params['b3']
    reg = self.reg
    
    #The convolutional layer, followed by the ReLU and pooling layers
    conv_out, conv_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    
    #A spatial batchnorm layer for the convolutional layer
    #sbn_out, sbn_cache = spatial_batchnorm_forward(conv_out, gamma1, beta1, sbn_param)
    
    #The fully connected hidden layer followed by a ReLU nonlinearity
    affine_out, affine_cache = affine_relu_forward(conv_out, W2, b2)
    
    #A batchnorm layer to speed up training
    bn_out, bn_cache = batchnorm_forward(affine_out, gamma2, beta2, bn_param)
    
    #The final fully connected layer
    scores, score_cache = affine_forward(bn_out, W3, b3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dscores = softmax_loss(scores, y)
    reg_loss = .5 * reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))
    loss += reg_loss #Adding L2 regularization loss
    
    #Backpropagation of the final layer
    dout, grads['W3'], grads['b3'] = affine_backward(dscores, score_cache)
    grads['W3'] += reg * W3
    
    #Backpropagation of the affine batchnorm layer
    dout, grads['gamma2'], grads['beta2'] = batchnorm_backward(dout, bn_cache)
    
    #Backpropagation of the affine hidden layer
    dout, grads['W2'], grads['b2'] = affine_relu_backward(dout, affine_cache)
    grads['W2'] += reg * W2
    
    #Backpropagation of the spatial batchnorm layer
    #dout, grads['gamma1'], grads['beta1'] = spatial_batchnorm_backward(dout, sbn_cache)
    grads['gamma1'], grads['beta1'] = np.ones(gamma1.shape), np.zeros(beta1.shape)
    
    #Backpropagation of the convolutional layer
    dx, grads['W1'], grads['b1'] = conv_relu_pool_backward(dout, conv_cache)
    grads['W1'] += reg * W1
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass

class FourLayerConvNet(object):
  """
  A four-layer convolutional network with the following architecture:
  
  [conv - relu]*2 - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    W1shape = (num_filters, C, filter_size, filter_size)
    self.params['W1'] = weight_scale * np.random.randn(*W1shape)
    self.params['b1'] = np.zeros(num_filters)
    
    #The shift and scale parameters for the spatial batchnorm layer
    #The second dimension of the output of the conv layer is num_filters
    self.params['gamma1'] = np.ones(num_filters)
    self.params['beta1'] = np.zeros(num_filters)
    
    #The batchnorm parameter dicts for the batchnorm layers
    self.bn_params = [{'mode':'train'} for i in range(3)]
    
    #Height and width after the conv layer assuming padding=(filter_size-1)/2
    #and stride=1, as described below in the loss function
    pad = (filter_size - 1) / 2
    stride = 1
    H_prime = 1 + (H + 2 * pad - filter_size) / stride
    W_prime = 1 + (W + 2 * pad - filter_size) / stride
    #print('H_prime', H_prime)
    #print('W_prime', W_prime)
    
    #The second conv layer has the same filter size and number of filters as the first
    W2shape = (num_filters, num_filters, filter_size, filter_size)
    self.params['W2'] = weight_scale * np.random.randn(*W2shape)
    self.params['b2'] = np.zeros(num_filters)
    
    #The shift and scale parameters for the second spatial batchnorm layer
    self.params['gamma2'] = np.ones(num_filters)
    self.params['beta2'] = np.zeros(num_filters)
    
    #Height and width after the second conv layer
    H_prime2 = 1 + (H_prime + 2 * pad - filter_size) / stride
    W_prime2 = 1 + (W_prime + 2 * pad - filter_size) / stride
    
    #Height and width after the pool layer assuming no padding and 2x2 pooling
    H_pool = 1 + (H_prime2 - 2) / 2
    W_pool = 1 + (W_prime2 - 2) / 2
    
    #The weights and biases for the affine layer after the conv layers
    W3shape = (num_filters * H_pool * W_pool, hidden_dim)
    self.params['W3'] = weight_scale * np.random.randn(*W3shape)
    self.params['b3'] = np.zeros(hidden_dim)
    
    #The shift and scale parameters for the affine batchnorm layer
    self.params['gamma3'] = np.ones(hidden_dim)
    self.params['beta3'] = np.zeros(hidden_dim)
    
    self.params['W4'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b4'] = np.zeros(num_classes)
    
    #print('W1', self.params['W1'].shape)
    #print('W2', self.params['W2'].shape)
    #print('b2', self.params['b2'].shape)
    #print('W3', self.params['W3'].shape)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
      #print(k, v.shape)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    
    #Unpacking some variables
    W1 = self.params['W1']
    b1 = self.params['b1']
    gamma1 = self.params['gamma1']
    beta1 = self.params['beta1']
    sbn1_param = self.bn_params[0]
    W2 = self.params['W2']
    b2 = self.params['b2']
    gamma2 = self.params['gamma2']
    beta2 = self.params['beta2']
    sbn2_param = self.bn_params[1]
    W3 = self.params['W3']
    b3 = self.params['b3']
    gamma3 = self.params['gamma3']
    beta3 = self.params['beta3']
    bn_param = self.bn_params[2]
    W4 = self.params['W4']
    b4 = self.params['b4']
    reg = self.reg
    
    loss, grads = 0, {}
    
    #The first convolutional layer followed by a ReLU nonlinearity
    conv1_out, conv1_cache = conv_relu_forward(X, W1, b1, conv_param)
    #print('X.shape is ', X.shape)
    #print('conv1_out.shape is ', conv1_out.shape)
    
    #The spatial batchnorm after the first convolutional layer
    sbn1_out, sbn1_cache = spatial_batchnorm_forward(conv1_out, gamma1, beta1, sbn1_param)
    
    #The second convolutional layer followed by ReLU and max pooling
    conv2_out, conv2_cache = conv_relu_pool_forward(sbn1_out, W2, b2, conv_param, pool_param)
    
    #The second spatial batchnorm layer
    sbn2_out, sbn2_cache = spatial_batchnorm_forward(conv2_out, gamma2, beta2, sbn2_param)
    
    #The affine hidden layer followed by a ReLU nonlinearity
    affine_out, affine_cache = affine_relu_forward(sbn2_out, W3, b3)
    
    #The batchnorm layer after the affine layer
    bn_out, bn_cache = batchnorm_forward(affine_out, gamma3, beta3, bn_param)
    
    #The class scores for the model
    scores, score_cache = affine_forward(bn_out, W4, b4)
    
    if y is None:
      return scores
    
    loss, dscores = softmax_loss(scores, y)
    reg_loss = .5 * reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2) + np.sum(W4**2))
    loss += reg_loss
    
    #Backpropagating the final affine layer
    dout, grads['W4'], grads['b4'] = affine_backward(dscores, score_cache)
    grads['W4'] += reg * W4
    
    #Backpropagating the batchnorm layer
    dout, grads['gamma3'], grads['beta3'] = batchnorm_backward(dout, bn_cache)
    
    #Backpropagating the hidden affine layer
    dout, grads['W3'], grads['b3'] = affine_relu_backward(dout, affine_cache)
    grads['W3'] += reg * W3
    
    #Backpropagating the second spatial batchnorm layer
    dout, grads['gamma2'], grads['beta2'] = spatial_batchnorm_backward(dout, sbn2_cache)
    
    #Backpropagating the second convolutional layer
    dout, grads['W2'], grads['b2'] = conv_relu_pool_backward(dout, conv2_cache)
    grads['W2'] += reg * W2
    
    #Backpropagating the first spatial batchnorm layer
    dout, grads['gamma1'], grads['beta1'] = spatial_batchnorm_backward(dout, sbn1_cache)
    
    #Backpropagating the first convolutional layer
    dx, grads['W1'], grads['b1'] = conv_relu_backward(dout, conv1_cache)
    grads['W1'] += reg * W1
    
    return loss, grads
    
    
    
    
    
