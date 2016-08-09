import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

#Still need to debug the initialization, since it doesn't work with num_affine>1


class NLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  [conv-relu-conv-relu-2x2 max pool]*N-[affine-relu]*M-affine-softmax
  
  There is the appropriate form of batchnorm between all layers.
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               num_conv=1, num_affine=1, hidden_dim=100, num_classes=10, 
               weight_scale=1e-3, reg=0.0, dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layers
    - filter_size: Size of filters to use in the convolutional layers
    - num_conv: Number of conv-relu-conv-relu-pool layers
    - num_affine: Number of affine-relu layers
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
    self.num_conv = num_conv
    self.num_affine = num_affine
    
    #It is explicitly assumed that there is at least one conv-relu-conv-relu-pool
    #layer and that there is at least one affine-relu layer
    
    #Initialize weights, biases, and other parameters
    C, H, W = input_dim
    
    #The height and width of the input is halved after every pooling layer.
    #More than log2 conv-relu-conv-relu-pool layers would destroy the input.
    if (num_conv > np.log2(H) or num_conv > np.log2(W)):
      print('Error: Too many pooling layers. Output size will be zero.')
    
    #The weights and biases of the first convolutional layer
    W1size = num_filters, C, filter_size, filter_size
    self.params['W1'] = weight_scale * np.random.randn(*W1size)
    self.params['b1'] = np.zeros(num_filters)
    
    self.params['gamma'+str(1)] = np.ones(num_filters)
    self.params['beta'+str(1)] = np.zeros(num_filters)
    
    #The shift and scale parameters of the spatial batchnorm layers.
    #Every conv-relu-conv-pool layer has two gammas and two betas, and all
    #gammas and betas are of size num_filters
    for i in range(num_conv):
      self.params['gamma'+str(2*i+1)] = np.ones(num_filters)
      self.params['beta'+str(2*i+1)] = np.zeros(num_filters)
      self.params['gamma'+str(2*i+2)] = np.ones(num_filters)
      self.params['beta'+str(2*i+2)] = np.zeros(num_filters)
      
    #Setup the pooling and convolutional parameters to define stride lengths and
    #padding. There are 2 convolutional layers for each conv-relu-conv-relu-pool layer,
    #so there are 2*num_conv parameter objects 
    self.pool_params = [{'pool_height':2, 'pool_width':2, 'stride':2} for i in range(num_conv)]
    self.conv_params = [{'stride':1, 'pad':(filter_size-1)/2} for i in range(2*num_conv)]
    
    #Batchnorm parameters to keep track of running means and variances. Every
    #conv-relu-conv-pool layer has two spatial batchnorm layers and every
    #affine relu layer has one, so the total is 2*num_conv + num_affine
    self.bn_params = [{'mode':'train'} for i in range(2*num_conv + num_affine)]
    
    #This is the size of the output of the first convolutional layer. With
    #padding set to (1+filter_size)/2 and a stride length of 1, the output
    #height and width of a convolutional layer are the same as the input height
    #and width
    pad = (filter_size - 1) / 2
    stride = 1
    H_prime = 1 + (H + 2 * pad - filter_size) / stride
    W_prime = 1 + (W + 2 * pad - filter_size) / stride
    #print('pad', pad)
    #print('H_prime', H_prime)
    W2size = num_filters, num_filters, filter_size, filter_size
    self.params['W2'] = weight_scale * np.random.randn(*W2size)
    self.params['b2'] = np.zeros(num_filters)
    
    #This is the size of the output of the first max-pooling layer. 2x2 max
    #pooling reduces the height and width of the input by a factor of 2.
    H_pool = 1 + (H_prime - 2) / 2
    W_pool = 1 + (W_prime - 2) / 2

    
    #Initializing weights and biases for all other conv-relu-conv-pool layers.
    for i in range(1, num_conv):
      Wsize = num_filters, num_filters, filter_size, filter_size
      self.params['W'+str(2*i+1)] = weight_scale * np.random.randn(*Wsize)
      self.params['b'+str(2*i+1)] = np.zeros(num_filters)
      self.params['W'+str(2*i+2)] = weight_scale * np.random.randn(*Wsize)
      self.params['b'+str(2*i+2)] = np.zeros(num_filters)
      H_pool = 1 + (H_pool - 2) / 2
      W_pool = 1 + (W_pool - 2) / 2
      
    #Initializing weights and biases for the first affine-relu layer
    Wshape = (num_filters * H_pool * W_pool, hidden_dim)
    self.params['W'+str(2*num_conv+1)] = weight_scale * np.random.randn(*Wshape)
    self.params['b'+str(2*num_conv+1)] = np.zeros(hidden_dim)
    self.params['gamma'+str(2*num_conv+1)] = np.ones(hidden_dim)
    self.params['beta'+str(2*num_conv+1)] = np.zeros(hidden_dim)
    
    #Initializing weights and biases for all other affine-relu layers
    W_shape = (hidden_dim, hidden_dim)
    for i in range(1, num_affine):
      self.params['W'+str(2*num_conv+i+1)] = weight_scale * np.random.randn(*W_shape)
      self.params['b'+str(2*num_conv+i+1)] = np.zeros(hidden_dim)
      self.params['gamma'+str(2*num_conv+i+1)] = np.ones(hidden_dim)
      self.params['beta'+str(2*num_conv+i+1)] = np.zeros(hidden_dim)
        
    #Initializing weights and biases for the final affine layer
    W_shape = (hidden_dim, num_classes)
    self.params['W'+str(2*num_conv+num_affine+1)] = weight_scale * np.random.randn(*W_shape)
    self.params['b'+str(2*num_conv+num_affine+1)] = np.zeros(num_classes)
    
    #Setting the data type of the parameters
    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
      #print(k, v.shape)
      
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the N-layer convolutional net.
  
    Input / Output: Same as for the TwoLayerNet in fc_net.py
    """
    out, cache = X, []
    reg = self.reg
    num_conv = self.num_conv
    num_affine = self.num_affine
  
    for i in range(num_conv):
      #The conv-relu part of the conv-relu-conv-relu-pool layers, with spatial batchnorm
      W1 = self.params['W'+str(2*i+1)]
      b1 = self.params['b'+str(2*i+1)]
      conv_param1 = self.conv_params[2*i]
      gamma1 = self.params['gamma'+str(2*i+1)]
      beta1 = self.params['beta'+str(2*i+1)]
      sbn_param1 = self.bn_params[2*i]
      #print()
      #print('out', out.shape)
      #print('W1', W1.shape)
      #print('b1', b1.shape)
      #print('gamma1', gamma1.shape)
      #print('beta1', beta1.shape)
      results = conv_relu_batchnorm_forward(out, W1, b1, conv_param1, gamma1, beta1, sbn_param1)
      #print(results)
      out = results[0]
      cache.append(results[1])
    
      #The conv-relu-pool part of the convolutional layers, with spatial batchnorm
      W2 = self.params['W'+str(2*i+2)]
      b2 = self.params['b'+str(2*i+2)]
      conv_param2 = self.conv_params[2*i+1]
      pool_param = self.pool_params[i]
      gamma2 = self.params['gamma'+str(2*i+2)]
      beta2 = self.params['beta'+str(2*i+2)]
      sbn_param2 = self.bn_params[2*i+1]
      results = conv_relu_pool_batchnorm_forward(out, W2, b2, conv_param2, pool_param, 
                                               gamma2, beta2, sbn_param2)
      out = results[0]
      cache.append(results[1])
      #print('out', out.shape)
      #print('W2', W2.shape)
      #print('b2', b2.shape)
      #print('gamma2', gamma2.shape)
      #print('beta2', beta2.shape)
  
    #Each of the num_affine fully-connected layers
    for i in range(num_affine):
      W = self.params['W'+str(2*num_conv+i+1)]
      b = self.params['b'+str(2*num_conv+i+1)]
      gamma = self.params['gamma'+str(2*num_conv+i+1)]
      beta = self.params['beta'+str(2*num_conv+i+1)]
      bn_param = self.bn_params[2*num_conv+i]
      results = affine_relu_batchnorm_forward(out, W, b, gamma, beta, bn_param)
      out = results[0]
      cache.append(results[1])
    
    #The weights and biases for the final affine layer
    Wout = self.params['W'+str(2*num_conv+num_affine+1)]
    bout = self.params['b'+str(2*num_conv+num_affine+1)]
    scores, score_cache = affine_forward(out, Wout, bout)
    
    #print('len(cache) is ', len(cache))
    #print('2*num_conv + num_affine is ', 2*num_conv+num_affine)
    if y is None:
      return scores
  
    loss, grads = None, {}
    
    #Softmax loss  
    loss, dscores = softmax_loss(scores, y)
    reg_loss = 0.0
    for i in range(2*num_conv+num_affine+1):
      reg_loss += .5 * reg * np.sum(self.params['W'+str(i+1)]**2)
    loss += reg_loss #Adding L2 regularization loss
  
    #Backpropagating the final affine layer
    dout, dWout, dbout = affine_backward(dscores, score_cache)
    grads['W'+str(2*num_conv+num_affine+1)] = dWout + reg*Wout
    grads['b'+str(2*num_conv+num_affine+1)] = dbout + reg*bout
  
    #Backpropagating the fully-connected hidden layers
    for i in range(num_affine, 0, -1):
      dout, dW, db, dgamma, dbeta = affine_relu_batchnorm_backward(dout, cache[-1])
      index = 2 * num_conv + i
      W = self.params['W'+str(index)]
      grads['W'+str(index)] = dW + reg * W
      grads['b'+str(index)] = db
      grads['gamma'+str(index)] = dgamma
      grads['beta'+str(index)] = dbeta
      cache.pop()
    
    #Backpropagating the convolutional layers
    for i in range(num_conv, 0, -1):
      dout, dW2, db2, dgamma2, dbeta2 = conv_relu_pool_batchnorm_backward(dout, cache[-1])
      index2 = 2 * i 
      W2 = self.params['W'+str(index2)]
      grads['W'+str(index2)] = dW2 + reg * W2
      grads['b'+str(index2)] = db2
      grads['gamma'+str(index2)] = dgamma2
      grads['beta'+str(index2)] = dbeta2
      cache.pop()
    
      dout, dW1, db1, dgamma1, dbeta1 = conv_relu_batchnorm_backward(dout, cache[-1])
      index1 = 2 * i - 1
      W1 = self.params['W'+str(index1)]
      grads['W'+str(index1)] = dW1 + reg * W1
      grads['b'+str(index1)] = db1
      grads['gamma'+str(index1)] = dgamma1
      grads['beta'+str(index1)] = dbeta1
      cache.pop()
  
    return loss, grads
      
    
