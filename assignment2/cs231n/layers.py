import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  N = x.shape[0]
  cache = (x, w, b)
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  x = x.reshape(N, -1) #This can be seen merely by observing the dimensions
  out = x.dot(w) + b #This can also be seen from the dimensions
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  N = x.shape[0]
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  dw = x.reshape(N, -1).T.dot(dout) #This can be determined by seeing which dimensions dw has
  dx = dout.dot(w.T).reshape(x.shape) #This can also be determined by dimensions
  db = np.sum(dout, axis=0) #This is the sum of individual output errors
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  out = np.maximum(0, x)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  dx = dout
  dx[x <= 0] = 0 #The derivative of ReLU is 1 for positive x, scaled by dout, 0 otherwise
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    x_sum = np.sum(x, axis=0) #This is done step by step to make determining backprop
    sample_mean = x_sum / N   #easier to do
    x_diff = x - sample_mean
    x_diff_sqr = x_diff**2
    x_diff_sqr_sum = np.sum(x_diff_sqr, axis=0)
    sample_var = x_diff_sqr_sum / N
    sample_std = np.sqrt(sample_var + eps)
    inv_std = 1./sample_std
    xhat = x_diff * inv_std
    cache = (x_diff, sample_mean, sample_var, sample_std, inv_std, xhat, gamma, eps)
    
    out = gamma * xhat + beta
    
    #The equations for these are given above
    running_mean = momentum * running_mean + (1-momentum) * sample_mean
    running_var = momentum * running_var + (1-momentum) * sample_var
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
    #This particular formula was stated above
    xhat = (x - running_mean) / np.sqrt(running_var + eps)
    out = gamma * xhat + beta
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
  N, D = dout.shape
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################
  (x_diff, sample_mean, sample_var, sample_std, inv_std, xhat, gamma, eps) = cache
  dbeta = dout.sum(axis=0) #Gradients for the beta parameter
  dgamma = np.sum(dout * xhat, axis=0) #Gradients for the gamma parameter
  dxhat = dout * gamma #Gradient w.r.t xhat
  dinv_std = np.sum(dxhat * x_diff, axis=0) #Gradient w.r.t. 1/std(x)
  dstd = -1 / (sample_std**2) * dinv_std #Gradient w.r.t. std(x)
  dvar = 1. / (2*np.sqrt(sample_var + eps)) * dstd #Gradient w.r.t var(x)
  dx_diff_sqr = 1. / N * dvar #Gradient w.r.t. 1/N*sum(x_i-mu)^2
  dx_diff = 2 * x_diff * dx_diff_sqr + dxhat * inv_std #Gradient w.r.t. x-mu
  dmean = -1. / N * np.sum(dx_diff, axis=0) #Gradient w.r.t. mean(x)
  dx = dx_diff + dmean #Gradient w.r.t. inputs

  return dx, dgamma, dbeta
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################
  N,D = dout.shape
  x_diff, sample_mean, sample_var, sample_std, inv_std, xhat, gamma, eps = cache
  #These equations are taken from the paper where batch normalization was first mentioned
  #I do not know why they do not work
  dxhat = gamma * dout
  dvar = np.sum(dxhat * x_diff, axis=0) * (-1./2) * (sample_var + eps)**(-1.5)
  dmu = -np.sum(dxhat, axis=0)*inv_std + dvar*(-2./N)*np.sum(x_diff,axis=0)
  dbeta = np.sum(dout, axis=0)
  dgamma = np.sum(dout * xhat, axis=0)
  dx = 1./N * gamma * inv_std * (N * dout - dbeta) + 1/(sample_var + eps) * np.sum(dout * x_diff,axis=0)
  
  assert dvar.shape[0] == D
  assert dmu.shape[0] == D
  assert dx.shape == dout.shape
  assert dxhat.shape == dout.shape
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    #Generate random numbers between 0 and 1 and zero out the ones less than p
    #This is very similar to the dropout implementation in the course notes
    mask = (np.random.rand(*x.shape) < p) / p
    out = x * mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    #Do nothing. Testing is done with the full network
    out = x
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################
    dx = dout * mask #Gradient of a multiplication step
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  #Defining some variables
  stride = conv_param['stride']
  pad = conv_param['pad']
  N, C, H, W = x.shape
  F = w.shape[0]
  HH, WW = w.shape[2:]
  
  H_prime = 1 + (H + 2 * pad - HH) / stride
  W_prime = 1 + (W + 2 * pad - WW) / stride
  
  #Padding x so that there is an integer number of convolutions
  x = np.pad(x, pad_width=((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant')
  out = np.zeros(N*F*H_prime*W_prime).reshape(N,F,H_prime,W_prime)
  #print(x.shape)
  
  #Very inefficient method that loops over four variables
  for i in range(N):
    for j in range(F):
      for k in range(H_prime):
        for l in range(W_prime):
          H_start = k * stride 
          H_end = H_start + HH
          W_start = l * stride
          W_end = W_start + WW 
          out[i, j, k, l] = np.sum(x[i, :, H_start:H_end, W_start:W_end] * w[j,:,:,:]) + b[j]  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  #Defining some variables
  x, w, b, conv_param = cache
  dx, dw, db = np.zeros(x.shape), np.zeros(w.shape), np.zeros(b.shape)
  pad = conv_param['pad']
  stride = conv_param['stride']
  N, C, H, W = x.shape
  F, H_prime, W_prime = dout.shape[1:]
  HH, WW = w.shape[2:]
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  #Gradient of b is just the sum over dout that has the right dimensions
  db = np.sum(dout, axis=(0,2,3))
  
  #Gradients of w and x are computed by multiplying each value of dout by the
  #appropriately shaped subset of the other variable and summing, i.e. dx is 
  #computed by multiplying each value of dout by some values of w and summing
  for i in range(N):
    for j in range(F):
      for k in range(H_prime):
        for l in range(W_prime):
          H_start = k * stride
          H_end = H_start + HH
          W_start = l * stride
          W_end = W_start + WW
          dx[i, :, H_start:H_end, W_start:W_end] += dout[i,j,k,l] * w[j,:,:,:]
          dw[j, :, :, :] += dout[i,j,k,l] * x[i,:,H_start:H_end,W_start:W_end]
  
  #Cut off the parts that were padded
  dx = dx[:, :, pad:-pad, pad:-pad]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  #Defining some variables
  N, C, H, W = x.shape
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']
  
  #Calculating these is the same as before. It is just assumed there is no padding
  H_prime = 1 + (H - pool_height) / stride
  W_prime = 1 + (W - pool_width) / stride
  
  #Defining a zero matrix of the right shape so that indices can be directly assigned
  out = np.zeros(N*C*H_prime*W_prime).reshape(N,C,H_prime,W_prime)
  
  for i in range(N):
    for j in range(C):
      for k in range(H_prime):
        for l in range(W_prime):
          H_start = k * stride
          H_end = H_start + pool_height
          W_start = l * stride
          W_end = W_start + pool_width
          out[i,j,k,l] = np.amax(x[i,j,H_start:H_end,W_start:W_end])
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  x, pool_param = cache
  
  #Initialize dx at zero so that its values can be directly incremented
  dx = np.zeros(x.shape)
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  #Defining some variables
  N, C, H, W = x.shape
  pool_width = pool_param['pool_width']
  pool_height = pool_param['pool_height']
  stride = pool_param['stride']
  H_prime = 1 + (H - pool_height) / stride
  W_prime = 1 + (W - pool_width) / stride
  
  
  for i in range(N):
    for j in range(C):
      for k in range(H_prime):
        for l in range(W_prime):
          H_start = k * stride
          H_end = H_start + pool_height
          W_start = l * stride
          W_end = W_start + pool_width
          
          #This is the maximum value in the current pooling area
          max_val = np.amax(x[i,j,H_start:H_end,W_start:W_end])
          
          #This gives an array of the shape of the current pooling area with
          #a 1 for the index of the max number and zeros everywhere else
          max_ind = (x == max_val)[i,j,H_start:H_end,W_start:W_end]
          
          #This adds the value of the current gradient to the index of the 
          #maximum value in the pooling area
          dx[i, j, H_start:H_end, W_start:W_end] += dout[i,j,k,l] * max_ind
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = np.zeros_like(x), []

  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  N, C, H, W = x.shape
  for i in range(C):
    x_vec = x[:, i, :, :].reshape(N, -1)
    output, new_cache = batchnorm_forward(x_vec, gamma[i], beta[i], bn_param)
    cache.append(new_cache)
    out[:, i, :, :] += output.reshape(N, H, W) 
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  
  N, C, H, W = dout.shape
  dx, dgamma, dbeta = np.zeros_like(dout), np.zeros(C), np.zeros(C)

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  for i in range(C):
    dout_vec = dout[:, i, :, :].reshape(N, -1)
    delta_x, delta_gamma, delta_beta = batchnorm_backward(dout_vec, cache[i])
    dx[:, i, :, :] += delta_x.reshape(N, H, W)
    dgamma[i] += np.sum(delta_gamma)
    dbeta[i] += np.sum(delta_beta)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta
  

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.amax(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
