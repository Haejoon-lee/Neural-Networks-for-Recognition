import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

############################## Q 2.1 ##############################
# Xavier initialization
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    W, b = None, None

    ##########################
    ##### your code here #####
    ##########################
    W = np.random.uniform(-np.sqrt(6)/np.sqrt(in_size + out_size), np.sqrt(6)/np.sqrt(in_size + out_size), (in_size, out_size))
    b = np.zeros((out_size))
    
    params['W' + name] = W
    params['b' + name] = b

############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = None

    res = 1.0 / (1.0 + np.exp(-x))

    return res

############################## Q 2.2.1 ##############################
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]

    ##########################
    ##### your code here #####
    pre_act = X @ W + b
    post_act = activation(pre_act)
    ##########################

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    res = None

    ##########################
    ##### your code here #####
    max_term = np.max(x, axis=1, keepdims=True)
    exp_term = np.exp(x-max_term)
    res = exp_term / np.sum(exp_term, axis=1, keepdims=True)
    ##########################

    return res

############################## Q 2.2.3 ##############################
# compute total loss and accuracy -> the cross-entropy loss.
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss, acc = None, None

    ##########################
    ##### your code here #####
    prediction = np.argmax(probs, axis=1)

    mask_true = y.astype(np.bool)
    probs_true = probs[mask_true]
    loss = 0.0 - np.sum(np.log(probs_true))

    mask_correct = mask_true[np.arange(prediction.shape[0]), prediction]
    num_correct = np.sum(mask_correct)
    acc = float(num_correct) / mask_correct.shape[0]
    ##########################

    return loss, acc 

############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]

    # do the derivative through activation first
    # (don't forget activation_deriv is a function of post_act)
    # then compute the derivative W, b, and X
    ##########################
    ##### your code here #####
    delta_pre = activation_deriv(post_act) * delta
    grad_W = X.transpose() @ delta_pre
    grad_b = np.sum(delta_pre, axis=0, keepdims=True)
    grad_b = grad_b[0,:] # (1, N) -> (N, )
    grad_X = delta_pre @ W.transpose()
    ##########################

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []
    ##########################
    ##### your code here #####
    num_data = x.shape[0]    
    idxs_random = list(np.random.permutation(num_data))
    
    bp = 0
    while bp < num_data:
        idxs_batch = idxs_random[bp:bp+batch_size]
        batches.append((x[idxs_batch, :], y[idxs_batch, :]))
        bp += batch_size
    ##########################
    return batches
