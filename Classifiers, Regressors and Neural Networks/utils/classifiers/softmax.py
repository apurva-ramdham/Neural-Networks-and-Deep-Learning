import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
      Softmax loss function, naive implementation (with loops)

      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
      - W: a numpy array of shape (D, C) containing weights.
      - X: a numpy array of shape (N, D) containing a minibatch of data.
      - y: a numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
      - reg: (float) regularization strength

      Returns a tuple of:
      - loss: (float) the mean value of loss functions over N examples in minibatch.
      - gradient: gradient wrt W, an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.    #
    # Store the loss in loss and the gradient in dW. If you are not careful    #
    # here, it is easy to run into numeric instability. Don't forget the      #
    # regularization!                                        #
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                  #
    #############################################################################
    C = W.shape[1]
    N = X.shape[0]

    #One-hot encoding
    y_new = np.squeeze(np.eye(C)[y.reshape(-1)])

    f = X.dot(W)

    for k in range(C):
      for i in range(N):
        f_j = f[i] - np.max(f[i])
        softmax = np.exp(f_j)/np.sum(np.exp(f_j))
        loss += np.sum(y_new[i,k]*np.log(softmax[k]))
        dW[:,k] += (y_new[i,k] - softmax[k])*X[i]
    

    # Average
    loss /= -N
    dW /= -N

    # Regularization
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W 
    #############################################################################
    #                     END OF YOUR CODE                   #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful    #
    # here, it is easy to run into numeric instability. Don't forget the      #
    # regularization!                                        #
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                  #
    #############################################################################
    C = W.shape[1]
    N = X.shape[0]
    #One-hot encoding
    y_new = np.squeeze(np.eye(C)[y.reshape(-1)])

    f = np.dot(X,W)
    f = f - np.max(f, axis=1, keepdims=True)
    softmax = np.exp(f) / np.exp(f).sum(axis=1, keepdims=True)
    loss = np.sum(y_new*np.log(softmax))
    dW = X.T.dot(y_new - softmax)

    # Average
    loss /= -N
    dW /= -N

    # Regularization
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W 
    #############################################################################
    #                     END OF YOUR CODE                   #
    #############################################################################

    return loss, dW
